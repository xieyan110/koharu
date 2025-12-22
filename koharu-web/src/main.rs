#[macro_use]
extern crate tracing;

use std::sync::Arc;

use clap::Parser;
use actix_cors::Cors;
use actix_web::{
    web::{self, Data},
    App, HttpServer, HttpResponse, Responder,
};
use actix_multipart::form::{MultipartForm, tempfile::TempFile, text::Text};
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use once_cell::sync::Lazy;
use tracing_subscriber::filter::EnvFilter;

use koharu_ml::{cuda_is_available, llm::ModelId};
use koharu_runtime::{ensure_dylibs, preload_dylibs};
use koharu::ml::Model as KoharuModel;
use koharu::renderer::Renderer;
use koharu::llm::Model as LLMModel;
use koharu::state::Document;

// 应用程序状态结构体，用于在Actix Web应用中共享模型和渲染器
struct AppState {
    model: Arc<KoharuModel>,        // Koharu模型实例，用于处理图像翻译
    llm_model: Arc<LLMModel>,       // LLM模型实例，用于文本翻译
    renderer: Arc<Renderer>,        // 渲染器实例，用于将翻译后的文本渲染到图像上
}

// 解析应用程序目录，类似于主koharu应用
static APP_ROOT: Lazy<std::path::PathBuf> = Lazy::new(|| {
    // 便携式模式检查：在可执行文件所在目录查找.portable文件
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let portable_file = exe_dir.join(".portable");
            if portable_file.exists() {
                return exe_dir.to_path_buf();
            }
        }
    }

    // 如果不是便携式模式，则使用本地数据目录
    dirs::data_local_dir()
        .map(|path| path.join("Koharu"))
        .unwrap_or(std::path::PathBuf::from("."))
});

// 库文件根目录
static LIB_ROOT: Lazy<std::path::PathBuf> = Lazy::new(|| APP_ROOT.join("libs"));
// 模型文件根目录
static MODEL_ROOT: Lazy<std::path::PathBuf> = Lazy::new(|| APP_ROOT.join("models"));

// 初始化日志和环境
fn initialize() -> Result<()> {
    // 初始化日志系统，默认日志级别为INFO
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        )
        .init();

    // 设置模型缓存目录
    koharu_ml::set_cache_dir(MODEL_ROOT.to_path_buf())?;

    Ok(())
}

// 预加载必要的数据和库
async fn preload() -> Result<()> {
    // 确保动态库存在，不存在则下载
    ensure_dylibs(LIB_ROOT.to_path_buf()).await?;

    // 如果CUDA可用，则预加载CUDA DLLs
    if cuda_is_available() {
        preload_dylibs(LIB_ROOT.to_path_buf())?;

        // 在Windows上，仅在自定义目录中搜索DLLs
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::ffi::OsStrExt;
            use windows_sys::Win32::System::LibraryLoader::{
                AddDllDirectory, LOAD_LIBRARY_SEARCH_SYSTEM32, LOAD_LIBRARY_SEARCH_USER_DIRS,
                SetDefaultDllDirectories,
            };

            let wide = LIB_ROOT
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect::<Vec<_>>();

            unsafe {
                // 设置默认DLL搜索目录
                if SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32) == 0 {
                    anyhow::bail!(
                        "Failed to set default DLL directories: {}",
                        std::io::Error::last_os_error()
                    );
                }

                // 将自定义库目录添加到搜索路径
                if AddDllDirectory(wide.as_ptr()).is_null() {
                    anyhow::bail!(
                        "Failed to add DLL directory: {}",
                        std::io::Error::last_os_error()
                    );
                }
            }
        }
    }

    Ok(())
}

// API请求结构，用于接收图像翻译请求
#[derive(Debug, MultipartForm)]
struct TranslateRequest {
    #[multipart(rename = "image")]  // 表单字段名称为image
    image: TempFile,                // 上传的图像文件
    #[multipart(rename = "config")]  // 表单字段名称为config，用于接收JSON配置
    config: Option<Text<String>>,         // 配置字符串，前端可能会传递，但我们会忽略
}

// API端点：图像翻译 - 非流式响应版本
async fn translate_image(
    data: Data<AppState>,          // 应用程序状态，包含模型和渲染器
    MultipartForm(form): MultipartForm<TranslateRequest>,  // 解析后的请求表单数据
) -> actix_web::Result<impl Responder> {
    // 步骤1：加载图像
    let image_data = std::fs::read(form.image.file.path())?;  // 读取临时文件
    debug!("图像文件读取成功，大小为 {} 字节", image_data.len());

    let dynamic_image = image::load_from_memory(&image_data)  // 将图像数据加载为DynamicImage
        .map_err(|e| {
            error!("图像加载失败: {}", e);
            actix_web::error::ErrorInternalServerError(e)
        })?;

    debug!("图像加载成功，尺寸为 {}x{}", dynamic_image.width(), dynamic_image.height());
    let serializable_image = koharu::image::SerializableDynamicImage::from(dynamic_image);  // 转换为可序列化图像格式

    // 步骤2：创建临时文档
    let id = blake3::hash(&image_data).to_hex().to_string();  // 使用blake3哈希生成文档ID
    let (width, height) = serializable_image.0.dimensions();  // 获取图像尺寸
    // 创建文档实例，用于存储处理过程中的所有信息
    let mut document = Document {
        id,
        path: std::path::PathBuf::new(),  // 临时文档，没有实际路径
        name: "temp".to_string(),  // 文档名称为temp
        image: serializable_image.clone(),  // 原始图像
        width,
        height,
        ..Default::default()  // 使用默认值填充其他字段
    };

    // 步骤3：检测对话框
    // 使用模型检测图像中的对话框区域和文本块
    let (text_blocks, segment) = data.model.detect_dialog(&serializable_image).await
        .map_err(|e| {
            error!("对话框检测失败: {}", e);
            actix_web::error::ErrorInternalServerError(e)
        })?;

    debug!("对话框检测成功，找到 {} 个文本块", text_blocks.len());
    document.text_blocks = text_blocks;  // 保存检测到的文本块
    document.segment = Some(segment);  // 保存分割结果

    // 步骤4：OCR识别
    // 对检测到的文本块进行光学字符识别
    let text_blocks = data.model.ocr(&serializable_image, &document.text_blocks).await
        .map_err(|e| {
            error!("OCR识别失败: {}", e);
            actix_web::error::ErrorInternalServerError(e)
        })?;

    debug!("OCR识别成功，识别 {} 个文本块", text_blocks.len());
    document.text_blocks = text_blocks;  // 保存识别后的文本块

    // 步骤5：图像修复(Inpaint)
    // 获取分割结果
    let segment = document.segment.as_ref()
        .ok_or_else(|| actix_web::error::ErrorInternalServerError("Segment not found"))?;

    // 处理分割掩码，只保留文本块内的区域
    let mut segment_data = segment.to_rgba8();
    let (seg_width, seg_height) = segment_data.dimensions();
    // 遍历所有像素
    for y in 0..seg_height {
        for x in 0..seg_width {
            let pixel = segment_data.get_pixel_mut(x, y);
            // 如果像素不是黑色(透明)
            if pixel.0 != [0, 0, 0, 255] {
                let mut inside_any_block = false;
                // 检查是否在任何文本块内
                for block in &document.text_blocks {
                    if x >= block.x as u32
                        && x < (block.x + block.width) as u32
                        && y >= block.y as u32
                        && y < (block.y + block.height) as u32
                    {
                        inside_any_block = true;
                        break;
                    }
                }

                // 如果不在任何文本块内，将像素设为黑色(透明)
                if !inside_any_block {
                    *pixel = image::Rgba([0, 0, 0, 255]);
                }
            }
        }
    }

    // 将处理后的分割数据转换为可序列化图像格式作为修复掩码
    let mask = koharu::image::SerializableDynamicImage::from(DynamicImage::ImageRgba8(segment_data));
    // 使用模型对图像进行修复，填充文本区域
    let inpainted = data.model.inpaint(&serializable_image, &mask).await
        .map_err(|e| {
            error!("图像修复失败: {}", e);
            actix_web::error::ErrorInternalServerError(e)
        })?;

    debug!("图像修复成功");

    document.inpainted = Some(inpainted);  // 保存修复后的图像

    // 步骤6：翻译文本（使用默认的SakuraGalTransl7Bv3_7模型）
    // 确保LLM模型已经准备好，如果未准备好则加载
    if !data.llm_model.ready().await {
        info!("LLM模型尚未加载，开始加载SakuraGalTransl7Bv3_7模型...");
        data.llm_model.load(ModelId::SakuraGalTransl7Bv3_7).await;
        info!("SakuraGalTransl7Bv3_7模型加载完成");
    }

    // 使用LLM模型翻译文本块
    data.llm_model.generate(&mut document).await
        .map_err(|e| {
            error!("文本翻译失败: {}", e);
            actix_web::error::ErrorInternalServerError(e)
        })?;

    debug!("文本翻译成功");


    // 步骤7：渲染翻译后的文本
    data.renderer.render(&mut document, None, Default::default())
        .map_err(|e| {
            error!("图像渲染失败: {}", e);
            actix_web::error::ErrorInternalServerError(e)
        })?;

    debug!("图像渲染成功");


    // 获取渲染后的图像
    let rendered_image = document.rendered.as_ref()
        .ok_or_else(|| {
            error!("渲染后的图像不存在");
            actix_web::error::ErrorInternalServerError("渲染后的图像不存在")
        })?;

    // 将渲染后的图像转换为PNG格式
    let mut buffer = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buffer);
    rendered_image.write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| {
            error!("图像转换失败: {}", e);
            actix_web::error::ErrorInternalServerError(e)
        })?;

    debug!("图像转换成功，PNG大小为 {} 字节", buffer.len());

    // 返回PNG图像响应
    info!("图像处理完成，返回结果图像");
    Ok(HttpResponse::Ok()
        .content_type("image/png")
        .body(buffer))
}

// 将图像转换为PNG格式的辅助函数
fn image_to_png(image: &koharu::image::SerializableDynamicImage) -> Result<Vec<u8>, image::ImageError> {
    let mut buffer = Vec::new();
    image.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)?;
    Ok(buffer)
}

// 构造二进制消息的辅助函数
fn build_message(message_type: u8, data: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(5 + data.len());
    message.push(message_type);
    let size = data.len() as u32;
    message.extend_from_slice(&size.to_be_bytes()); // 大端格式
    message.extend_from_slice(data);
    message
}

// API端点：图像翻译 - 流式响应版本
async fn translate_image_stream(
    data: Data<AppState>,
    MultipartForm(form): MultipartForm<TranslateRequest>,
) -> actix_web::Result<impl Responder> {

        // 加载图像
        let image_data = std::fs::read(form.image.file.path())?;
        let dynamic_image = image::load_from_memory(&image_data)
            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
        let serializable_image = koharu::image::SerializableDynamicImage::from(dynamic_image);

        // 创建临时文档
        let id = blake3::hash(&image_data).to_hex().to_string();
        let (width, height) = serializable_image.0.dimensions();
        let mut document = Document {
            id,
            path: std::path::PathBuf::new(),
            name: "temp".to_string(),
            image: serializable_image.clone(),
            width,
            height,
            ..Default::default()
        };

        // 使用actix-web的流响应
        Ok(HttpResponse::Ok()
            .content_type("application/octet-stream")
            .streaming(futures::stream::once(async move {
                // 步骤3：检测对话框
                let (text_blocks, segment) = match data.model.detect_dialog(&serializable_image).await {
                    Ok(result) => result,
                    Err(e) => {
                        let msg = build_message(2, format!("对话框检测失败: {}", e).as_bytes());
                        error!("对话框检测失败: {}", e);
                        return Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg));
                    }
                };
                document.text_blocks = text_blocks;
                document.segment = Some(segment);

                // 步骤4：OCR识别
                let text_blocks = match data.model.ocr(&serializable_image, &document.text_blocks).await {
                    Ok(result) => result,
                    Err(e) => {
                        let msg = build_message(2, format!("OCR识别失败: {}", e).as_bytes());
                        error!("OCR识别失败: {}", e);
                        return Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg));
                    }
                };
                document.text_blocks = text_blocks;

                // 步骤5：图像修复(Inpaint)
                let segment = match document.segment.as_ref() {
                    Some(seg) => seg,
                    None => {
                        let msg = build_message(2, "分割结果不存在".as_bytes());
                        error!("分割结果不存在");
                        return Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg));
                    }
                };

                let mut segment_data = segment.to_rgba8();
                let (seg_width, seg_height) = segment_data.dimensions();
                for y in 0..seg_height {
                    for x in 0..seg_width {
                        let pixel = segment_data.get_pixel_mut(x, y);
                        if pixel.0 != [0, 0, 0, 255] {
                            let mut inside_any_block = false;
                            for block in &document.text_blocks {
                                if x >= block.x as u32
                                    && x < (block.x + block.width) as u32
                                    && y >= block.y as u32
                                    && y < (block.y + block.height) as u32
                                {
                                    inside_any_block = true;
                                    break;
                                }
                            }
                            if !inside_any_block {
                                *pixel = image::Rgba([0, 0, 0, 255]);
                            }
                        }
                    }
                }

                let mask = koharu::image::SerializableDynamicImage::from(DynamicImage::ImageRgba8(segment_data));
                let inpainted = match data.model.inpaint(&serializable_image, &mask).await {
                    Ok(result) => result,
                    Err(e) => {
                        let msg = build_message(2, format!("图像修复失败: {}", e).as_bytes());
                        error!("图像修复失败: {}", e);
                        return Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg));
                    }
                };
                document.inpainted = Some(inpainted);

                // 步骤6：翻译文本
                if !data.llm_model.ready().await {
                    info!("LLM模型尚未加载，开始加载SakuraGalTransl7Bv3_7模型...");
                    data.llm_model.load(ModelId::SakuraGalTransl7Bv3_7).await;
                    info!("SakuraGalTransl7Bv3_7模型加载完成");
                }

                if let Err(e) = data.llm_model.generate(&mut document).await {
                    let msg = build_message(2, format!("文本翻译失败: {}", e).as_bytes());
                    error!("文本翻译失败: {}", e);
                    return Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg));
                }

                // 步骤7：渲染翻译后的文本
                if let Err(e) = data.renderer.render(&mut document, None, Default::default()) {
                    let msg = build_message(2, format!("图像渲染失败: {}", e).as_bytes());
                    error!("图像渲染失败: {}", e);
                    return Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg));
                }

                // 步骤8：返回结果图像
                match document.rendered.as_ref() {
                    Some(rendered_image) => match image_to_png(rendered_image) {
                        Ok(png_data) => {
                            let msg = build_message(0, &png_data);
                            Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg))
                        }
                        Err(e) => {
                            let msg = build_message(2, format!("图像转换失败: {}", e).as_bytes());
                            error!("图像转换失败: {}", e);
                            Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg))
                        }
                    },
                    None => {
                        let msg = build_message(2, "渲染后的图像不存在".as_bytes());
                        error!("渲染后的图像不存在");
                        Ok::<actix_web::web::Bytes, actix_web::Error>(actix_web::web::Bytes::from(msg))
                    }
                }
            })))
}

// CLI命令行参数
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[arg(short = 'd', long, help = "下载动态库并退出")]
    download: bool,  // -d 或 --download 选项
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();  // 解析CLI命令行参数

    initialize()?;  // 初始化日志和环境
    preload().await?;  // 预加载必要的数据和库

    // 如果设置了download标志，则在预加载后退出
    if cli.download {
        info!("Download completed successfully");
        return Ok(());
    }

    // 加载模型（如果CUDA不可用则使用CPU）
    let use_cpu = !cuda_is_available();
    let model = Arc::new(KoharuModel::new(use_cpu).await?);  // 创建模型实例
    let llm_model = Arc::new(LLMModel::new(use_cpu));  // 创建LLM模型实例，用于文本翻译
    let renderer = Arc::new(Renderer::new()?);  // 创建渲染器实例

    // 默认不加载翻译模型，将在API请求时按需加载

    // 创建应用程序状态
    let app_state = Data::new(AppState { model, llm_model, renderer });

    // 启动HTTP服务器
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())  // 允许所有CORS请求
            .app_data(app_state.clone())  // 共享应用程序状态
            .service(web::resource("/translate/with-form/image/stream")  // API端点路径 (流式响应)
                .route(web::post().to(translate_image_stream))  // POST请求处理函数
            )
            .service(web::resource("/translate/with-form/image")  // API端点路径 (非流式响应)
                .route(web::post().to(translate_image))  // POST请求处理函数
            )
            // 添加旧版 manga-image-translator 支持的API端点
            .service(web::resource("/").route(web::get().to(|| async {
                HttpResponse::Ok().body(r#"<html><body>validTranslators: ['SakuraGalTransl7Bv3_7'],</body></html>"#)
            })))
    })
    .bind(("0.0.0.0", 5003))?  // 绑定到所有接口的5003端口
    .run()
    .await?;

    Ok(())
}

