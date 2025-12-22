#[macro_use]
extern crate tracing;

use std::sync::Arc;

use clap::{Parser, arg};
use actix_web::{
    web::{self, Data},
    App, HttpServer, HttpResponse, Responder,
};
use actix_multipart::{Multipart, form::tempfile::TempFile, form::MultipartForm};
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use once_cell::sync::Lazy;
use tracing_subscriber::filter::EnvFilter;

use koharu_ml::cuda_is_available;
use koharu_runtime::{ensure_dylibs, preload_dylibs};
use koharu::ml::Model as KoharuModel;
use koharu::renderer::Renderer;
use koharu::state::{Document, TextStyle};

// Application state
struct AppState {
    model: Arc<KoharuModel>,
    renderer: Arc<Renderer>,
}

// Resolve app directories similar to the main koharu app
static APP_ROOT: Lazy<std::path::PathBuf> = Lazy::new(|| {
    // Portable mode check: look for .portable file in the same directory as the executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let portable_file = exe_dir.join(".portable");
            if portable_file.exists() {
                return exe_dir.to_path_buf();
            }
        }
    }

    dirs::data_local_dir()
        .map(|path| path.join("Koharu"))
        .unwrap_or(std::path::PathBuf::from("."))
});

static LIB_ROOT: Lazy<std::path::PathBuf> = Lazy::new(|| APP_ROOT.join("libs"));
static MODEL_ROOT: Lazy<std::path::PathBuf> = Lazy::new(|| APP_ROOT.join("models"));

// Initialize logging and environment
fn initialize() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        )
        .init();

    // Hook model cache dir
    koharu_ml::set_cache_dir(MODEL_ROOT.to_path_buf())?;

    Ok(())
}

// Preload necessary data
async fn preload() -> Result<()> {
    ensure_dylibs(LIB_ROOT.to_path_buf()).await?;

    // Preload CUDA dlls if available
    if cuda_is_available() {
        preload_dylibs(LIB_ROOT.to_path_buf())?;

        // Only search DLLs in the custom directory on Windows
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
                if SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32) == 0 {
                    anyhow::bail!(
                        "Failed to set default DLL directories: {}",
                        std::io::Error::last_os_error()
                    );
                }

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

// API request structure
#[derive(Debug, MultipartForm)]
struct TranslateRequest {
    #[multipart(rename = "image")]
    image: TempFile,
    // Add other parameters as needed
}

// API endpoint for image translation
async fn translate_image(
    data: Data<AppState>,
    MultipartForm(form): MultipartForm<TranslateRequest>,
) -> actix_web::Result<impl Responder> {
    // Step 1: Load the image
    let image_data = std::fs::read(form.image.file.path())?;
    let dynamic_image = image::load_from_memory(&image_data)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    let serializable_image = koharu::image::SerializableDynamicImage::from(dynamic_image);

    // Step 2: Create a temporary document
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

    // Step 3: Detect dialogs
    let (text_blocks, segment) = data.model.detect_dialog(&serializable_image).await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    document.text_blocks = text_blocks;
    document.segment = Some(segment);

    // Step 4: OCR
    let text_blocks = data.model.ocr(&serializable_image, &document.text_blocks).await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    document.text_blocks = text_blocks;

    // Step 5: Inpaint
    let segment = document.segment.as_ref()
        .ok_or_else(|| actix_web::error::ErrorInternalServerError("Segment not found"))?;

    // Process the segment mask to only keep areas inside text blocks
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
    let inpainted = data.model.inpaint(&serializable_image, &mask).await
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
    document.inpainted = Some(inpainted);

    // Step 6: Render (this would normally include translation, but we'll skip for now)
    // We need to implement translation here or add it to the pipeline

    // For now, just render without translation
    data.renderer.render(&mut document, None, Default::default())
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    // Step 7: Return the result
    let rendered_image = document.rendered.as_ref()
        .ok_or_else(|| actix_web::error::ErrorInternalServerError("Rendered image not found"))?;

    let mut buffer = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut buffer);
    rendered_image.write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    Ok(HttpResponse::Ok()
        .content_type("image/png")
        .body(buffer))
}

// CLI arguments
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[arg(short = 'd', long, help = "Download dynamic libraries and exit")]
    download: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    initialize()?;
    preload().await?;

    // If download flag is set, exit after preloading
    if cli.download {
        info!("Download completed successfully");
        return Ok(());
    }

    // Load models (use CPU if CUDA is not available)
    let use_cpu = !cuda_is_available();
    let model = Arc::new(KoharuModel::new(use_cpu).await?);
    let renderer = Arc::new(Renderer::new()?);

    // Create app state
    let app_state = Data::new(AppState { model, renderer });

    // Start HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(web::resource("/translate/with-form/image/stream")
                .route(web::post().to(translate_image))
            )
    })
    .bind(("0.0.0.0", 12345))? // Listen on all interfaces
    .run()
    .await?;

    Ok(())
}

