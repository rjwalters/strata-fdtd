use tauri::Emitter;
use tauri_plugin_updater::UpdaterExt;

/// Open a native file dialog for selecting HDF5 files
/// Returns the selected file path or None if cancelled
#[tauri::command]
async fn open_file_dialog(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let file_path = app
        .dialog()
        .file()
        .add_filter("HDF5 Files", &["h5", "hdf5"])
        .blocking_pick_file();

    Ok(file_path.map(|path| path.to_string()))
}

/// Open a recent file and emit navigation event
/// Triggers the frontend to navigate to the viewer with the specified file
#[tauri::command]
async fn open_recent_file(app: tauri::AppHandle, path: String) -> Result<(), String> {
    // Emit event to frontend with the file path
    app.emit("open-file", serde_json::json!({ "path": path }))
        .map_err(|e| format!("Failed to emit open-file event: {}", e))?;

    // Also emit navigation to viewer route
    app.emit("navigate", serde_json::json!({ "route": "/viewer" }))
        .map_err(|e| format!("Failed to emit navigate event: {}", e))?;

    Ok(())
}

/// Check for available updates
/// Returns update info if available, None otherwise
#[tauri::command]
async fn check_for_updates(app: tauri::AppHandle) -> Result<Option<UpdateInfo>, String> {
    let updater = app.updater().map_err(|e| e.to_string())?;

    match updater.check().await {
        Ok(Some(update)) => Ok(Some(UpdateInfo {
            version: update.version.clone(),
            current_version: update.current_version.clone(),
            body: update.body.clone(),
            date: update.date.map(|d| d.to_string()),
        })),
        Ok(None) => Ok(None),
        Err(e) => Err(format!("Failed to check for updates: {}", e)),
    }
}

/// Download and install an available update
#[tauri::command]
async fn install_update(app: tauri::AppHandle) -> Result<(), String> {
    let updater = app.updater().map_err(|e| e.to_string())?;

    let update = updater
        .check()
        .await
        .map_err(|e| format!("Failed to check for updates: {}", e))?
        .ok_or_else(|| "No update available".to_string())?;

    // Download the update
    let mut downloaded = 0;
    let bytes = update
        .download(
            |chunk_length, content_length| {
                downloaded += chunk_length;
                log::info!(
                    "Downloaded {} of {}",
                    downloaded,
                    content_length.unwrap_or(0)
                );
            },
            || {
                log::info!("Download finished");
            },
        )
        .await
        .map_err(|e| format!("Failed to download update: {}", e))?;

    // Install the update (this will restart the app)
    update
        .install(bytes)
        .map_err(|e| format!("Failed to install update: {}", e))?;

    Ok(())
}

#[derive(serde::Serialize)]
struct UpdateInfo {
    version: String,
    current_version: String,
    body: Option<String>,
    date: Option<String>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Check for updates on startup (in release mode only)
            #[cfg(not(debug_assertions))]
            {
                let handle = app.handle().clone();
                std::thread::spawn(move || {
                    // Wait a bit before checking to let the UI load
                    std::thread::sleep(std::time::Duration::from_secs(3));

                    tauri::async_runtime::block_on(async {
                        match handle.updater() {
                            Ok(updater) => match updater.check().await {
                                Ok(Some(update)) => {
                                    log::info!(
                                        "Update available: {} -> {}",
                                        update.current_version,
                                        update.version
                                    );
                                    // Emit event to frontend to show update notification
                                    let _ = handle.emit(
                                        "update-available",
                                        serde_json::json!({
                                            "version": update.version,
                                            "current_version": update.current_version,
                                            "body": update.body,
                                        }),
                                    );
                                }
                                Ok(None) => {
                                    log::info!("App is up to date");
                                }
                                Err(e) => {
                                    log::warn!("Failed to check for updates: {}", e);
                                }
                            },
                            Err(e) => {
                                log::warn!("Failed to get updater: {}", e);
                            }
                        }
                    });
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            open_file_dialog,
            open_recent_file,
            check_for_updates,
            install_update
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
