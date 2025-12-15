use tauri::Emitter;

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            open_file_dialog,
            open_recent_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
