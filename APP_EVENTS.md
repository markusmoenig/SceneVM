# App Events System

This document explains how apps communicate with the host/wrapper for file operations and other platform-specific actions.

## Design Philosophy

**Apps emit events, hosts handle them.**

- **App**: Emits `AppEvent::RequestSave` when user clicks Save button
- **Host**: Shows file dialog, writes file, manages recent files
- **Result**: Clean separation, platform-agnostic app code

## How It Works

### 1. App Emits Events

```rust
use scenevm::prelude::*;

struct MyApp {
    workspace: Workspace,
    app_events: AppEventQueue,  // Event queue
    has_changes: bool,
}

impl SceneVMApp for MyApp {
    fn render(&mut self, vm: &mut SceneVM, ctx: &mut dyn SceneVMRenderCtx) {
        // Handle UI actions
        for action in self.workspace.take_actions() {
            match action {
                UiAction::ButtonPressed(id) => {
                    match id.as_str() {
                        "save_btn" => {
                            // Emit event instead of handling directly
                            self.app_events.emit(AppEvent::RequestSave);
                        }
                        "open_btn" => {
                            self.app_events.emit(AppEvent::RequestOpen);
                        }
                        "export_png" => {
                            self.app_events.emit(AppEvent::RequestExport {
                                format: "png".to_string()
                            });
                        }
                        "import_image" => {
                            self.app_events.emit(AppEvent::RequestImport {
                                file_types: vec!["png".to_string(), "jpg".to_string()]
                            });
                        }
                        _ => {}
                    }
                }
                UiAction::SliderChanged(_, _) => {
                    // Mark as dirty, emit state change
                    self.has_changes = true;
                    self.app_events.emit(AppEvent::StateChanged {
                        has_unsaved_changes: true
                    });
                }
                _ => {}
            }
        }

        // Render UI...
        let _ = ctx.present(vm);
    }

    // Host calls this to get events
    fn take_app_events(&mut self) -> Vec<AppEvent> {
        self.app_events.take()
    }
}
```

### 2. Host Handles Events

#### Windows/Linux Runner

```rust
// In your native runner main loop
fn main_loop(app: &mut MyApp, vm: &mut SceneVM) {
    loop {
        // Render frame
        app.render(&mut vm, &mut ctx);

        // Check for app events
        let events = app.take_app_events();
        for event in events {
            match event {
                AppEvent::RequestSave => {
                    handle_save(app, vm);
                }
                AppEvent::RequestOpen => {
                    handle_open(app, vm);
                }
                AppEvent::RequestExport { format } => {
                    handle_export(app, vm, &format);
                }
                AppEvent::RequestImport { file_types } => {
                    handle_import(app, vm, &file_types);
                }
                AppEvent::StateChanged { has_unsaved_changes } => {
                    update_window_title(has_unsaved_changes);
                }
                _ => {}
            }
        }
    }
}

fn handle_save(app: &mut MyApp, vm: &mut SceneVM) {
    use scenevm::FileDialog;

    // Show save dialog
    if let Some(path) = FileDialog::save(
        "Save Project",
        "untitled.scenevm",
        &[("SceneVM Project", &["scenevm"])]
    ) {
        // Get JSON from app
        if let Some(json) = app.save_to_json(vm) {
            // Write to file
            std::fs::write(&path, json).ok();
            
            // Update recent files
            update_recent_files(&path);
        }
    }
}

fn handle_open(app: &mut MyApp, vm: &mut SceneVM) {
    use scenevm::FileDialog;

    // Show open dialog
    if let Some(path) = FileDialog::open(
        "Open Project",
        &[("SceneVM Project", &["scenevm"])]
    ) {
        // Read file
        if let Ok(json) = std::fs::read_to_string(&path) {
            // Load into app
            if app.load_from_json(vm, &json) {
                update_recent_files(&path);
            }
        }
    }
}

fn handle_export(app: &mut MyApp, vm: &mut SceneVM, format: &str) {
    use scenevm::FileDialog;

    let (desc, ext) = match format {
        "png" => ("PNG Image", &["png"][..]),
        "jpg" => ("JPEG Image", &["jpg", "jpeg"][..]),
        _ => return,
    };

    if let Some(path) = FileDialog::save("Export", "untitled", &[(desc, ext)]) {
        // App would have export logic
        // export_to_format(app, vm, format, &path);
    }
}

fn handle_import(app: &mut MyApp, vm: &mut SceneVM, file_types: &[String]) {
    use scenevm::FileDialog;

    // Build file filter from types
    let exts: Vec<&str> = file_types.iter().map(|s| s.as_str()).collect();
    
    if let Some(path) = FileDialog::open("Import", &[("Images", &exts)]) {
        // Read and pass to app
        if let Ok(data) = std::fs::read(&path) {
            // App would have import logic
            // app.import_file(vm, &path, &data);
        }
    }
}
```

#### macOS/iOS (Swift)

```swift
// In your view controller or document class
func renderLoop() {
    sceneVMHandle.render()
    
    // Get app events
    let events = getAppEvents()
    for event in events {
        handleAppEvent(event)
    }
}

func handleAppEvent(_ event: AppEvent) {
    switch event {
    case .requestSave:
        // macOS: Auto-save via NSDocument
        // iOS: Auto-save via UIDocument
        document.save(to: document.fileURL, for: .forOverwriting)
        
    case .requestOpen:
        // macOS: Use NSDocumentController
        NSDocumentController.shared.openDocument(nil)
        
        // iOS: Present document picker
        let picker = UIDocumentPickerViewController(
            forOpeningContentTypes: [.sceneVMProject]
        )
        present(picker, animated: true)
        
    case .requestExport(let format):
        handleExport(format: format)
        
    case .requestImport(let fileTypes):
        handleImport(fileTypes: fileTypes)
        
    case .stateChanged(let hasChanges):
        // Update UI to show unsaved state
        updateTitle(hasUnsavedChanges: hasChanges)
        
    default:
        break
    }
}
```

## Complete Example

### App with File Menu Buttons

```rust
use scenevm::prelude::*;

struct MyApp {
    workspace: Workspace,
    renderer: UiRenderer,
    app_events: AppEventQueue,
    has_changes: bool,
    data: MyData,
}

#[derive(Serialize, Deserialize)]
struct MyData {
    value: f32,
}

impl MyApp {
    fn new() -> Self {
        Self {
            workspace: Workspace::new(),
            renderer: UiRenderer::new(),
            app_events: AppEventQueue::new(),
            has_changes: false,
            data: MyData { value: 50.0 },
        }
    }

    fn setup_ui(&mut self, vm: &mut SceneVM) {
        // File menu toolbar
        let toolbar = Toolbar::new(ToolbarStyle {
            rect: [0.0, 0.0, 960.0, 48.0],
            fill: Vec4::new(0.1, 0.1, 0.12, 1.0),
            border: Vec4::new(0.2, 0.2, 0.22, 1.0),
            radius_px: 0.0,
            border_px: 1.0,
            layer: 100,
        }, ToolbarOrientation::Horizontal)
        .with_id("file_toolbar");
        
        let toolbar_node = self.workspace.add_view(toolbar);
        self.workspace.add_root(toolbar_node);

        // Platform-specific buttons
        #[cfg(not(target_os = "macos"))]
        {
            // Windows/Linux need explicit file buttons
            self.add_file_buttons(vm);
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS uses menu bar, just show project browser button
            self.add_browser_button(vm);
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn add_file_buttons(&mut self, vm: &mut SceneVM) {
        let buttons = vec![
            ("new_btn", "New"),
            ("open_btn", "Open"),
            ("save_btn", "Save"),
            ("save_as_btn", "Save As"),
            ("import_btn", "Import"),
            ("export_btn", "Export"),
        ];

        let mut x = 8.0;
        for (id, label) in buttons {
            let btn = TextButton::new(
                ButtonStyle {
                    rect: [x, 8.0, 80.0, 32.0],
                    fill: Vec4::new(0.15, 0.15, 0.18, 1.0),
                    border: Vec4::new(0.25, 0.25, 0.28, 1.0),
                    pressed_fill: Vec4::new(0.2, 0.2, 0.24, 1.0),
                    pressed_border: Vec4::new(0.3, 0.3, 0.34, 1.0),
                    radius_px: 4.0,
                    border_px: 1.0,
                    layer: 101,
                },
                label
            )
            .with_id(id)
            .with_text_size(14.0);
            
            let node = self.workspace.add_view(btn);
            self.workspace.add_root(node);
            
            x += 88.0;
        }
    }
}

impl SceneVMApp for MyApp {
    fn init(&mut self, vm: &mut SceneVM, _size: (u32, u32)) {
        self.setup_ui(vm);
    }

    fn render(&mut self, vm: &mut SceneVM, ctx: &mut dyn SceneVMRenderCtx) {
        // Handle UI actions
        for action in self.workspace.take_actions() {
            if let UiAction::ButtonPressed(id) = action {
                match id.as_str() {
                    "new_btn" => self.app_events.emit(AppEvent::RequestNew),
                    "open_btn" => self.app_events.emit(AppEvent::RequestOpen),
                    "save_btn" => self.app_events.emit(AppEvent::RequestSave),
                    "save_as_btn" => self.app_events.emit(AppEvent::RequestSaveAs),
                    "import_btn" => self.app_events.emit(AppEvent::RequestImport {
                        file_types: vec!["png".into(), "jpg".into(), "svg".into()]
                    }),
                    "export_btn" => self.app_events.emit(AppEvent::RequestExport {
                        format: "png".into()
                    }),
                    _ => {}
                }
            }
        }

        // Render
        let text_cache = self.renderer.text_cache();
        let drawables = self.workspace.build(text_cache);
        self.renderer.render(vm.active_vm_mut(), &drawables);
        let _ = ctx.present(vm);
    }

    fn take_app_events(&mut self) -> Vec<AppEvent> {
        self.app_events.take()
    }

    fn save_to_json(&mut self, _vm: &mut SceneVM) -> Option<String> {
        serde_json::to_string_pretty(&self.data).ok()
    }

    fn load_from_json(&mut self, _vm: &mut SceneVM, json: &str) -> bool {
        if let Ok(data) = serde_json::from_str(json) {
            self.data = data;
            self.has_changes = false;
            true
        } else {
            false
        }
    }

    fn new_project(&mut self, _vm: &mut SceneVM) {
        self.data = MyData { value: 50.0 };
        self.has_changes = false;
    }

    fn has_unsaved_changes(&self) -> bool {
        self.has_changes
    }
}
```

## Event Reference

| Event | Host Action |
|-------|-------------|
| `RequestSave` | Save to current file (show dialog if no file) |
| `RequestSaveAs` | Show "Save As" dialog, save to new file |
| `RequestOpen` | Show open dialog, load selected file |
| `RequestNew` | Prompt to save, create new project |
| `RequestClose` | Prompt to save, close window/document |
| `RequestExport { format }` | Show export dialog for specified format |
| `RequestImport { file_types }` | Show import dialog for specified types |
| `RequestShowBrowser` | Show project gallery/browser |
| `StateChanged { has_unsaved_changes }` | Update UI (title, save button state) |

## Platform Differences

### Windows/Linux
- ✅ Need explicit Save/Open/Import/Export buttons
- ✅ Use `FileDialog` for all file operations
- ✅ Manually track recent files

### macOS
- ✅ Use system menu bar (File → Save, etc.)
- ✅ `NSDocument` handles auto-save
- ✅ System tracks recent files
- ⚠️ Can still show buttons for quick access

### iOS/iPadOS
- ✅ No file menu
- ✅ Use document picker or `UIDocumentBrowserViewController`
- ✅ Auto-save via `UIDocument`
- ⚠️ Import/Export via share sheet or document picker

## Best Practices

1. **Always emit state changes**: Host needs to know when to enable "Save"
2. **Use conditional compilation**: Different UI for different platforms
3. **Let host handle dialogs**: Don't show dialogs from app code
4. **Trust the host**: App doesn't need to know about file paths
5. **Keep it simple**: Just emit events, host does the rest

## Benefits

- ✅ **Platform-agnostic app code**: No `#[cfg]` in app logic
- ✅ **Clean separation**: UI emits events, host handles files
- ✅ **Easy testing**: Mock the host, test the app
- ✅ **Flexible hosts**: Same app works with different wrappers
- ✅ **No file I/O in Rust**: All file operations in platform code
