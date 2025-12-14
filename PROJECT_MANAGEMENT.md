# SceneVM Project Management System

This document describes the project management system for SceneVM applications, which provides a consistent way to save/load application state across all platforms (macOS, iOS, Windows, Linux, WASM).

## Architecture Overview

The project management system has three layers:

1. **App Layer**: Your SceneVMApp implementation handles JSON serialization/deserialization of app state
2. **Wrapper Layer**: Platform-specific wrappers handle file I/O and file path management
3. **FFI Layer**: C FFI bridge for Swift/Xcode integration

## Key Design Principle

**File paths are managed by the wrapper, apps only deal with JSON**

Your app doesn't need to know about file paths, file dialogs, or file I/O. It only needs to serialize/deserialize its state to/from JSON strings. The platform wrapper handles all file operations.

## Enabling Project Management

Add the `project` feature to your app's Cargo.toml:

```toml
[dependencies]
scenevm = { path = "..", features = ["ui", "project"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## Implementing Project Management in Your App

### 1. Define Your Data Model

Create a serializable struct for your app's state:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MyAppData {
    slider_value: f32,
    #[serde(default)]
    settings: Vec<String>,
    // ... other app state
}

impl Default for MyAppData {
    fn default() -> Self {
        Self {
            slider_value: 50.0,
            settings: vec![],
        }
    }
}
```

### 2. Add State to Your App Struct

```rust
struct MyApp {
    workspace: Workspace,
    renderer: UiRenderer,
    // Your app state
    slider_value: f32,
    settings: Vec<String>,
    has_changes: bool,
}
```

### 3. Implement SceneVMApp Project Methods

```rust
impl SceneVMApp for MyApp {
    // ... other trait methods ...
    
    fn save_to_json(&mut self, _vm: &mut SceneVM) -> Option<String> {
        let data = MyAppData {
            slider_value: self.slider_value,
            settings: self.settings.clone(),
        };
        
        match serde_json::to_string_pretty(&data) {
            Ok(json) => {
                self.has_changes = false;
                Some(json)
            }
            Err(e) => {
                eprintln!("Failed to serialize: {}", e);
                None
            }
        }
    }
    
    fn load_from_json(&mut self, vm: &mut SceneVM, json: &str) -> bool {
        match serde_json::from_str::<MyAppData>(json) {
            Ok(data) => {
                self.slider_value = data.slider_value;
                self.settings = data.settings;
                self.has_changes = false;
                
                // Update UI to reflect loaded state
                self.update_ui_from_state();
                
                true
            }
            Err(e) => {
                eprintln!("Failed to deserialize: {}", e);
                false
            }
        }
    }
    
    fn new_project(&mut self, vm: &mut SceneVM) {
        let default = MyAppData::default();
        self.slider_value = default.slider_value;
        self.settings = default.settings;
        self.has_changes = false;
        
        self.update_ui_from_state();
    }
    
    fn has_unsaved_changes(&self) -> bool {
        self.has_changes
    }
}
```

### 4. Track Changes

Mark when state changes:

```rust
UiAction::SliderChanged(id, value) => {
    self.slider_value = value;
    self.has_changes = true; // Mark as modified
}
```

## Platform Integration

### Windows/Linux (Native with File Dialogs)

The native runner can use `FileDialog` for standard file operations:

```rust
use scenevm::prelude::*;

// Open project
if let Some(path) = FileDialog::open("Open Project", &[("SceneVM Project", &["json"])]) {
    if let Ok(json) = std::fs::read_to_string(&path) {
        app.load_from_json(&mut vm, &json);
    }
}

// Save project
if let Some(path) = FileDialog::save("Save Project", "untitled.json", &[("SceneVM Project", &["json"])]) {
    if let Some(json) = app.save_to_json(&mut vm) {
        std::fs::write(&path, json).ok();
    }
}
```

### macOS/iOS (Xcode Wrapper)

The Swift wrapper provides these methods on `SceneVMHandle`:

```swift
// Save current state to JSON string
if let json = sceneVMHandle.saveProject() {
    // Write to file, iCloud, etc.
    try? json.write(to: url, atomically: true, encoding: .utf8)
}

// Load state from JSON string
if let json = try? String(contentsOf: url) {
    let success = sceneVMHandle.loadProject(json: json)
}

// Check for unsaved changes
if sceneVMHandle.hasUnsavedChanges() {
    // Prompt user before closing
}
```

### FFI Functions

The following C FFI functions are available:

- `unified_app_runner_save_project(ptr, out_json, out_len) -> i32`
  - Returns: 0 on success, negative on error
  - Allocates JSON string that must be freed with `unified_app_runner_free_json`

- `unified_app_runner_load_project(ptr, json_data, json_len) -> i32`
  - Returns: 0 on success, negative on error

- `unified_app_runner_free_json(json_ptr, json_len)`
  - Frees JSON string allocated by save_project

- `unified_app_runner_has_unsaved_changes(ptr) -> i32`
  - Returns: 1 if has changes, 0 if not, negative on error

## Example: Complete Implementation

See `ui-demo/src/main.rs` for a complete working example that demonstrates:

- Serializable data model with `UiDemoData`
- State tracking with `has_changes` flag
- Save/load implementation
- UI synchronization after load
- Change tracking on user interactions

## JSON Format

Projects are saved as human-readable JSON:

```json
{
  "slider_value": 75.5,
  "param_sliders": [50.0, 60.0, 70.0, 80.0]
}
```

## Best Practices

1. **Keep it Simple**: Only serialize application state, not derived data
2. **Use Defaults**: Add `#[serde(default)]` for optional fields to maintain backward compatibility
3. **Version Your Format**: Consider adding a version field for future migrations
4. **Validate on Load**: Return `false` from `load_from_json` if the data is invalid
5. **Update UI**: Always sync UI state after loading
6. **Track Changes**: Set `has_changes = true` on every state modification
7. **Clear Changes**: Set `has_changes = false` after successful save/load

## Error Handling

- `save_to_json` returns `None` on serialization errors
- `load_from_json` returns `false` on deserialization errors
- FFI functions return negative error codes (see comments in code)

## Future Enhancements

The `Project` struct in `src/project.rs` provides a richer metadata wrapper if needed:

```rust
pub struct Project {
    pub metadata: ProjectMetadata,  // name, version, timestamps, author
    pub data: serde_json::Value,    // your app data
}
```

Currently apps use raw JSON for simplicity, but you can wrap your data in `Project` for metadata support.
