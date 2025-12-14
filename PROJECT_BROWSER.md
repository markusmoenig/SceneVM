# SceneVM Project Browser

This document describes how to implement a Procreate-style project gallery/browser at app startup, allowing users to browse and select from their recent projects.

## Architecture Overview

The project browser system consists of several components:

1. **ProjectBrowser Widget**: Grid-based UI widget for displaying project thumbnails
2. **RecentProjects**: Persistent storage for recently opened projects with metadata
3. **Thumbnail Generation**: App-provided thumbnail rendering for visual previews
4. **Platform Integration**: Recent files integration for macOS/iOS

## Components

### 1. ProjectBrowser Widget

A scrollable grid widget that displays project thumbnails with names.

```rust
use scenevm::prelude::*;

let browser = ProjectBrowser::new(ProjectBrowserStyle {
    rect: [0.0, 0.0, 960.0, 600.0],
    background: Vec4::new(0.08, 0.08, 0.1, 1.0),
    border: Vec4::new(0.2, 0.2, 0.25, 1.0),
    border_px: 1.0,
    radius_px: 8.0,
    layer: 0,
    
    // Grid layout
    columns: 3,
    cell_width: 280.0,
    cell_height: 320.0,
    spacing: 20.0,
    padding: 40.0,
    
    // Cell styling
    cell_background: Vec4::new(0.12, 0.12, 0.15, 1.0),
    cell_border: Vec4::new(0.25, 0.25, 0.28, 1.0),
    cell_hover_background: Vec4::new(0.15, 0.18, 0.22, 1.0),
    cell_radius_px: 6.0,
    cell_border_px: 1.0,
    
    thumbnail_padding: 12.0,
    
    text_color: Vec4::new(0.9, 0.9, 0.95, 1.0),
    text_size: 16.0,
})
.with_id("project_browser");
```

### 2. RecentProjects Storage

Manages a list of recently opened projects with metadata and thumbnails.

```rust
use scenevm::prelude::*;
use std::path::PathBuf;

// Load recent projects
let recent_projects = RecentProjects::load(
    &dirs::config_dir()
        .unwrap()
        .join("MyApp")
        .join("recent_projects.json")
).unwrap_or_default();

// Add/update a project
let project = RecentProject::new("My Artwork", PathBuf::from("/path/to/project.json"))
    .with_thumbnail(&thumbnail_rgba, 256, 256);
    
recent_projects.add_or_update(project);

// Save
recent_projects.save(&config_path).ok();

// Get sorted by most recent
for project in recent_projects.sorted_by_recent() {
    println!("{}: {}", project.name, project.path.display());
}
```

### 3. Thumbnail Generation

Apps implement thumbnail generation in the SceneVMApp trait:

```rust
impl SceneVMApp for MyApp {
    fn generate_thumbnail(&mut self, vm: &mut SceneVM) -> Option<(u32, u32, Vec<u8>)> {
        // Render current state to a small texture
        // Return (width, height, rgba_pixels)
        
        // Example: Use VM's readback to capture current frame
        let width = 256;
        let height = 256;
        
        // Resize VM temporarily, render, capture pixels
        // This is app-specific based on your rendering approach
        
        Some((width, height, rgba_pixels))
    }
}
```

### 4. Complete Example

Here's a complete example showing project browser integration:

```rust
use scenevm::prelude::*;
use std::path::PathBuf;

struct MyApp {
    mode: AppMode,
    workspace: Workspace,
    renderer: UiRenderer,
    recent_projects: RecentProjects,
    config_dir: PathBuf,
    current_project_path: Option<PathBuf>,
}

enum AppMode {
    Browser,  // Show project browser
    Editor,   // Editing a project
}

impl MyApp {
    fn new() -> Self {
        let config_dir = dirs::config_dir()
            .unwrap()
            .join("MyApp");
        std::fs::create_dir_all(&config_dir).ok();
        
        let recent_projects = RecentProjects::load(
            &config_dir.join("recent_projects.json")
        ).unwrap_or_default();
        
        Self {
            mode: AppMode::Browser,
            workspace: Workspace::new(),
            renderer: UiRenderer::new(),
            recent_projects,
            config_dir,
            current_project_path: None,
        }
    }
    
    fn setup_browser(&mut self, vm: &mut SceneVM) {
        self.workspace.clear();
        
        // Create browser items from recent projects
        let mut items = Vec::new();
        for (i, project) in self.recent_projects.sorted_by_recent().iter().enumerate() {
            let mut item = ProjectBrowserItem {
                id: project.path.to_string_lossy().to_string(),
                name: project.name.clone(),
                thumbnail_tile: None,
                subtitle: Some(format_time_ago(project.last_opened)),
            };
            
            // Load thumbnail if available
            if let Some(ref thumb_base64) = project.thumbnail_base64 {
                if let Ok((rgba, w, h)) = decode_thumbnail_from_base64(thumb_base64) {
                    let tile_id = uuid::Uuid::new_v4();
                    vm.execute(Atom::AddTile {
                        id: tile_id,
                        width: w,
                        height: h,
                        frames: vec![rgba],
                        material_frames: None,
                    });
                    item.thumbnail_tile = Some(tile_id);
                }
            }
            
            items.push(item);
        }
        
        // Add "New Project" button as first item
        items.insert(0, ProjectBrowserItem {
            id: "new_project".to_string(),
            name: "+ New Project".to_string(),
            thumbnail_tile: None,
            subtitle: None,
        });
        
        let browser = ProjectBrowser::new(ProjectBrowserStyle {
            rect: [0.0, 0.0, 960.0, 600.0],
            background: Vec4::new(0.08, 0.08, 0.1, 1.0),
            border: Vec4::new(0.2, 0.2, 0.25, 1.0),
            border_px: 0.0,
            radius_px: 0.0,
            layer: 0,
            columns: 3,
            cell_width: 280.0,
            cell_height: 320.0,
            spacing: 20.0,
            padding: 40.0,
            cell_background: Vec4::new(0.12, 0.12, 0.15, 1.0),
            cell_border: Vec4::new(0.25, 0.25, 0.28, 1.0),
            cell_hover_background: Vec4::new(0.15, 0.18, 0.22, 1.0),
            cell_radius_px: 6.0,
            cell_border_px: 1.0,
            thumbnail_padding: 12.0,
            text_color: Vec4::new(0.9, 0.9, 0.95, 1.0),
            text_size: 16.0,
        })
        .with_id("project_browser")
        .with_items(items);
        
        let browser_node = self.workspace.add_view(browser);
        self.workspace.add_root(browser_node);
        
        vm.execute(Atom::BuildAtlas);
    }
    
    fn setup_editor(&mut self, vm: &mut SceneVM) {
        self.workspace.clear();
        // Setup your normal editor UI
        // ...
    }
}

impl SceneVMApp for MyApp {
    fn init(&mut self, vm: &mut SceneVM, _size: (u32, u32)) {
        vm.execute(Atom::SetBackground(Vec4::new(0.08, 0.08, 0.1, 1.0)));
        vm.execute(Atom::SetRenderMode(RenderMode::Compute2D));
        
        if let Some(bytes) = Embedded::get("ui_body.wgsl") {
            if let Ok(src) = std::str::from_utf8(bytes.data.as_ref()) {
                vm.execute(Atom::SetSource2D(src.to_string()));
            }
        }
        
        // Start in browser mode
        self.setup_browser(vm);
    }
    
    fn render(&mut self, vm: &mut SceneVM, ctx: &mut dyn SceneVMRenderCtx) {
        // Handle UI actions
        for action in self.workspace.take_actions() {
            if let UiAction::Custom { source_id, action } = action {
                if source_id == "project_browser" && action.starts_with("project_selected:") {
                    let project_id = action.strip_prefix("project_selected:").unwrap();
                    
                    if project_id == "new_project" {
                        // Create new project
                        self.new_project(vm);
                        self.mode = AppMode::Editor;
                        self.setup_editor(vm);
                    } else {
                        // Load selected project
                        if let Ok(json) = std::fs::read_to_string(project_id) {
                            if self.load_from_json(vm, &json) {
                                self.current_project_path = Some(PathBuf::from(project_id));
                                self.mode = AppMode::Editor;
                                self.setup_editor(vm);
                                
                                // Update recent projects
                                if let Some(project) = self.recent_projects.projects.iter_mut()
                                    .find(|p| p.path.to_str() == Some(project_id)) {
                                    project.update_last_opened();
                                }
                                self.recent_projects.save(
                                    &self.config_dir.join("recent_projects.json")
                                ).ok();
                            }
                        }
                    }
                }
            }
        }
        
        // Render UI
        let text_cache = self.renderer.text_cache();
        let drawables = self.workspace.build(text_cache);
        self.renderer.render(vm.active_vm_mut(), &drawables);
        
        let _ = ctx.present(vm);
    }
    
    fn save_to_json(&mut self, vm: &mut SceneVM) -> Option<String> {
        let json = /* serialize your app state */;
        
        // Generate and save thumbnail
        if let Some((w, h, rgba)) = self.generate_thumbnail(vm) {
            if let Some(path) = &self.current_project_path {
                let project = RecentProject::new(
                    path.file_stem()?.to_str()?,
                    path.clone()
                ).with_thumbnail(&rgba, w, h);
                
                self.recent_projects.add_or_update(project);
                self.recent_projects.save(
                    &self.config_dir.join("recent_projects.json")
                ).ok();
            }
        }
        
        Some(json)
    }
    
    fn generate_thumbnail(&mut self, vm: &mut SceneVM) -> Option<(u32, u32, Vec<u8>)> {
        // App-specific thumbnail generation
        // Example: render current state at 256x256
        None // Implement based on your app
    }
}

// Helper function
fn format_time_ago(timestamp: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let diff = now.saturating_sub(timestamp);
    
    match diff {
        0..=59 => "Just now".to_string(),
        60..=3599 => format!("{} min ago", diff / 60),
        3600..=86399 => format!("{} hours ago", diff / 3600),
        86400..=2591999 => format!("{} days ago", diff / 86400),
        _ => "Long ago".to_string(),
    }
}
```

## Platform Integration

### macOS/iOS Recent Files

On macOS/iOS, you can integrate with the system's recent files:

```swift
// In your Swift wrapper
func openProject(url: URL) {
    // Load and pass JSON to Rust
    if let json = try? String(contentsOf: url) {
        let _ = sceneVMHandle.loadProject(json: json)
    }
    
    // Add to recent documents
    NSDocumentController.shared.noteNewRecentDocumentURL(url)
}

func saveProject(url: URL) {
    // Get JSON from Rust
    if let json = sceneVMHandle.saveProject() {
        try? json.write(to: url, atomically: true, encoding: .utf8)
        
        // Add to recent documents
        NSDocumentController.shared.noteNewRecentDocumentURL(url)
    }
}
```

### Windows/Linux

Use the standard file dialog and maintain your own recent files list in the config directory.

## Best Practices

1. **Thumbnail Size**: Keep thumbnails small (256x256 or smaller) to minimize JSON file size
2. **Max Recent**: Limit to 20-30 recent projects to keep the browser manageable
3. **Lazy Loading**: Only decode thumbnails for visible cells if you have many projects
4. **Error Handling**: Handle missing/deleted project files gracefully
5. **Performance**: Use base64 encoding for thumbnails to keep everything in JSON
6. **Cleanup**: Remove deleted projects from recent list periodically

## UI/UX Tips

- Add a "+" or "New Project" card as the first item
- Show file modification time or "last opened" timestamp
- Support keyboard navigation (arrow keys, Enter to open)
- Add context menu for delete, rename, show in finder
- Show loading state while thumbnails decode
- Add search/filter functionality for many projects
- Consider grid/list view toggle for user preference

## Future Enhancements

- Project tags/categories
- Favorites/starred projects
- Cloud sync integration
- Project templates
- Batch operations (delete multiple, export, etc.)
