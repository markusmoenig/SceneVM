use scenevm::prelude::*;

struct UiDemo {
    workspace: Workspace,
    renderer: UiRenderer,
    slider_value: f32,
}

impl UiDemo {
    fn new() -> Self {
        Self {
            workspace: Workspace::new(),
            renderer: UiRenderer::new(),
            slider_value: 50.0,
        }
    }
}

impl SceneVMApp for UiDemo {
    fn window_title(&self) -> Option<String> {
        Some("SceneVM UI Demo".into())
    }

    fn initial_window_size(&self) -> Option<(u32, u32)> {
        Some((960, 600))
    }

    fn init(&mut self, vm: &mut SceneVM, _size: (u32, u32)) {
        // Simple background and render mode
        vm.execute(Atom::SetBackground(Vec4::new(0.08, 0.08, 0.1, 1.0)));
        vm.execute(Atom::SetRenderMode(RenderMode::Compute2D));
        if let Some(bytes) = Embedded::get("ui_body.wgsl") {
            if let Ok(src) = std::str::from_utf8(bytes.data.as_ref()) {
                vm.execute(Atom::SetSource2D(src.to_string()));
            }
        }

        // Create a test tile with a gradient pattern to test offset
        let test_tile_id = uuid::Uuid::new_v4();
        let tile_size = 64;
        let mut pixels = vec![0u8; tile_size * tile_size * 4];
        for y in 0..tile_size {
            for x in 0..tile_size {
                let idx = (y * tile_size + x) * 4;
                // Create a gradient pattern - red to blue gradient
                pixels[idx] = ((x as f32 / tile_size as f32) * 255.0) as u8; // R
                pixels[idx + 1] = 128; // G
                pixels[idx + 2] = ((y as f32 / tile_size as f32) * 255.0) as u8; // B
                pixels[idx + 3] = 255; // A

                // Add a white border to see the edges clearly
                if x == 0 || x == tile_size - 1 || y == 0 || y == tile_size - 1 {
                    pixels[idx] = 255;
                    pixels[idx + 1] = 255;
                    pixels[idx + 2] = 255;
                }
            }
        }
        // Create material frame with zeros (non-style tile)
        let mat_pixels = vec![0u8; tile_size * tile_size * 4];

        vm.execute(Atom::AddTile {
            id: test_tile_id,
            width: tile_size as u32,
            height: tile_size as u32,
            frames: vec![pixels],
            material_frames: Some(vec![mat_pixels.clone()]),
        });

        // Create a pressed state tile with inverted gradient
        let pressed_tile_id = uuid::Uuid::new_v4();
        let mut pressed_pixels = vec![0u8; tile_size * tile_size * 4];
        for y in 0..tile_size {
            for x in 0..tile_size {
                let idx = (y * tile_size + x) * 4;
                // Inverted gradient - blue to red
                pressed_pixels[idx] = ((y as f32 / tile_size as f32) * 255.0) as u8; // R
                pressed_pixels[idx + 1] = 200; // G
                pressed_pixels[idx + 2] = ((x as f32 / tile_size as f32) * 255.0) as u8; // B
                pressed_pixels[idx + 3] = 255; // A

                // Add a yellow border for pressed state
                if x == 0 || x == tile_size - 1 || y == 0 || y == tile_size - 1 {
                    pressed_pixels[idx] = 255;
                    pressed_pixels[idx + 1] = 255;
                    pressed_pixels[idx + 2] = 0;
                }
            }
        }
        vm.execute(Atom::AddTile {
            id: pressed_tile_id,
            width: tile_size as u32,
            height: tile_size as u32,
            frames: vec![pressed_pixels],
            material_frames: Some(vec![mat_pixels]),
        });

        vm.execute(Atom::BuildAtlas);

        // Add a basic button to the workspace
        let button_rect = [40.0, 40.0, 180.0, 56.0];
        let button = Button::new(ButtonStyle {
            rect: button_rect,
            fill: Vec4::new(0.20, 0.25, 0.35, 1.0),
            border: Vec4::new(0.05, 0.08, 0.12, 1.0),
            pressed_fill: Vec4::new(0.15, 0.18, 0.24, 1.0),
            pressed_border: Vec4::new(0.04, 0.07, 0.10, 1.0),
            radius_px: 12.0,
            border_px: 1.0,
            layer: 10,
        })
        .with_id("toggle_button")
        .with_kind(ButtonKind::Toggle);
        let node = self.workspace.add_view(button);
        self.workspace.add_root(node);

        // Add a centered label inside the button
        let label = LabelRect::new(
            "Toggle Me",
            button_rect,
            18.0,
            Vec4::new(0.9, 0.9, 0.95, 1.0),
        )
        .with_layer(11); // Layer above button
        let label_node = self.workspace.add_view(label);
        self.workspace.add_root(label_node);

        // Add an image button with the test tile and offset
        let image_button = Button::new(ButtonStyle {
            rect: [250.0, 40.0, 64.0, 64.0],
            fill: Vec4::new(0.2, 0.2, 0.25, 1.0),
            border: Vec4::new(0.4, 0.5, 0.7, 1.0),
            pressed_fill: Vec4::new(0.3, 0.4, 0.5, 1.0),
            pressed_border: Vec4::new(0.6, 0.7, 0.9, 1.0),
            radius_px: 8.0,
            border_px: 2.0,
            layer: 10,
        })
        .with_id("image_button")
        .with_kind(ButtonKind::Toggle)
        .with_tile(test_tile_id)
        .with_pressed_tile(pressed_tile_id) // Different tile when toggled
        .with_tile_offset(4.0); // 4px offset inside the button
        let image_button_node = self.workspace.add_view(image_button);
        self.workspace.add_root(image_button_node);

        // Add a slider below the button
        let slider = Slider::new(
            SliderStyle {
                rect: [40.0, 120.0, 200.0, 32.0],
                track_color: Vec4::new(0.15, 0.15, 0.18, 1.0),
                fill_color: Vec4::new(0.3, 0.5, 0.8, 1.0),
                thumb_color: Vec4::new(0.4, 0.6, 0.9, 1.0), // Blue color for thumb
                thumb_radius: 12.0,
                track_height: 6.0,
                layer: 10,
            },
            0.0,
            100.0,
        )
        .with_id("main_slider")
        .with_value(self.slider_value);
        let slider_node = self.workspace.add_view(slider);
        self.workspace.add_root(slider_node);

        // Add a label for the slider (using fixed position Label, not LabelRect)
        let slider_label = Label::new(
            format!("Value: {:.1}", self.slider_value),
            [250.0, 126.0],
            16.0,
            Vec4::new(0.9, 0.9, 0.95, 1.0),
        )
        .with_id("slider_label")
        .with_layer(10);
        let slider_label_node = self.workspace.add_view(slider_label);
        self.workspace.add_root(slider_label_node);
    }

    fn needs_update(&mut self) -> bool {
        self.workspace.is_dirty()
    }

    fn render(&mut self, vm: &mut SceneVM, ctx: &mut dyn SceneVMRenderCtx) {
        // Handle actions and update state
        for action in self.workspace.take_actions() {
            match action {
                UiAction::ButtonPressed(id) => println!("Button pressed: {id}"),
                UiAction::ButtonToggled(id, on) => {
                    println!("Button toggled: {id} -> {on}");

                    // Example: sync image_button state with toggle_button
                    if id == "toggle_button" {
                        if let Some(img_btn) =
                            self.workspace.find_view_mut::<Button>("image_button")
                        {
                            img_btn.set_toggled(on);
                        }
                    }
                }
                UiAction::SliderChanged(id, value) => {
                    if id == "main_slider" {
                        self.slider_value = value;
                        // Update just the label text using its string ID
                        if let Some(label) = self.workspace.find_view_mut::<Label>("slider_label") {
                            label.set_text(format!("Value: {:.1}", self.slider_value));
                        }
                    }
                }
            }
        }

        // Build drawables from workspace
        let text_cache = self.renderer.text_cache();
        let drawables = self.workspace.build(text_cache);
        self.renderer.render(vm.active_vm_mut(), &drawables);
        let _ = ctx.present(vm);
    }

    fn mouse_down(&mut self, _vm: &mut SceneVM, x: f32, y: f32) {
        self.workspace.handle_event(&UiEvent {
            kind: UiEventKind::PointerDown,
            pos: [x, y],
            pointer_id: 0,
        });
    }

    fn mouse_up(&mut self, _vm: &mut SceneVM, x: f32, y: f32) {
        self.workspace.handle_event(&UiEvent {
            kind: UiEventKind::PointerUp,
            pos: [x, y],
            pointer_id: 0,
        });
    }

    fn mouse_move(&mut self, _vm: &mut SceneVM, x: f32, y: f32) {
        self.workspace.handle_event(&UiEvent {
            kind: UiEventKind::PointerMove,
            pos: [x, y],
            pointer_id: 0,
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    scenevm::run_scenevm_app(UiDemo::new())
}
