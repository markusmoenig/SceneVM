use scenevm::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UiDemoData {
    slider_value: f32,
    #[serde(default)]
    param_sliders: Vec<f32>,
}

impl Default for UiDemoData {
    fn default() -> Self {
        Self {
            slider_value: 50.0,
            param_sliders: vec![50.0, 60.0, 70.0, 80.0],
        }
    }
}

struct UiDemo {
    workspace: Workspace,
    renderer: UiRenderer,
    slider_value: f32,
    noise_layer: usize,
    param_sliders: Vec<f32>,
    has_changes: bool,
}

impl UiDemo {
    fn new() -> Self {
        let default = UiDemoData::default();
        Self {
            workspace: Workspace::new(),
            renderer: UiRenderer::new(),
            slider_value: default.slider_value,
            noise_layer: 0,
            param_sliders: default.param_sliders,
            has_changes: false,
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
        vm.execute(Atom::AddTile {
            id: test_tile_id,
            width: tile_size as u32,
            height: tile_size as u32,
            frames: vec![pixels],
            material_frames: Some(vec![create_tile_material(
                tile_size as u32,
                tile_size as u32,
            )]),
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
            material_frames: Some(vec![create_tile_material(
                tile_size as u32,
                tile_size as u32,
            )]),
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

        // Add a plain image widget (no border, just the texture)
        let plain_image = Image::new(
            ImageStyle {
                rect: [350.0, 40.0, 64.0, 64.0],
                layer: 10,
            },
            test_tile_id,
        )
        .with_id("plain_image");
        let plain_image_node = self.workspace.add_view(plain_image);
        self.workspace.add_root(plain_image_node);

        // Add a slider below the button
        let slider = Slider::new(
            SliderStyle {
                rect: [40.0, 120.0, 200.0, 32.0],
                track_color: Vec4::new(0.15, 0.15, 0.18, 1.0),
                fill_color: Vec4::new(0.3, 0.5, 0.8, 1.0),
                thumb_color: Vec4::new(0.4, 0.6, 0.9, 1.0), // Blue color for thumb
                thumb_radius: 8.0,
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

        // Add a toolbar with image buttons
        let toolbar = Toolbar::new(
            ToolbarStyle {
                rect: [40.0, 180.0, 600.0, 48.0],
                fill: Vec4::new(0.15, 0.15, 0.18, 1.0),
                border: Vec4::new(0.25, 0.25, 0.28, 1.0),
                radius_px: 6.0,
                border_px: 1.0,
                layer: 10,
            },
            ToolbarOrientation::Horizontal,
        )
        .with_id("main_toolbar")
        .with_spacing(4.0)
        .with_offset(8.0);

        let toolbar_node = self.workspace.add_view(toolbar.clone());
        self.workspace.add_root(toolbar_node);

        // Add 8 image buttons to the toolbar with manual positioning
        let button_size = 32.0;
        let toolbar_x = 40.0;
        let toolbar_y = 180.0;
        let spacing = 4.0;
        let offset = 8.0;
        let extra_gap = 16.0; // Extra spacing after 4th button

        for i in 0..8 {
            // Calculate x position with extra spacing after 4th button
            let x_pos = toolbar_x
                + offset
                + (i as f32 * (button_size + spacing))
                + if i >= 4 { extra_gap } else { 0.0 };
            let y_pos = toolbar_y + 8.0; // Center vertically in toolbar (48px tall, button 32px)

            let btn = Button::new(ButtonStyle {
                rect: [x_pos, y_pos, button_size, button_size],
                fill: Vec4::new(0.2, 0.2, 0.25, 1.0),
                border: Vec4::new(0.3, 0.3, 0.35, 1.0),
                pressed_fill: Vec4::new(0.3, 0.4, 0.5, 1.0),
                pressed_border: Vec4::new(0.4, 0.5, 0.6, 1.0),
                radius_px: 4.0,
                border_px: 1.0,
                layer: 11,
            })
            .with_id(format!("toolbar_btn_{}", i))
            .with_kind(ButtonKind::Momentary)
            .with_tile(test_tile_id)
            .with_tile_offset(2.0);

            let btn_node = self.workspace.add_view(btn);
            self.workspace.add_root(btn_node);
        }

        // Add a separator after the 4th button
        if let Some(toolbar_view) = self.workspace.find_view_mut::<Toolbar>("main_toolbar") {
            // Calculate separator position (after 4th button, in the middle of the extra gap)
            let separator_x =
                toolbar_x + offset + (4.0 * (button_size + spacing)) + (extra_gap / 2.0);
            toolbar_view.add_separator_at(separator_x, None);
        }

        // Add a parameter list below the toolbar
        let mut param_list = ParamList::new(ParamListStyle {
            rect: [40.0, 250.0, 350.0, 200.0], // Widened from 300 to 350 to fit value text
            fill: Vec4::new(0.12, 0.12, 0.15, 1.0),
            border: Vec4::new(0.25, 0.25, 0.28, 1.0),
            radius_px: 6.0,
            border_px: 1.0,
            layer: 10,
        })
        .with_id("param_list")
        .with_item_height(32.0)
        .with_label_width(80.0)
        .with_spacing(8.0)
        .with_label_size(14.0);

        // Create sliders and labels for the parameter list
        let param_slider_width = 180.0; // Slider width, value text appears 8px to the right

        for i in 0..4 {
            let label_text = match i {
                0 => "Speed",
                1 => "Volume",
                2 => "Opacity",
                3 => "Scale",
                _ => "Unknown",
            };

            // Get the position for this widget from the param list
            let widget_rect = param_list.get_widget_rect(i, param_slider_width);

            let slider = Slider::new(
                SliderStyle {
                    rect: widget_rect,
                    track_color: Vec4::new(0.15, 0.15, 0.18, 1.0),
                    fill_color: Vec4::new(0.3, 0.5, 0.8, 1.0),
                    thumb_color: Vec4::new(0.4, 0.6, 0.9, 1.0),
                    thumb_radius: 6.0,
                    track_height: 4.0,
                    layer: 11,
                },
                0.0,
                100.0,
            )
            .with_id(format!("param_slider_{}", i))
            .with_value(50.0 + (i as f32 * 10.0))
            .with_show_value(true)
            .with_value_precision(1)
            .with_value_color(Vec4::new(0.6, 0.6, 0.65, 1.0))
            .with_value_size(12.0);

            let slider_node = self.workspace.add_view(slider);
            param_list.add_item(label_text, slider_node);
            self.workspace.add_root(slider_node);
        }

        let param_list_node = self.workspace.add_view(param_list);
        self.workspace.add_root(param_list_node);

        // Create a popup ParamList for a button
        let mut popup_param_list = ParamList::new(ParamListStyle {
            rect: [0.0, 0.0, 250.0, 150.0], // Position will be set by popup system
            fill: Vec4::new(0.18, 0.18, 0.22, 1.0), // Slightly lighter for better visibility
            border: Vec4::new(0.4, 0.5, 0.7, 1.0),
            radius_px: 6.0,
            border_px: 2.0,
            layer: 100, // High layer for popup
        })
        .with_id("popup_param_list")
        .with_item_height(28.0)
        .with_label_width(70.0)
        .with_spacing(6.0)
        .with_label_size(13.0);

        // Add sliders to the popup
        // Width accounts for: label_width (80) + slider (90) + gap (8) + value text (~30) + padding (16)
        let popup_slider_width = 90.0;
        let mut slider_nodes = Vec::new();
        for i in 0..3 {
            let label_text = match i {
                0 => "Red",
                1 => "Green",
                2 => "Blue",
                _ => "Unknown",
            };

            let widget_rect = popup_param_list.get_widget_rect(i, popup_slider_width);

            let slider = Slider::new(
                SliderStyle {
                    rect: widget_rect,
                    track_color: Vec4::new(0.15, 0.15, 0.18, 1.0),
                    fill_color: Vec4::new(0.3, 0.5, 0.8, 1.0),
                    thumb_color: Vec4::new(0.4, 0.6, 0.9, 1.0),
                    thumb_radius: 5.0,
                    track_height: 3.0,
                    layer: 101,
                },
                0.0,
                255.0,
            )
            .with_id(format!("popup_slider_{}", i))
            .with_value(128.0 + (i as f32 * 20.0))
            .with_show_value(true)
            .with_value_precision(0)
            .with_value_color(Vec4::new(0.7, 0.7, 0.75, 1.0))
            .with_value_size(11.0);

            let slider_node = self.workspace.add_view(slider);
            popup_param_list.add_item(label_text, slider_node);
            slider_nodes.push(slider_node);
        }

        // Add a ButtonGroup to the popup ParamList
        let popup_button_group = ButtonGroup::new(
            "popup_group",
            ButtonGroupStyle {
                rect: [0.0, 0.0, 140.0, 28.0], // Will be positioned by ParamList
                button_width: 44.0,
                button_height: 28.0,
                spacing: 2.0,
                fill: Vec4::new(0.18, 0.18, 0.22, 1.0),
                border: Vec4::new(0.3, 0.3, 0.35, 1.0),
                active_fill: Vec4::new(0.4, 0.5, 0.7, 1.0),
                active_border: Vec4::new(0.5, 0.6, 0.8, 1.0),
                radius_px: 3.0,
                border_px: 1.0,
                layer: 101,
            },
        )
        .with_id("popup_group")
        .with_labels(vec![
            "RGB".to_string(),
            "HSV".to_string(),
            "HEX".to_string(),
        ]);

        let popup_group_node = self.workspace.add_view(popup_button_group);
        popup_param_list.add_item("Mode", popup_group_node);
        slider_nodes.push(popup_group_node);

        let popup_param_list_node = self.workspace.add_view(popup_param_list);

        // Attach all child widgets (sliders and button group) to the popup ParamList
        for child_node in slider_nodes {
            self.workspace.attach(popup_param_list_node, child_node);
        }

        // Create a button that opens the popup
        let popup_button = Button::new(ButtonStyle {
            rect: [450.0, 250.0, 120.0, 44.0],
            fill: Vec4::new(0.25, 0.35, 0.5, 1.0),
            border: Vec4::new(0.4, 0.5, 0.7, 1.0),
            pressed_fill: Vec4::new(0.2, 0.28, 0.4, 1.0),
            pressed_border: Vec4::new(0.35, 0.45, 0.65, 1.0),
            radius_px: 8.0,
            border_px: 2.0,
            layer: 10,
        })
        .with_id("popup_button")
        .with_kind(ButtonKind::Momentary)
        .with_popup(popup_param_list_node, PopupAlignment::Right);

        let popup_button_node = self.workspace.add_view(popup_button);
        self.workspace.add_root(popup_button_node);

        // Add label for popup button
        let popup_button_label = LabelRect::new(
            "Colors",
            [450.0, 250.0, 120.0, 44.0],
            16.0,
            Vec4::new(0.9, 0.9, 0.95, 1.0),
        )
        .with_layer(11);
        let popup_button_label_node = self.workspace.add_view(popup_button_label);
        self.workspace.add_root(popup_button_label_node);

        // Add a ButtonGroup to the toolbar area with textures
        // Toolbar is at [40.0, 180.0, 600.0, 48.0]
        let toolbar_button_group = ButtonGroup::new(
            "toolbar_group",
            ButtonGroupStyle {
                rect: [420.0, 186.0, 200.0, 40.0], // Inside toolbar, right side, vertically centered
                button_width: 60.0,
                button_height: 36.0,
                spacing: 4.0,
                fill: Vec4::new(0.2, 0.2, 0.25, 1.0),
                border: Vec4::new(0.3, 0.3, 0.35, 1.0),
                active_fill: Vec4::new(0.3, 0.5, 0.7, 1.0),
                active_border: Vec4::new(0.4, 0.6, 0.8, 1.0),
                radius_px: 4.0,
                border_px: 1.0,
                layer: 11,
            },
        )
        .with_id("toolbar_group")
        .with_textures(vec![
            Some(test_tile_id),
            Some(pressed_tile_id),
            Some(test_tile_id),
        ]);

        let toolbar_group_node = self.workspace.add_view(toolbar_button_group);
        self.workspace.add_root(toolbar_group_node);

        // === Canvas Demo: Two modes that can be toggled ===

        // Create Main Canvas
        let main_canvas = Canvas::new().with_id("main_canvas").with_visible(true);
        let main_canvas_node = self.workspace.add_view(main_canvas);
        self.workspace.add_root(main_canvas_node);

        // Add widgets to main canvas
        let main_label = LabelRect::new(
            "Main Mode - Press button below to switch",
            [40.0, 520.0, 400.0, 30.0],
            16.0,
            Vec4::new(0.9, 0.9, 0.95, 1.0),
        )
        .with_layer(10);
        let main_label_node = self.workspace.add_view(main_label);
        self.workspace.attach(main_canvas_node, main_label_node);

        // Create Settings Canvas (initially hidden)
        let settings_canvas = Canvas::new().with_id("settings_canvas").with_visible(false);
        let settings_canvas_node = self.workspace.add_view(settings_canvas);
        self.workspace.add_root(settings_canvas_node);

        // Add widgets to settings canvas
        let settings_label = LabelRect::new(
            "Settings Mode - Press button below to switch back",
            [40.0, 520.0, 400.0, 30.0],
            16.0,
            Vec4::new(0.9, 0.9, 0.5, 1.0),
        )
        .with_layer(10);
        let settings_label_node = self.workspace.add_view(settings_label);
        self.workspace
            .attach(settings_canvas_node, settings_label_node);

        let settings_slider = Slider::new(
            SliderStyle {
                rect: [40.0, 560.0, 300.0, 40.0],
                track_color: Vec4::new(0.2, 0.2, 0.25, 1.0),
                fill_color: Vec4::new(0.4, 0.5, 0.6, 1.0),
                thumb_color: Vec4::new(0.5, 0.6, 0.7, 1.0),
                thumb_radius: 12.0,
                track_height: 4.0,
                layer: 10,
            },
            0.0,
            100.0,
        )
        .with_id("settings_slider")
        .with_value(75.0)
        .with_show_value(true);
        let settings_slider_node = self.workspace.add_view(settings_slider);
        self.workspace
            .attach(settings_canvas_node, settings_slider_node);

        // Add a button to toggle between canvases (using TextButton)
        let canvas_toggle_button = TextButton::new(
            ButtonStyle {
                rect: [450.0, 510.0, 150.0, 44.0],
                fill: Vec4::new(0.25, 0.35, 0.5, 1.0),
                border: Vec4::new(0.4, 0.5, 0.7, 1.0),
                pressed_fill: Vec4::new(0.2, 0.28, 0.4, 1.0),
                pressed_border: Vec4::new(0.35, 0.45, 0.65, 1.0),
                radius_px: 8.0,
                border_px: 2.0,
                layer: 10,
            },
            "Switch Mode",
        )
        .with_id("canvas_toggle")
        .with_text_size(14.0);
        let canvas_toggle_node = self.workspace.add_view(canvas_toggle_button);
        self.workspace.add_root(canvas_toggle_node);

        // === Color Wheel Demo ===
        let color_wheel = ColorWheel::new(
            [740.0, 40.0, 180.0, 180.0],   // Position in top-right area
            Vec4::new(1.0, 0.5, 0.2, 1.0), // Initial orange color
        )
        .with_id("demo_color_wheel");

        // Create the atlas tile for the color wheel
        color_wheel.ensure_tile(vm.active_vm_mut());

        let color_wheel_node = self.workspace.add_view(color_wheel);
        self.workspace.add_root(color_wheel_node);

        // Label for color wheel
        let color_wheel_label = LabelRect::new(
            "Color Wheel",
            [740.0, 10.0, 180.0, 25.0],
            14.0,
            Vec4::new(0.7, 0.7, 0.75, 1.0),
        );
        let color_wheel_label_node = self.workspace.add_view(color_wheel_label);
        self.workspace.add_root(color_wheel_label_node);

        // Create a new VM layer for procedural noise shader
        self.noise_layer = vm.add_vm_layer();
        vm.set_active_vm(self.noise_layer);

        // Load the noise shader on this layer
        if let Some(bytes) = Embedded::get("noise_shader.wgsl") {
            if let Ok(src) = std::str::from_utf8(bytes.data.as_ref()) {
                vm.execute(Atom::SetSource2D(src.to_string()));
            }
        }

        // Set viewport rect to a region in the top-right corner (400x300 box)
        vm.execute(Atom::SetViewportRect2D(Some([900.0, 700.0, 400.0, 300.0])));

        // Optional: set brightness via gp0
        vm.execute(Atom::SetGP0(Vec4::new(0.1, 0.0, 0.0, 0.0)));

        // Switch back to layer 0 for normal rendering
        vm.set_active_vm(0);
    }

    fn needs_update(&mut self) -> bool {
        // Always update for animated noise layer
        self.workspace.is_dirty()
        // true
    }

    fn render(&mut self, vm: &mut SceneVM, ctx: &mut dyn SceneVMRenderCtx) {
        // Handle actions and update state
        for action in self.workspace.take_actions() {
            match action {
                UiAction::ButtonPressed(id) => {
                    println!("Button pressed: {id}");

                    // Toggle between canvases
                    if id == "canvas_toggle" {
                        if let Some(main_canvas) =
                            self.workspace.find_view_mut::<Canvas>("main_canvas")
                        {
                            let is_visible = main_canvas.is_visible();
                            main_canvas.set_visible(!is_visible);
                        }
                        if let Some(settings_canvas) =
                            self.workspace.find_view_mut::<Canvas>("settings_canvas")
                        {
                            let is_visible = settings_canvas.is_visible();
                            settings_canvas.set_visible(!is_visible);
                        }
                    }
                }
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
                        self.has_changes = true;
                        // Update just the label text using its string ID
                        if let Some(label) = self.workspace.find_view_mut::<Label>("slider_label") {
                            label.set_text(format!("Value: {:.1}", self.slider_value));
                        }
                    } else if id.starts_with("param_slider_") {
                        // Update param sliders
                        if let Some(idx_str) = id.strip_prefix("param_slider_") {
                            if let Ok(idx) = idx_str.parse::<usize>() {
                                if idx < self.param_sliders.len() {
                                    self.param_sliders[idx] = value;
                                    self.has_changes = true;
                                }
                            }
                        }
                    }
                }
                UiAction::ButtonGroupChanged(name, index) => {
                    println!("Button group '{}' changed to index {}", name, index);
                }
                UiAction::ColorChanged(id, color) => {
                    println!(
                        "Color changed from '{}': RGBA({:.3}, {:.3}, {:.3}, {:.3})",
                        id, color[0], color[1], color[2], color[3]
                    );
                }
                UiAction::Custom { source_id, action } => {
                    println!("Custom action from {}: {}", source_id, action);
                }
            }
        }

        // Set GP0.z to the color wheel's HSV value for shader (ensure we're on layer 0)
        vm.set_active_vm(0);
        if let Some(color_wheel) = self
            .workspace
            .find_view_mut::<ColorWheel>("demo_color_wheel")
        {
            let hsv_value = color_wheel.hsv_value();
            vm.execute(Atom::SetGP0(Vec4::new(0.0, 0.0, hsv_value, 0.0)));
        }

        // Build drawables from workspace
        let text_cache = self.renderer.text_cache();
        let drawables = self.workspace.build(text_cache);
        self.renderer.render(vm.active_vm_mut(), &drawables);

        // Animate the noise layer
        vm.set_active_vm(self.noise_layer);
        let counter = vm.active_vm().animation_counter;
        vm.execute(Atom::SetAnimationCounter(counter + 1));
        vm.set_active_vm(0);

        let _ = ctx.present(vm);
    }

    fn mouse_down(&mut self, _vm: &mut SceneVM, x: f32, y: f32) {
        // Check if click is outside popup system - close all popups if so
        if !self.workspace.is_click_inside_popup_system([x, y]) {
            self.workspace.close_all_popups();
        }

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

    // Project Management Implementation

    fn save_to_json(&mut self, _vm: &mut SceneVM) -> Option<String> {
        let data = UiDemoData {
            slider_value: self.slider_value,
            param_sliders: self.param_sliders.clone(),
        };

        match serde_json::to_string_pretty(&data) {
            Ok(json) => {
                self.has_changes = false;
                Some(json)
            }
            Err(e) => {
                eprintln!("Failed to serialize project: {}", e);
                None
            }
        }
    }

    fn load_from_json(&mut self, vm: &mut SceneVM, json: &str) -> bool {
        match serde_json::from_str::<UiDemoData>(json) {
            Ok(data) => {
                self.slider_value = data.slider_value;
                self.param_sliders = data.param_sliders;
                self.has_changes = false;

                // Update UI to reflect loaded values
                if let Some(slider) = self.workspace.find_view_mut::<Slider>("main_slider") {
                    slider.set_value(self.slider_value);
                }
                if let Some(label) = self.workspace.find_view_mut::<Label>("slider_label") {
                    label.set_text(format!("Value: {:.1}", self.slider_value));
                }

                // Update parameter sliders
                for (i, value) in self.param_sliders.iter().enumerate() {
                    if let Some(slider) = self
                        .workspace
                        .find_view_mut::<Slider>(&format!("param_slider_{}", i))
                    {
                        slider.set_value(*value);
                    }
                }

                // Rebuild the UI
                vm.execute(Atom::SetBackground(Vec4::new(0.08, 0.08, 0.1, 1.0)));

                true
            }
            Err(e) => {
                eprintln!("Failed to deserialize project: {}", e);
                false
            }
        }
    }

    fn new_project(&mut self, vm: &mut SceneVM) {
        let default = UiDemoData::default();
        self.slider_value = default.slider_value;
        self.param_sliders = default.param_sliders;
        self.has_changes = false;

        // Reset UI
        if let Some(slider) = self.workspace.find_view_mut::<Slider>("main_slider") {
            slider.set_value(self.slider_value);
        }
        if let Some(label) = self.workspace.find_view_mut::<Label>("slider_label") {
            label.set_text(format!("Value: {:.1}", self.slider_value));
        }

        for (i, value) in self.param_sliders.iter().enumerate() {
            if let Some(slider) = self
                .workspace
                .find_view_mut::<Slider>(&format!("param_slider_{}", i))
            {
                slider.set_value(*value);
            }
        }

        vm.execute(Atom::SetBackground(Vec4::new(0.08, 0.08, 0.1, 1.0)));
    }

    fn has_unsaved_changes(&self) -> bool {
        self.has_changes
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    scenevm::run_scenevm_app(UiDemo::new())
}
