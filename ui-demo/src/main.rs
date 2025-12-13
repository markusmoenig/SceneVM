use scenevm::Embedded;
use scenevm::app_trait::{SceneVMApp, SceneVMRenderCtx};
use scenevm::{Atom, RenderMode, SceneVM};
use scenevm::{
    Button, ButtonKind, ButtonStyle, Label, LabelRect, NodeId, Slider, SliderStyle, UiAction,
    UiEvent, UiEventKind, UiRenderer, Workspace,
};
use uuid::Uuid;
use vek::Vec4;

struct UiDemo {
    workspace: Workspace,
    renderer: UiRenderer,
    slider_id: Option<Uuid>,
    slider_label_node: Option<NodeId>,
    slider_value: f32,
}

impl UiDemo {
    fn new() -> Self {
        Self {
            workspace: Workspace::new(),
            renderer: UiRenderer::new(),
            slider_id: None,
            slider_label_node: None,
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
        .with_value(self.slider_value);
        self.slider_id = Some(slider.id);
        let slider_node = self.workspace.add_view(slider);
        self.workspace.add_root(slider_node);

        // Add a label for the slider (using fixed position Label, not LabelRect)
        let slider_label = Label::new(
            format!("Value: {:.1}", self.slider_value),
            [250.0, 126.0],
            16.0,
            Vec4::new(0.9, 0.9, 0.95, 1.0),
        )
        .with_layer(10);
        let slider_label_node = self.workspace.add_view(slider_label);
        self.slider_label_node = Some(slider_label_node);
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
                UiAction::ButtonToggled(id, on) => println!("Button toggled: {id} -> {on}"),
                UiAction::SliderChanged(id, value) => {
                    if Some(id) == self.slider_id {
                        self.slider_value = value;
                        // Update just the label node with new text
                        if let Some(label_node) = self.slider_label_node {
                            let updated_label = Label::new(
                                format!("Value: {:.1}", self.slider_value),
                                [250.0, 126.0],
                                16.0,
                                Vec4::new(0.9, 0.9, 0.95, 1.0),
                            )
                            .with_layer(10);
                            self.workspace.update_view(label_node, updated_label);
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
