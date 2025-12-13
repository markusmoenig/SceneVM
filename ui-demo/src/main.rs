use scenevm::Embedded;
use scenevm::app_trait::{SceneVMApp, SceneVMRenderCtx};
use scenevm::{Atom, RenderMode, SceneVM};
use scenevm::{
    Button, ButtonKind, ButtonStyle, UiAction, UiEvent, UiEventKind, UiRenderer, Workspace,
};
use vek::Vec4;

struct UiDemo {
    workspace: Workspace,
    renderer: UiRenderer,
}

impl UiDemo {
    fn new() -> Self {
        Self {
            workspace: Workspace::new(),
            renderer: UiRenderer::new(),
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
        let button = Button::new(ButtonStyle {
            rect: [40.0, 40.0, 180.0, 56.0],
            fill: Vec4::new(0.20, 0.25, 0.35, 1.0),
            border: Vec4::new(0.05, 0.08, 0.12, 1.0),
            pressed_fill: Vec4::new(0.15, 0.18, 0.24, 1.0),
            pressed_border: Vec4::new(0.04, 0.07, 0.10, 1.0),
            radius_norm: 0.15,
            border_norm: 0.015,
            layer: 10,
        })
        .with_kind(ButtonKind::Toggle);
        let node = self.workspace.add_view(button);
        self.workspace.add_root(node);
    }

    fn needs_update(&mut self) -> bool {
        self.workspace.is_dirty()
    }

    fn render(&mut self, vm: &mut SceneVM, ctx: &mut dyn SceneVMRenderCtx) {
        let drawables = self.workspace.build();
        self.renderer.render(vm.active_vm_mut(), &drawables);

        for action in self.workspace.take_actions() {
            match action {
                UiAction::ButtonPressed(id) => println!("Button pressed: {id}"),
                UiAction::ButtonToggled(id, on) => println!("Button toggled: {id} -> {on}"),
            }
        }
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
