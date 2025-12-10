use crate::{RenderResult, SceneVM, SceneVMResult};

/// Minimal app abstraction to write one SceneVM app for native + wasm.
pub trait SceneVMApp {
    /// Optional preferred initial window size on native (physical pixels). If `None`, a platform default is used.
    fn initial_window_size(&self) -> Option<(u32, u32)> {
        None
    }
    /// Optional window title on native. If `None`, `"SceneVM"` is used.
    fn window_title(&self) -> Option<String> {
        None
    }
    /// Optional target frame rate on native (FPS). If `None`, the runner will poll/redraw as fast as possible.
    fn target_fps(&self) -> Option<f32> {
        None
    }
    /// Called once after the renderer is created and sized.
    fn init(&mut self, _vm: &mut SceneVM, _size: (u32, u32)) {}
    /// Per-frame update hook (e.g. animation).
    fn update(&mut self, _vm: &mut SceneVM) {}
    /// Render hook: call `ctx.present(vm)` to display.
    fn render(&mut self, vm: &mut SceneVM, ctx: &mut dyn SceneVMRenderCtx);
    /// Resize callback with new logical size.
    fn resize(&mut self, _vm: &mut SceneVM, _size: (u32, u32)) {}
    /// Mouse/touch down callback in logical pixels.
    fn mouse_down(&mut self, _vm: &mut SceneVM, _x: f32, _y: f32) {}
    /// Mouse/touch up callback in logical pixels.
    fn mouse_up(&mut self, _vm: &mut SceneVM, _x: f32, _y: f32) {}
    /// Mouse/touch move callback in logical pixels.
    fn mouse_move(&mut self, _vm: &mut SceneVM, _x: f32, _y: f32) {}
    /// Scroll/pan delta (e.g. trackpad or wheel) in logical units.
    fn scroll(&mut self, _vm: &mut SceneVM, _dx: f32, _dy: f32) {}
}

/// Rendering context supplied to `SceneVMApp::render`.
pub trait SceneVMRenderCtx {
    fn size(&self) -> (u32, u32);
    fn present(&mut self, vm: &mut SceneVM) -> SceneVMResult<RenderResult>;
}
