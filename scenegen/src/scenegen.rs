use scenevm::SceneVM;

#[cfg(not(target_arch = "wasm32"))]
use scenevm::Atom;

use theframework::prelude::*;

pub struct Circle {
    vm: SceneVM,
}

impl TheTrait for Circle {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            vm: SceneVM::new(100, 100),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn init(&mut self, _ctx: &mut TheContext) {
        if let Some((data, width, height)) = self
            .vm
            .load_image_rgba(std::path::Path::new("images/logo.png"))
        {
            self.vm.execute(Atom::AddTile {
                id: Uuid::new_v4(),
                width: width,
                height: height,
                frames: vec![data],
            });
            self.vm.execute(Atom::BuildAtlas);
        }
    }

    /// Draw a circle in the middle of the window
    fn draw(&mut self, pixels: &mut [u8], ctx: &mut TheContext) {
        self.vm
            .render_frame(pixels, ctx.width as u32, ctx.height as u32);
    }

    /// Touch down event
    fn touch_down(&mut self, _x: f32, _y: f32, _ctx: &mut TheContext) -> bool {
        false
    }

    /// Touch up event
    fn touch_up(&mut self, _x: f32, _y: f32, _ctx: &mut TheContext) -> bool {
        false
    }

    /// Query if the widget needs a redraw
    fn update(&mut self, _ctx: &mut TheContext) -> bool {
        false
    }
}
