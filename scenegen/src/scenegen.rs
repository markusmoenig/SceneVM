use scenevm::{Atom, SceneVM, vm::GeoId};
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

    // #[cfg(not(target_arch = "wasm32"))]
    fn init(&mut self, _ctx: &mut TheContext) {
        let tile_id = Uuid::new_v4();

        // if let Some((data, width, height)) = self
        //     .vm
        //     .load_image_rgba(std::path::Path::new("images/logo.png"))
        // {
        //     self.vm.execute(Atom::AddTile {
        //         id: tile_id,
        //         width: width,
        //         height: height,
        //         frames: vec![data],
        //     });
        //     self.vm.execute(Atom::BuildAtlas);
        // }

        self.vm.execute(Atom::SetBackground(Vec4::one()));

        self.vm.execute(Atom::AddSolid {
            id: tile_id,
            color: [255, 0, 0, 255],
        });
        self.vm.execute(Atom::BuildAtlas);
        self.vm.execute(Atom::AddPoly {
            id: GeoId::Unknown(0),
            tile_id: tile_id,
            vertices: vec![
                [100.0, 100.0],
                [300.0, 100.0],
                [300.0, 300.0],
                [100.0, 300.0],
            ],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            indices: vec![(0, 1, 2), (0, 2, 3)],
        });

        // Add a line strip
        self.vm.execute(Atom::AddLineStrip2D {
            id: GeoId::Linedef(1),
            tile_id: tile_id,
            points: vec![[400.0, 100.0], [500.0, 120.0], [560.0, 200.0]],
            width: 1.5,
        });
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
