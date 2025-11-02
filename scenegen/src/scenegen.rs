use scenevm::{Atom, GeoId, Light, Poly2D, Poly3D, RenderMode, SceneVM};
use theframework::prelude::*;

use vek::Mat4;

pub struct Circle {
    vm: SceneVM,

    matrix: Mat4<f32>,
}

impl TheTrait for Circle {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            vm: SceneVM::new(100, 100),
            matrix: Mat4::identity(),
        }
    }

    // #[cfg(not(target_arch = "wasm32"))]
    fn init(&mut self, _ctx: &mut TheContext) {
        let tile_id = Uuid::new_v4();

        if let Some((data, width, height)) = self
            .vm
            .load_image_rgba(std::path::Path::new("images/logo.png"))
        {
            self.vm.execute(Atom::AddTile {
                id: tile_id,
                width: width,
                height: height,
                frames: vec![data],
                material_frames: None,
            });
            self.vm.execute(Atom::BuildAtlas);
        }

        self.vm.execute(Atom::SetBackground(Vec4::zero()));
        // self.vm.execute(Atom::AddSolid {
        //     id: tile_id,
        //     color: [255, 0, 0, 255],
        // });
        self.vm.execute(Atom::BuildAtlas);

        self.vm.execute(Atom::AddPoly {
            poly: Poly2D::poly(
                GeoId::Unknown(0),
                tile_id,
                vec![
                    [100.0, 100.0],
                    [300.0, 100.0],
                    [300.0, 300.0],
                    [100.0, 300.0],
                ],
                vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                vec![(0, 1, 2), (0, 2, 3)],
            ),
        });

        // Add a line strip
        self.vm.execute(Atom::AddLineStrip2D {
            id: GeoId::Linedef(1),
            tile_id: tile_id,
            points: vec![[400.0, 100.0], [500.0, 120.0], [560.0, 200.0]],
            width: 1.5,
            material_id: None,
        });

        // self.vm.execute(Atom::SetRenderMode(RenderMode::Compute3D));

        self.vm.execute(Atom::AddPoly3D {
            poly: Poly3D::cube(GeoId::Unknown(0), tile_id, Vec3::zero(), 2.0),
        });

        self.vm.execute(Atom::AddLight {
            id: GeoId::Light(0),
            light: Light::new_pointlight(Vec3::new(0.0, 1.0, -4.0)),
        });

        // Enable bump mapping
        self.vm
            .execute(Atom::SetGP8(vek::Vec4::new(1.0, 0.0, 0.0, 0.0)));

        // Enable bump mapping
        self.vm.execute(Atom::SetGP9(vek::Vec4::new(
            1.0 / 4096.0,
            1.0 / 4096.0,
            0.0,
            1.0,
        )));

        // self.vm.execute(Atom::SetCamera3D {
        //     camera: Camera3D::iso(),
        // });
    }

    /// Draw a circle in the middle of the window
    fn draw(&mut self, pixels: &mut [u8], ctx: &mut TheContext) {
        // Rotate a bit every frame to see the cube spinning (angles in radians per frame)
        let rot = Mat4::<f32>::rotation_y(0.02) * Mat4::<f32>::rotation_x(0.01);
        self.matrix = rot * self.matrix;
        self.vm.execute(Atom::SetTransform3D(self.matrix));

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
