use scenevm::{Atom, GeoId, Light, RenderMode, SceneVM};
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
    fn init(&mut self, ctx: &mut TheContext) {
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

        self.vm.execute(Atom::SetBackground(Vec4::zero()));

        self.vm.execute(Atom::AddSolid {
            id: tile_id,
            color: [255, 0, 0, 255],
        });
        self.vm.execute(Atom::BuildAtlas);

        /*
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
        });*/

        self.vm.execute(Atom::SetRenderMode(RenderMode::Compute3D));

        let fb_w = 1200.0;
        let fb_h = 700.0;
        let aspect = fb_w / fb_h;

        let cam_pos = Vec3::new(0.0, 0.0, 4.0);
        let cam_dir = Vec3::new(0.0, 0.0, -1.0).normalized();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let right = cam_dir.cross(world_up).normalized();
        let up = right.cross(cam_dir).normalized();

        self.vm.execute(Atom::SetGP0(vek::Vec4::new(
            cam_pos.x,
            cam_pos.y,
            cam_pos.z,
            60f32.to_radians(),
        )));
        self.vm.execute(Atom::SetGP1(vek::Vec4::new(
            cam_dir.x, cam_dir.y, cam_dir.z, aspect,
        )));
        self.vm
            .execute(Atom::SetGP2(vek::Vec4::new(right.x, right.y, right.z, 0.0)));
        self.vm
            .execute(Atom::SetGP3(vek::Vec4::new(up.x, up.y, up.z, 0.0)));

        let size = 2.0;
        let center = Vec4::zero();

        let h = size * 0.5;
        let cx: f32 = center.x;
        let cy = center.y;
        let cz = center.z;

        let verts = vec![
            [cx - h, cy - h, cz - h, 1.0],
            [cx + h, cy - h, cz - h, 1.0],
            [cx + h, cy + h, cz - h, 1.0],
            [cx - h, cy + h, cz - h, 1.0], // back (z-)
            [cx - h, cy - h, cz + h, 1.0],
            [cx + h, cy - h, cz + h, 1.0],
            [cx + h, cy + h, cz + h, 1.0],
            [cx - h, cy + h, cz + h, 1.0], // front (z+)
        ];
        // simple per-vertex UVs (reused)
        let uvs = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        let idx = vec![
            // back  (0,1,2,3) : z = -h  (viewed from -Z side)
            (0, 1, 2),
            (0, 2, 3),
            // front (4,5,6,7) : z = +h  (viewed from +Z side)
            (4, 5, 6),
            (4, 6, 7),
            // left  (0,3,7,4) : x = -h  (viewed from -X side)
            (0, 3, 7),
            (0, 7, 4),
            // right (1,5,6,2) : x = +h  (viewed from +X side)
            (1, 5, 6),
            (1, 6, 2),
            // bottom(0,4,5,1) : y = -h  (viewed from -Y side)
            (0, 4, 5),
            (0, 5, 1),
            // top   (3,2,6,7) : y = +h  (viewed from +Y side)
            (3, 2, 6),
            (3, 6, 7),
        ];

        self.vm.execute(Atom::AddPoly3D {
            id: GeoId::Triangle(0),
            tile_id,
            vertices: verts,
            uvs,
            indices: idx,
        });

        self.vm.execute(Atom::AddLight {
            id: Uuid::new_v4(),
            light: Light::new_pointlight(Vec3::new(0.0, 1.0, -4.0)),
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
