use scenevm::{Atom, GeoId, Light, RenderMode, SceneVM};
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
    fn init(&mut self, ctx: &mut TheContext) {
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
            });
            self.vm.execute(Atom::BuildAtlas);
        }

        self.vm.execute(Atom::SetBackground(Vec4::zero()));
        // self.vm.execute(Atom::AddSolid {
        //     id: tile_id,
        //     color: [255, 0, 0, 255],
        // });
        // self.vm.execute(Atom::BuildAtlas);

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
        let center: Vec4<f32> = Vec4::zero();

        let h = size * 0.5;
        let (cx, cy, cz) = (center.x, center.y, center.z);

        // 24 vertices (4 per face) so each face can have its own UVs
        let verts = vec![
            // -Z (back)
            [cx - h, cy - h, cz - h, 1.0],
            [cx + h, cy - h, cz - h, 1.0],
            [cx + h, cy + h, cz - h, 1.0],
            [cx - h, cy + h, cz - h, 1.0],
            // +Z (front)
            [cx - h, cy - h, cz + h, 1.0],
            [cx + h, cy - h, cz + h, 1.0],
            [cx + h, cy + h, cz + h, 1.0],
            [cx - h, cy + h, cz + h, 1.0],
            // -X (left)
            [cx - h, cy - h, cz - h, 1.0],
            [cx - h, cy + h, cz - h, 1.0],
            [cx - h, cy + h, cz + h, 1.0],
            [cx - h, cy - h, cz + h, 1.0],
            // +X (right)
            [cx + h, cy - h, cz - h, 1.0],
            [cx + h, cy + h, cz - h, 1.0],
            [cx + h, cy + h, cz + h, 1.0],
            [cx + h, cy - h, cz + h, 1.0],
            // -Y (bottom)
            [cx - h, cy - h, cz - h, 1.0],
            [cx - h, cy - h, cz + h, 1.0],
            [cx + h, cy - h, cz + h, 1.0],
            [cx + h, cy - h, cz - h, 1.0],
            // +Y (top)
            [cx - h, cy + h, cz - h, 1.0],
            [cx - h, cy + h, cz + h, 1.0],
            [cx + h, cy + h, cz + h, 1.0],
            [cx + h, cy + h, cz - h, 1.0],
        ];

        // same 0..1 UVs per face (6 faces × 4 verts)
        let face_uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mut uvs = Vec::with_capacity(24);
        for _ in 0..6 {
            uvs.extend_from_slice(&face_uv);
        }

        // two triangles per face, using the 24-vertex layout above
        let idx = vec![
            (0, 1, 2),
            (0, 2, 3), // -Z
            (4, 5, 6),
            (4, 6, 7), // +Z
            (8, 9, 10),
            (8, 10, 11), // -X
            (12, 13, 14),
            (12, 14, 15), // +X
            (16, 17, 18),
            (16, 18, 19), // -Y
            (20, 21, 22),
            (20, 22, 23), // +Y
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
