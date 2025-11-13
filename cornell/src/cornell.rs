use scenevm::{Atom, GeoId, Light, Poly3D, RenderMode, SceneVM};
use theframework::prelude::*;
use uuid::Uuid;
use vek::{Vec3, Vec4};

pub struct CornellBox {
    vm: SceneVM,
}

impl TheTrait for CornellBox {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {
            vm: SceneVM::new(100, 100),
        }
    }

    fn init(&mut self, _ctx: &mut TheContext) {
        // Create unique IDs for our materials
        let red_wall_id = Uuid::new_v4();
        let green_wall_id = Uuid::new_v4();
        let white_wall_id = Uuid::new_v4();
        let light_id = Uuid::new_v4();
        let cube1_id = Uuid::new_v4();
        let sphere_id = Uuid::new_v4();
        let metal_id = Uuid::new_v4();
        let glass_id = Uuid::new_v4();

        // Create solid color materials for walls
        self.vm.execute(Atom::AddSolid {
            id: red_wall_id,
            color: [200, 50, 50, 255], // Red
        });
        self.vm.execute(Atom::AddSolid {
            id: green_wall_id,
            color: [50, 200, 50, 255], // Green
        });
        self.vm.execute(Atom::AddSolid {
            id: white_wall_id,
            color: [200, 200, 200, 255], // White
        });
        self.vm.execute(Atom::AddSolid {
            id: light_id,
            color: [255, 255, 255, 255], // White light
        });
        self.vm.execute(Atom::AddSolid {
            id: cube1_id,
            color: [180, 180, 220, 255], // Light blue
        });
        self.vm.execute(Atom::AddSolid {
            id: sphere_id,
            color: [220, 180, 180, 255], // Light pink
        });

        // Create solid materials with material properties using AddSolidWithMaterial
        // Material properties: RGBA = roughness/metallic/opacity/emission
        self.vm.execute(Atom::AddSolidWithMaterial {
            id: metal_id,
            color: [150, 150, 170, 255], // Metallic gray color
            material: [50, 255, 255, 0], // Low roughness, high metallic, full opacity, no emission
        });

        self.vm.execute(Atom::AddSolidWithMaterial {
            id: glass_id,
            color: [200, 230, 255, 128], // Semi-transparent blue glass color
            material: [10, 0, 128, 0], // Very low roughness, non-metallic, semi-transparent, no emission
        });

        self.vm.execute(Atom::BuildAtlas);

        // Cornell box dimensions
        let box_size = 10.0;
        let half_size = box_size / 2.0;

        // Create Cornell box walls as individual polygons (hollow box)

        // Back wall (white) - facing inward (positive Z)
        let back_wall = Poly3D::poly(
            GeoId::Unknown(0),
            white_wall_id,
            vec![
                [-half_size, -half_size, half_size, 1.0],
                [half_size, -half_size, half_size, 1.0],
                [half_size, half_size, half_size, 1.0],
                [-half_size, half_size, half_size, 1.0],
            ],
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![(0, 1, 2), (0, 2, 3)],
        );
        self.vm.execute(Atom::AddPoly3D { poly: back_wall });

        // Left wall (red) - facing inward (positive X)
        let left_wall = Poly3D::poly(
            GeoId::Unknown(1),
            red_wall_id,
            vec![
                [-half_size, -half_size, -half_size, 1.0],
                [-half_size, -half_size, half_size, 1.0],
                [-half_size, half_size, half_size, 1.0],
                [-half_size, half_size, -half_size, 1.0],
            ],
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![(0, 1, 2), (0, 2, 3)],
        );
        self.vm.execute(Atom::AddPoly3D { poly: left_wall });

        // Right wall (green) - facing inward (negative X)
        let right_wall = Poly3D::poly(
            GeoId::Unknown(2),
            green_wall_id,
            vec![
                [half_size, -half_size, half_size, 1.0],
                [half_size, -half_size, -half_size, 1.0],
                [half_size, half_size, -half_size, 1.0],
                [half_size, half_size, half_size, 1.0],
            ],
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![(0, 1, 2), (0, 2, 3)],
        );
        self.vm.execute(Atom::AddPoly3D { poly: right_wall });

        // Floor (white) - facing inward (positive Y)
        let floor = Poly3D::poly(
            GeoId::Unknown(3),
            white_wall_id,
            vec![
                [-half_size, -half_size, -half_size, 1.0],
                [half_size, -half_size, -half_size, 1.0],
                [half_size, -half_size, half_size, 1.0],
                [-half_size, -half_size, half_size, 1.0],
            ],
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![(0, 1, 2), (0, 2, 3)],
        );
        self.vm.execute(Atom::AddPoly3D { poly: floor });

        // Ceiling (white) - facing inward (negative Y)
        let ceiling = Poly3D::poly(
            GeoId::Unknown(4),
            white_wall_id,
            vec![
                [half_size, half_size, half_size, 1.0],
                [-half_size, half_size, half_size, 1.0],
                [-half_size, half_size, -half_size, 1.0],
                [half_size, half_size, -half_size, 1.0],
            ],
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![(0, 1, 2), (0, 2, 3)],
        );
        self.vm.execute(Atom::AddPoly3D { poly: ceiling });

        // Add a cube and a sphere inside the Cornell box
        let cube_width = 3.0;
        let cube_height = 4.0;
        let cube_depth = 3.0;
        let sphere_radius = 1.5;

        // Cube (metallic, rotated 45 degrees)
        let cube_transform = vek::Mat4::rotation_y(std::f32::consts::FRAC_PI_4);
        let mut cube_poly = Poly3D::box_(
            GeoId::Unknown(5),
            metal_id,
            Vec3::new(-2.5, -half_size + cube_height / 2.0, -1.0),
            cube_width,
            cube_height,
            cube_depth,
        );
        cube_poly.vertices = cube_poly
            .vertices
            .iter()
            .map(|v| {
                let mut vec = vek::Vec4::from(*v);
                vec = cube_transform * vec;
                [vec.x, vec.y, vec.z, vec.w]
            })
            .collect();
        self.vm.execute(Atom::AddPoly3D { poly: cube_poly });

        // Sphere (glass)
        let sphere_poly = Poly3D::sphere(
            GeoId::Unknown(6),
            glass_id,
            Vec3::new(2.5, -half_size + sphere_radius, -1.5),
            sphere_radius,
            16, // stacks
            16, // slices
        );
        self.vm.execute(Atom::AddPoly3D { poly: sphere_poly });

        // Add area light at the top
        self.vm.execute(Atom::AddLight {
            id: GeoId::Light(0),
            light: Light::new_pointlight(Vec3::new(0.0, half_size - 1.0, 0.0))
                .with_color(Vec3::new(1.0, 1.0, 0.9)) // Slightly warm white
                .with_intensity(200.0)
                .with_radius(12.0),
        });

        // Add secondary fill light for better illumination
        self.vm.execute(Atom::AddLight {
            id: GeoId::Light(1),
            light: Light::new_pointlight(Vec3::new(0.0, 0.0, half_size - 2.0))
                .with_color(Vec3::new(0.9, 0.9, 1.0)) // Slightly cool white
                .with_intensity(50.0)
                .with_radius(6.0),
        });

        // Set up rendering
        self.vm
            .execute(Atom::SetBackground(Vec4::new(0.1, 0.1, 0.1, 1.0))); // Dark gray background
        self.vm.execute(Atom::SetRenderMode(RenderMode::Compute3D));

        // Set up camera to look inside the Cornell box
        use scenevm::Camera3D;
        let camera = Camera3D::default()
            .look_at(
                Vec3::new(0.0, 0.0, -12.0), // Camera position inside the box, looking toward back wall
                Vec3::new(0.0, 0.0, 0.0),   // Look at center of box
                Vec3::new(0.0, 1.0, 0.0),   // Up vector
            )
            .with_perspective(60.0, 0.1, 100.0);

        self.vm.execute(Atom::SetCamera3D { camera });
    }

    fn draw(&mut self, pixels: &mut [u8], ctx: &mut TheContext) {
        // No rotation - keep the Cornell box stationary
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
        true
    }
}
