use crate::{
    Atom, GeoId, Light, Poly2D, Poly3D, RenderMode, SceneVM, SceneVMApp, SceneVMRenderCtx,
};
use uuid::Uuid;
use vek::{Mat4, Vec3, Vec4};

fn pack_material(
    roughness: f32,
    metallic: f32,
    opacity: f32,
    emissive: f32,
    normal_x: Option<f32>,
    normal_y: Option<f32>,
) -> [u8; 4] {
    let r = (roughness.clamp(0.0, 1.0) * 15.0).round() as u8;
    let m = (metallic.clamp(0.0, 1.0) * 15.0).round() as u8;
    let o = (opacity.clamp(0.0, 1.0) * 15.0).round() as u8;
    let e = (emissive.clamp(0.0, 1.0) * 15.0).round() as u8;

    let mat_lo = r | (m << 4);
    let mat_hi = o | (e << 4);

    let nx = normal_x.unwrap_or(0.0);
    let ny = normal_y.unwrap_or(0.0);
    let norm_x = ((nx.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8;
    let norm_y = ((ny.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8;

    [mat_lo, mat_hi, norm_x, norm_y]
}

pub struct DemoApp {
    matrix: Mat4<f32>,
}

impl DemoApp {
    pub fn new() -> Self {
        Self {
            matrix: Mat4::identity(),
        }
    }
}

impl SceneVMApp for DemoApp {
    fn init(&mut self, vm: &mut SceneVM, _size: (u32, u32)) {
        let tile_id = Uuid::new_v4();
        let overlay_tile = Uuid::new_v4();

        vm.execute(Atom::SetBackground(Vec4::zero()));
        vm.execute(Atom::AddSolidWithMaterial {
            id: tile_id,
            color: [180, 180, 200, 255],
            material: pack_material(0.1, 0.0, 1.0, 0.0, None, None),
        });
        vm.execute(Atom::AddSolid {
            id: overlay_tile,
            color: [255, 96, 96, 180],
        });
        vm.execute(Atom::BuildAtlas);

        vm.execute(Atom::AddPoly3D {
            poly: Poly3D::cube(GeoId::Unknown(0), tile_id, Vec3::zero(), 2.0),
        });
        vm.execute(Atom::AddLight {
            id: GeoId::Light(0),
            light: Light::new_pointlight(Vec3::new(0.0, 1.5, -4.0))
                .with_color(Vec3::new(1.0, 0.95, 0.9))
                .with_intensity(160.0)
                .with_radius(12.0)
                .with_end_distance(18.0),
        });

        vm.execute(Atom::SetGP3(Vec4::new(0.6, 0.6, 0.7, 0.15)));
        vm.execute(Atom::SetGP6(Vec4::new(10.0, 50.0, 2.0, 16.0)));
        vm.execute(Atom::SetRenderMode(RenderMode::Compute3D));

        let overlay_index = vm.add_vm_layer();
        vm.set_active_vm(overlay_index);
        vm.execute(Atom::AddPoly {
            poly: Poly2D::poly(
                GeoId::Unknown(0),
                overlay_tile,
                vec![[40.0, 40.0], [160.0, 40.0], [160.0, 120.0], [40.0, 120.0]],
                vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                vec![(0, 1, 2), (0, 2, 3)],
            ),
        });
        vm.set_active_vm(0);
    }

    fn update(&mut self, vm: &mut SceneVM) {
        let rot = Mat4::<f32>::rotation_y(0.02) * Mat4::<f32>::rotation_x(0.01);
        self.matrix = rot * self.matrix;
        vm.execute(Atom::SetTransform3D(self.matrix));
    }

    fn render(&mut self, vm: &mut SceneVM, ctx: &mut dyn SceneVMRenderCtx) {
        let _ = ctx.present(vm);
    }

    fn mouse_down(&mut self, _vm: &mut SceneVM, _x: f32, _y: f32) {}
}
