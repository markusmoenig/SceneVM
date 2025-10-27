use vek::Vec4;

/// Describes surface appearance parameters shared by both 2D and 3D rendering.
impl Default for Material {
    fn default() -> Self {
        Self {
            tint: Vec4::new(1.0, 1.0, 1.0, 1.0),
            roughness: 0.5,
            metallic: 0.0,
            opacity: 1.0,
            emission: 0.0,
        }
    }
}

/// The `tint` is multiplied with the sampled atlas color to recolor tiles easily.
#[derive(Debug, Clone)]
pub struct Material {
    /// Base color multiplier (RGBA).
    pub tint: Vec4<f32>,

    /// Surface roughness (0 = smooth, 1 = rough).
    pub roughness: f32,

    /// Metallicness (0 = nonmetal, 1 = pure metal).
    pub metallic: f32,

    /// Opacity (0 = fully transparent, 1 = fully opaque).
    pub opacity: f32,

    /// Emission intensity (0 = none, >0 = glowing).
    pub emission: f32,
}
