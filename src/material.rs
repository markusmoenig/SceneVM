use vek::Vec4;

/// Lighting/shading model used by a material.
#[derive(Clone, Copy, Debug)]
pub enum ShadingModel {
    /// Lit surface. This is the default.
    Shaded = 0,
    /// Flat (Unlit)
    Flat = 1,
}

/// Describes surface appearance parameters shared by both 2D and 3D rendering.
impl Default for Material {
    fn default() -> Self {
        Self {
            model: ShadingModel::Shaded,
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
    pub model: ShadingModel,

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

impl Material {
    pub fn encode_model(&self) -> f32 {
        match self.model {
            ShadingModel::Shaded => 0.0,
            ShadingModel::Flat => 1.0,
        }
    }

    /// Create a lit material with default PBR-ish parameters.
    pub fn shaded() -> Self {
        Self::default()
    }

    /// Create an unlit (flat) material with default parameters.
    pub fn flat() -> Self {
        Self {
            model: ShadingModel::Flat,
            ..Self::default()
        }
    }

    /// Set the shading model.
    pub fn with_model(mut self, m: ShadingModel) -> Self {
        self.model = m;
        self
    }

    /// Set the base color/tint (RGBA).
    pub fn with_tint(mut self, tint: Vec4<f32>) -> Self {
        self.tint = tint;
        self
    }

    /// Set surface roughness.
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness;
        self
    }

    /// Set metallicness.
    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic = metallic;
        self
    }

    /// Set opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set emission intensity.
    pub fn with_emission(mut self, emission: f32) -> Self {
        self.emission = emission;
        self
    }
}
