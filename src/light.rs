use vek::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightType {
    Point,
}

#[derive(Debug, Clone)]
pub struct Light {
    pub light_type: LightType,
    pub position: Vec3<f32>,
    pub color: Vec3<f32>,
    pub intensity: f32,
    pub radius: f32,
    pub emitting: bool,
    pub start_distance: f32,
    pub end_distance: f32,
    pub flicker: f32,
}
