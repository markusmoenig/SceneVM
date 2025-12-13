use uuid::Uuid;
use vek::Vec4;

/// Simple drawable primitives emitted by UI views.
#[derive(Debug, Clone)]
pub enum Drawable {
    /// A textured quad in logical space.
    Quad {
        id: Uuid,
        tile_id: Uuid,
        rect: [f32; 4],    // x, y, w, h
        uv: [[f32; 2]; 4], // uv for each corner
        layer: i32,
        tint: Vec4<f32>,
    },
    /// Solid color rect (no atlas dependency).
    Rect {
        id: Uuid,
        rect: [f32; 4],
        fill: Vec4<f32>,
        border: Vec4<f32>,
        radius_norm: f32,
        border_norm: f32,
        layer: i32,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct UiColor(pub Vec4<f32>);

#[derive(Debug, Clone)]
pub struct UiImage {
    pub tile_id: Uuid,
    pub uv: [[f32; 2]; 4],
}
