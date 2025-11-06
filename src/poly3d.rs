use crate::GeoId;
use uuid::Uuid;
use vek::Vec3;

#[derive(Debug, Clone)]
pub struct Poly3D {
    pub id: GeoId,
    pub tile_id: uuid::Uuid,
    pub vertices: Vec<[f32; 4]>, // world-space XYZ(W)
    pub uvs: Vec<[f32; 2]>,      // per-vertex UV
    pub indices: Vec<(usize, usize, usize)>,
    pub layer: i32, // for future (not used by ray depth)
    pub visible: bool,
}

impl Poly3D {
    /// Construct a Poly3D manually from geometry arrays.
    #[inline]
    pub fn poly(
        id: GeoId,
        tile_id: Uuid,
        vertices: Vec<[f32; 4]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<(usize, usize, usize)>,
    ) -> Self {
        Self {
            id,
            tile_id,
            vertices,
            uvs,
            indices,
            layer: 0,
            visible: true,
        }
    }

    /// Construct a cube centered at `center` with edge length `size`.
    #[inline]
    pub fn cube(id: GeoId, tile_id: Uuid, center: Vec3<f32>, size: f32) -> Self {
        let h = 0.5 * size;
        let (cx, cy, cz) = (center[0], center[1], center[2]);
        let p = |x: f32, y: f32, z: f32| -> [f32; 4] { [cx + x * h, cy + y * h, cz + z * h, 1.0] };

        // 24 verts (4 per face) in the order: -Z(front), +Z(back), -X(left), +X(right), +Y(top), -Y(bottom)
        // Each face wound CCW looking at the face.
        let vertices: Vec<[f32; 4]> = vec![
            // front (-Z)
            p(-1.0, -1.0, -1.0),
            p(1.0, -1.0, -1.0),
            p(1.0, 1.0, -1.0),
            p(-1.0, 1.0, -1.0),
            // back (+Z)
            p(-1.0, -1.0, 1.0),
            p(-1.0, 1.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(1.0, -1.0, 1.0),
            // left (-X)
            p(-1.0, -1.0, 1.0),
            p(-1.0, -1.0, -1.0),
            p(-1.0, 1.0, -1.0),
            p(-1.0, 1.0, 1.0),
            // right (+X)
            p(1.0, -1.0, -1.0),
            p(1.0, -1.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(1.0, 1.0, -1.0),
            // top (+Y)
            p(-1.0, 1.0, -1.0),
            p(1.0, 1.0, -1.0),
            p(1.0, 1.0, 1.0),
            p(-1.0, 1.0, 1.0),
            // bottom (-Y)
            p(-1.0, -1.0, 1.0),
            p(1.0, -1.0, 1.0),
            p(1.0, -1.0, -1.0),
            p(-1.0, -1.0, -1.0),
        ];

        // Per-face UVs (full 0..1 quad per face)
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(24);
        for _ in 0..6 {
            uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        }

        // Two triangles per face
        let mut indices: Vec<(usize, usize, usize)> = Vec::with_capacity(12);
        for f in 0..6 {
            let b = f * 4;
            indices.push((b + 0, b + 1, b + 2));
            indices.push((b + 0, b + 2, b + 3));
        }

        Self {
            id,
            tile_id,
            vertices,
            uvs,
            indices,
            layer: 0,
            visible: true,
        }
    }

    #[inline]
    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }

    #[inline]
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    #[inline]
    pub fn with_vertices(mut self, vertices: Vec<[f32; 4]>) -> Self {
        self.vertices = vertices;
        self
    }

    #[inline]
    pub fn with_uvs(mut self, uvs: Vec<[f32; 2]>) -> Self {
        self.uvs = uvs;
        self
    }

    #[inline]
    pub fn with_indices(mut self, indices: Vec<(usize, usize, usize)>) -> Self {
        self.indices = indices;
        self
    }

    #[inline]
    pub fn with_tile_id(mut self, tile_id: Uuid) -> Self {
        self.tile_id = tile_id;
        self
    }
}
