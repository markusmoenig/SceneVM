use crate::{BBox2D, GeoId, Poly2D};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use vek::{Mat3, Vec2};

#[derive(Debug, Default, Clone)]
pub struct Chunk {
    pub origin: Vec2<i32>,
    pub size: i32,
    pub bbox: BBox2D,

    /// 2D Geometrt
    pub polys_map: FxHashMap<GeoId, Poly2D>,

    /// The priority of the chunk.
    pub priority: i32,
}

impl Chunk {
    pub fn new(origin: Vec2<i32>, size: i32) -> Self {
        let bbox = BBox2D::from_pos_size(origin.map(|v| v as f32), Vec2::broadcast(size as f32));
        Self {
            origin,
            size,
            bbox,
            ..Default::default()
        }
    }

    /// Add a 2D polygon with explicit vertices/uvs/indices. Indices are local to this chunk.
    pub fn add_poly_2d(
        &mut self,
        id: GeoId,
        tile_id: Uuid,
        vertices: Vec<[f32; 2]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<(usize, usize, usize)>,
        layer: i32,
    ) {
        let poly = Poly2D {
            id,
            tile_id,
            vertices,
            uvs,
            indices,
            transform: Mat3::identity(),
            layer,
        };
        self.polys_map.insert(id, poly);
    }

    /// Add a 2D line strip tessellated into thick quads (no caps/joins) as one poly.
    /// `points` are in world coords; `width` is in world units.
    pub fn add_line_strip_2d(
        &mut self,
        id: GeoId,
        tile_id: Uuid,
        points: Vec<[f32; 2]>,
        width: f32,
        layer: i32,
    ) {
        if points.len() < 2 {
            return;
        }
        let half = 0.5 * width;
        let mut vertices: Vec<[f32; 2]> = Vec::with_capacity(points.len() * 4);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(points.len() * 4);
        let mut indices: Vec<(usize, usize, usize)> = Vec::with_capacity((points.len() - 1) * 2);

        for seg in 0..(points.len() - 1) {
            let p0 = points[seg];
            let p1 = points[seg + 1];
            let dx = p1[0] - p0[0];
            let dy = p1[1] - p0[1];
            let len = (dx * dx + dy * dy).sqrt();
            if len == 0.0 {
                continue;
            }
            let nx = -dy / len; // left-hand normal (perp)
            let ny = dx / len;
            let ox = nx * half;
            let oy = ny * half;

            // Quad corners (consistent winding: 0-1-2, 0-2-3)
            let v0 = [p0[0] - ox, p0[1] - oy]; // bottom-left
            let v1 = [p0[0] + ox, p0[1] + oy]; // top-left
            let v2 = [p1[0] + ox, p1[1] + oy]; // top-right
            let v3 = [p1[0] - ox, p1[1] - oy]; // bottom-right

            let base = vertices.len();
            vertices.extend_from_slice(&[v0, v1, v2, v3]);
            // Simple UVs per quad (stretch along segment)
            uvs.extend_from_slice(&[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
            indices.push((base + 0, base + 1, base + 2));
            indices.push((base + 0, base + 2, base + 3));
        }

        if vertices.is_empty() {
            return;
        }

        let poly = Poly2D {
            id,
            tile_id,
            vertices,
            uvs,
            indices,
            transform: Mat3::identity(),
            layer,
        };
        self.polys_map.insert(id, poly);
    }

    /// Add a square (axis-aligned) centered at `center` with edge length `size`.
    /// Inserts a new Poly2D using `tile_id` and `id`. UVs cover the full tile.
    pub fn add_square_2d(
        &mut self,
        id: GeoId,
        tile_id: Uuid,
        center: [f32; 2],
        size: f32,
        layer: i32,
    ) {
        if size <= 0.0 {
            return;
        }
        let half = 0.5 * size;
        let (cx, cy) = (center[0], center[1]);
        let x0 = cx - half; // left
        let x1 = cx + half; // right
        let y0 = cy - half; // bottom
        let y1 = cy + half; // top

        let vertices = vec![
            [x0, y0], // bottom-left
            [x0, y1], // top-left
            [x1, y1], // top-right
            [x1, y0], // bottom-right
        ];
        let uvs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let indices = vec![(0, 1, 2), (0, 2, 3)];

        let poly = Poly2D {
            id,
            tile_id,
            vertices,
            uvs,
            indices,
            transform: Mat3::identity(),
            layer,
        };
        self.polys_map.insert(id, poly);
    }
}
