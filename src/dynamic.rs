use uuid::Uuid;
use vek::{Vec2, Vec3};

use crate::vm::GeoId;

/// Types of dynamic objects that can be injected per-frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynamicKind {
    BillboardTile = 0,
}

impl Default for DynamicKind {
    fn default() -> Self {
        DynamicKind::BillboardTile
    }
}

/// Per-frame dynamic object description (billboards, particles, etc.).
#[derive(Clone, Debug)]
pub struct DynamicObject {
    pub id: GeoId,
    pub kind: DynamicKind,
    pub tile_id: Option<Uuid>,
    pub center: Vec3<f32>,
    pub view_right: Vec3<f32>,
    pub view_up: Vec3<f32>,
    pub size: f32,
}

impl Default for DynamicObject {
    fn default() -> Self {
        Self {
            id: GeoId::Unknown(0),
            kind: DynamicKind::BillboardTile,
            tile_id: None,
            center: Vec3::zero(),
            view_right: Vec3::unit_x(),
            view_up: Vec3::unit_y(),
            size: 1.0,
        }
    }
}

impl DynamicObject {
    /// Convenience constructor for a billboard that references a tile.
    pub fn billboard_tile(
        id: GeoId,
        tile_id: Uuid,
        center: Vec3<f32>,
        view_right: Vec3<f32>,
        view_up: Vec3<f32>,
        size: f32,
    ) -> Self {
        Self {
            id,
            kind: DynamicKind::BillboardTile,
            tile_id: Some(tile_id),
            center,
            view_right,
            view_up,
            size,
        }
    }

    /// Convenience constructor for a 2D billboard, supplying only the XY position and size.
    pub fn billboard_tile_2d(id: GeoId, tile_id: Uuid, pos: Vec2<f32>, size: f32) -> Self {
        Self::billboard_tile(
            id,
            tile_id,
            Vec3::new(pos.x, pos.y, 0.0),
            Vec3::unit_x(),
            Vec3::unit_y(),
            size,
        )
    }
}
