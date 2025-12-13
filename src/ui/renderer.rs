use super::drawable::Drawable;
use super::style::{StyleParams, StyleRegistry};
use crate::{
    poly2d::Poly2D,
    vm::{Atom, GeoId, VM},
};

/// Renders UI drawables into the 2D layer by emitting quads.
pub struct UiRenderer {
    styles: StyleRegistry,
}

impl UiRenderer {
    pub fn new() -> Self {
        Self {
            styles: StyleRegistry::new(),
        }
    }

    /// Emit drawables into the current chunk as 2D polys.
    pub fn render(&mut self, vm: &mut VM, drawables: &[Drawable]) {
        for d in drawables {
            match d {
                Drawable::Quad {
                    tile_id,
                    rect,
                    uv,
                    layer,
                    ..
                } => {
                    let verts = quad_verts(*rect);
                    let poly = Poly2D::poly(
                        GeoId::Unknown(0),
                        *tile_id,
                        verts,
                        uv.to_vec(),
                        vec![(0, 1, 2), (0, 2, 3)],
                    )
                    .with_layer(*layer);
                    vm.execute(Atom::AddPoly { poly });
                }
                Drawable::Rect {
                    rect,
                    fill,
                    border,
                    radius_norm,
                    border_norm,
                    layer,
                    ..
                } => {
                    let verts = quad_verts(*rect);
                    let style = StyleParams {
                        fill: *fill,
                        border: *border,
                        radius_norm: *radius_norm,
                        border_norm: *border_norm,
                    };
                    let style_id = self.styles.ensure_style(vm, style);
                    let tile_id = self.styles.tile_id(style_id).expect("missing style tile");
                    let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
                    let poly = Poly2D::poly(
                        GeoId::Unknown(0),
                        tile_id,
                        verts,
                        uv,
                        vec![(0, 1, 2), (0, 2, 3)],
                    )
                    .with_layer(*layer);
                    vm.execute(Atom::AddPoly { poly });
                }
            }
        }
        self.styles.build_if_dirty(vm);
    }
}

fn quad_verts(rect: [f32; 4]) -> Vec<[f32; 2]> {
    let [x, y, w, h] = rect;
    vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
}
