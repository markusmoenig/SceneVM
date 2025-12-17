use super::drawable::Drawable;
use super::style::{StyleParams, StyleRegistry};
use super::text::TextCache;
use crate::{
    Embedded,
    poly2d::Poly2D,
    vm::{Atom, GeoId, VM},
};

/// Renders UI drawables into the 2D layer by emitting quads.
pub struct UiRenderer {
    styles: StyleRegistry,
    text: TextCache,
    next_id: u32,
}

impl UiRenderer {
    pub fn new() -> Self {
        let font_bytes = Embedded::get("ui_font.ttf").map(|d| d.data.to_vec());
        Self {
            styles: StyleRegistry::new(),
            text: TextCache::new(font_bytes),
            next_id: 0,
        }
    }

    pub fn text_cache(&self) -> &TextCache {
        &self.text
    }

    fn alloc_id(&mut self) -> GeoId {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        GeoId::Unknown(id)
    }

    /// Emit drawables into the current chunk as 2D polys.
    pub fn render(&mut self, vm: &mut VM, drawables: &[Drawable]) {
        self.render_internal(vm, drawables, true);
    }

    /// Emit drawables without clearing geometry (for rendering to separate layers).
    /// Used when rendering popups to a different VM layer.
    pub fn render_no_clear(&mut self, vm: &mut VM, drawables: &[Drawable]) {
        self.render_internal(vm, drawables, false);
    }

    fn render_internal(&mut self, vm: &mut VM, drawables: &[Drawable], clear: bool) {
        if clear {
            // Wipe previous UI geometry so we don't accumulate quads across frames.
            vm.execute(Atom::ClearGeometry);
        }

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
                        self.alloc_id(),
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
                    radius_px,
                    border_px,
                    layer,
                    ..
                } => {
                    let verts = quad_verts(*rect);
                    let style = StyleParams {
                        fill: *fill,
                        border: *border,
                        radius_px: *radius_px,
                        border_px: *border_px,
                    };
                    let style_id = self.styles.ensure_style(vm, style);
                    let tile_id = self.styles.tile_id(style_id).expect("missing style tile");
                    let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
                    let poly = Poly2D::poly(
                        self.alloc_id(),
                        tile_id,
                        verts,
                        uv,
                        vec![(0, 1, 2), (0, 2, 3)],
                    )
                    .with_layer(*layer);
                    vm.execute(Atom::AddPoly { poly });
                }
                Drawable::Text {
                    id: _,
                    text,
                    origin,
                    px_size,
                    color,
                    layer,
                } => {
                    let glyphs = self.text.layout_positions(text, *px_size);
                    let start_x = origin[0];
                    let start_y = origin[1];
                    for g in glyphs {
                        let Some(entry) = self.text.ensure_glyph(vm, g.parent, *px_size, *color)
                        else {
                            continue;
                        };
                        // Layout gives glyph bounds at (x, y) with width/height.
                        let x0 = start_x + g.x;
                        let y0 = start_y + g.y;
                        let w = g.width as f32;
                        let h = g.height as f32;
                        if w <= 0.0 || h <= 0.0 {
                            continue;
                        }
                        let verts = vec![[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]];
                        let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
                        let poly = Poly2D::poly(
                            self.alloc_id(),
                            entry.tile_id,
                            verts,
                            uv,
                            vec![(0, 1, 2), (0, 2, 3)],
                        )
                        .with_layer(*layer);
                        vm.execute(Atom::AddPoly { poly });
                    }
                }
            }
        }
        self.styles.build_if_dirty(vm);
        self.text.build_if_dirty(vm);
    }
}

fn quad_verts(rect: [f32; 4]) -> Vec<[f32; 2]> {
    let [x, y, w, h] = rect;
    vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
}
