use uuid::Uuid;
use vek::Vec4;

use crate::ui::{
    drawable::Drawable,
    event::{UiAction, UiEvent, UiEventKind, UiEventOutcome},
    workspace::{UiView, ViewContext},
};

#[derive(Debug, Clone)]
pub struct SliderStyle {
    pub rect: [f32; 4], // x, y, w, h
    pub track_color: Vec4<f32>,
    pub fill_color: Vec4<f32>,
    pub thumb_color: Vec4<f32>,
    pub thumb_radius: f32,
    pub track_height: f32,
    pub layer: i32,
}

impl Default for SliderStyle {
    fn default() -> Self {
        Self {
            rect: [10.0, 10.0, 200.0, 32.0],
            track_color: Vec4::new(0.2, 0.2, 0.22, 1.0),
            fill_color: Vec4::new(0.4, 0.5, 0.7, 1.0),
            thumb_color: Vec4::new(0.9, 0.9, 0.95, 1.0),
            thumb_radius: 12.0,
            track_height: 6.0,
            layer: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Slider {
    pub id: Uuid,
    pub style: SliderStyle,
    pub value: f32, // 0.0 to 1.0
    pub min: f32,
    pub max: f32,
    dragging: bool,
    active_pointer: Option<u32>,
}

impl Slider {
    pub fn new(style: SliderStyle, min: f32, max: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            style,
            value: 0.5,
            min,
            max,
            dragging: false,
            active_pointer: None,
        }
    }

    pub fn with_value(mut self, value: f32) -> Self {
        self.value = (value - self.min) / (self.max - self.min).max(0.001);
        self.value = self.value.clamp(0.0, 1.0);
        self
    }

    pub fn get_value(&self) -> f32 {
        self.min + self.value * (self.max - self.min)
    }

    fn thumb_position(&self) -> [f32; 2] {
        let [x, y, w, h] = self.style.rect;
        let center_y = y + h * 0.5;
        let thumb_x = x + self.value * w;
        [thumb_x, center_y]
    }

    fn hit_thumb(&self, pos: [f32; 2]) -> bool {
        let [tx, ty] = self.thumb_position();
        let r = self.style.thumb_radius;
        let dx = pos[0] - tx;
        let dy = pos[1] - ty;
        (dx * dx + dy * dy) <= r * r
    }

    fn hit_track(&self, pos: [f32; 2]) -> bool {
        let [x, y, w, h] = self.style.rect;
        pos[0] >= x && pos[0] <= x + w && pos[1] >= y && pos[1] <= y + h
    }

    fn update_value_from_pos(&mut self, pos: [f32; 2]) -> bool {
        let [x, _y, w, _h] = self.style.rect;
        let new_value = ((pos[0] - x) / w).clamp(0.0, 1.0);
        if (new_value - self.value).abs() > 0.001 {
            self.value = new_value;
            true
        } else {
            false
        }
    }
}

impl UiView for Slider {
    fn build(&mut self, ctx: &mut ViewContext) {
        let [x, y, w, h] = self.style.rect;
        let center_y = y + h * 0.5;
        let track_h = self.style.track_height;
        let half_track = track_h * 0.5;

        // Draw background track (pill shape - radius = half height in pixels)
        let track_radius_px = track_h * 0.5;
        ctx.push(Drawable::Rect {
            id: Uuid::new_v4(),
            rect: [x, center_y - half_track, w, track_h],
            fill: self.style.track_color,
            border: Vec4::new(0.0, 0.0, 0.0, 0.0),
            radius_px: track_radius_px,
            border_px: 0.0,
            layer: self.style.layer,
        });

        // Draw filled track (up to thumb position) with same rounding
        let fill_w = self.value * w;
        if fill_w > 0.0 {
            ctx.push(Drawable::Rect {
                id: Uuid::new_v4(),
                rect: [x, center_y - half_track, fill_w, track_h],
                fill: self.style.fill_color,
                border: Vec4::new(0.0, 0.0, 0.0, 0.0),
                radius_px: track_radius_px, // Same pixel radius as background
                border_px: 0.0,
                layer: self.style.layer + 1,
            });
        }

        // Draw thumb (circle - radius in pixels)
        let [tx, ty] = self.thumb_position();
        let r = self.style.thumb_radius;
        let thumb_size = r * 2.0;
        ctx.push(Drawable::Rect {
            id: self.id,
            rect: [tx - r, ty - r, thumb_size, thumb_size],
            fill: self.style.thumb_color,
            border: Vec4::new(0.0, 0.0, 0.0, 0.0),
            radius_px: r, // Pixel radius
            border_px: 0.0,
            layer: self.style.layer + 2,
        });
    }

    fn handle_event(&mut self, evt: &UiEvent) -> UiEventOutcome {
        match evt.kind {
            UiEventKind::PointerDown => {
                if self.hit_thumb(evt.pos) || self.hit_track(evt.pos) {
                    self.dragging = true;
                    self.active_pointer = Some(evt.pointer_id);
                    let changed = self.update_value_from_pos(evt.pos);
                    let mut outcome = UiEventOutcome::dirty();
                    if changed {
                        outcome.merge(UiEventOutcome::with_action(UiAction::SliderChanged(
                            self.id,
                            self.get_value(),
                        )));
                    }
                    return outcome;
                }
            }
            UiEventKind::PointerMove => {
                if self.dragging && self.active_pointer == Some(evt.pointer_id) {
                    let changed = self.update_value_from_pos(evt.pos);
                    if changed {
                        let mut outcome = UiEventOutcome::dirty();
                        outcome.merge(UiEventOutcome::with_action(UiAction::SliderChanged(
                            self.id,
                            self.get_value(),
                        )));
                        return outcome;
                    }
                }
            }
            UiEventKind::PointerUp => {
                if self.active_pointer == Some(evt.pointer_id) {
                    self.dragging = false;
                    self.active_pointer = None;
                }
            }
        }
        UiEventOutcome::none()
    }
}
