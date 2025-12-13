use uuid::Uuid;
use vek::Vec4;

use crate::ui::{
    drawable::Drawable,
    event::{UiAction, UiEvent, UiEventKind, UiEventOutcome},
    workspace::{UiView, ViewContext},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonKind {
    Momentary,
    Toggle,
}

#[derive(Debug, Clone)]
pub struct ButtonStyle {
    pub rect: [f32; 4], // x, y, w, h in logical space
    pub fill: Vec4<f32>,
    pub border: Vec4<f32>,
    pub pressed_fill: Vec4<f32>,
    pub pressed_border: Vec4<f32>,
    pub radius_px: f32, // Corner radius in pixels
    pub border_px: f32, // Border width in pixels
    pub layer: i32,
}

impl Default for ButtonStyle {
    fn default() -> Self {
        Self {
            rect: [10.0, 10.0, 120.0, 44.0],
            fill: Vec4::new(0.15, 0.15, 0.18, 1.0),
            border: Vec4::new(0.0, 0.0, 0.0, 0.0),
            pressed_fill: Vec4::new(0.12, 0.12, 0.15, 1.0),
            pressed_border: Vec4::new(0.0, 0.0, 0.0, 0.0),
            radius_px: 4.0,
            border_px: 0.0,
            layer: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ButtonState {
    Idle,
    Hover,
    Pressed,
    On,
}

#[derive(Debug, Clone)]
pub struct Button {
    pub id: Uuid,
    pub style: ButtonStyle,
    pub kind: ButtonKind,
    toggled: bool,
    state: ButtonState,
    active_pointer: Option<u32>,
}

impl Button {
    pub fn new(style: ButtonStyle) -> Self {
        Self {
            id: Uuid::new_v4(),
            style,
            kind: ButtonKind::Momentary,
            toggled: false,
            state: ButtonState::Idle,
            active_pointer: None,
        }
    }

    pub fn with_kind(mut self, kind: ButtonKind) -> Self {
        self.kind = kind;
        self
    }

    fn hit(&self, pos: [f32; 2]) -> bool {
        let [x, y, w, h] = self.style.rect;
        pos[0] >= x && pos[0] <= x + w && pos[1] >= y && pos[1] <= y + h
    }

    fn set_state(&mut self, next: ButtonState) -> UiEventOutcome {
        if self.state != next {
            self.state = next;
            UiEventOutcome::dirty()
        } else {
            UiEventOutcome::none()
        }
    }
}

impl UiView for Button {
    fn build(&mut self, ctx: &mut ViewContext) {
        let (fill, border) = match self.state {
            ButtonState::Pressed | ButtonState::On => {
                (self.style.pressed_fill, self.style.pressed_border)
            }
            _ => (self.style.fill, self.style.border),
        };
        let [_x, _y, w, h] = self.style.rect;
        let min_dim = w.min(h);
        ctx.push(Drawable::Rect {
            id: self.id,
            rect: self.style.rect,
            fill,
            border,
            radius_norm: self.style.radius_px / min_dim,
            border_norm: self.style.border_px / min_dim,
            layer: self.style.layer,
        });
    }

    fn handle_event(&mut self, evt: &UiEvent) -> UiEventOutcome {
        let inside = self.hit(evt.pos);
        match evt.kind {
            UiEventKind::PointerDown => {
                if inside {
                    self.active_pointer = Some(evt.pointer_id);
                    return self.set_state(ButtonState::Pressed);
                }
            }
            UiEventKind::PointerUp => {
                let was_active = self.active_pointer == Some(evt.pointer_id);
                self.active_pointer = None;
                if was_active {
                    match self.kind {
                        ButtonKind::Momentary => {
                            let mut outcome = self.set_state(if inside {
                                ButtonState::Hover
                            } else {
                                ButtonState::Idle
                            });
                            if inside {
                                outcome.merge(UiEventOutcome::with_action(
                                    UiAction::ButtonPressed(self.id),
                                ));
                            }
                            return outcome;
                        }
                        ButtonKind::Toggle => {
                            if inside {
                                self.toggled = !self.toggled;
                                let mut outcome = self.set_state(if self.toggled {
                                    ButtonState::On
                                } else {
                                    ButtonState::Hover
                                });
                                outcome.merge(UiEventOutcome::with_action(
                                    UiAction::ButtonToggled(self.id, self.toggled),
                                ));
                                return outcome;
                            } else {
                                // Pointer released outside; keep current toggled state.
                                return self.set_state(if self.toggled {
                                    ButtonState::On
                                } else {
                                    ButtonState::Idle
                                });
                            }
                        }
                    }
                }
            }
            UiEventKind::PointerMove => {
                if let Some(pid) = self.active_pointer {
                    if pid == evt.pointer_id {
                        return self.set_state(if inside {
                            ButtonState::Pressed
                        } else if self.kind == ButtonKind::Toggle && self.toggled {
                            ButtonState::On
                        } else {
                            ButtonState::Idle
                        });
                    }
                } else {
                    if self.kind == ButtonKind::Toggle && self.state == ButtonState::On {
                        // Keep toggle-on state until the next click changes it.
                        return UiEventOutcome::none();
                    }
                    return self.set_state(if inside {
                        ButtonState::Hover
                    } else {
                        ButtonState::Idle
                    });
                }
            }
        }
        UiEventOutcome::none()
    }
}
