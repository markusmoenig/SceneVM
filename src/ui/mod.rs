//! UI module (feature `ui`): node-driven workspace and drawable emission.
//! This is a lightweight scaffold for a Procreate-like UI layer. It currently
//! defines node/view plumbing and drawable collection; rendering is expected
//! to use the existing 2D path.

mod drawable;
mod event;
pub mod layouts;
mod renderer;
mod style;
mod text;
mod widgets;
mod workspace;

pub use drawable::{Drawable, UiColor, UiImage};
pub use event::{UiAction, UiEvent, UiEventKind, UiEventOutcome};
pub use layouts::{Alignment, HStack, VStack};
pub use renderer::UiRenderer;
pub use style::{StyleId, StyleParams, StyleRegistry};
pub use text::TextCache;
pub use widgets::{
    Button, ButtonKind, ButtonStyle, HAlign, Label, LabelRect, Slider, SliderStyle, VAlign,
};
pub use workspace::{NodeId, UiView, ViewContext, Workspace};

/// Helper function to create empty material data for non-style tiles.
/// Use this when adding image tiles that don't need style rendering.
pub fn create_tile_material(width: u32, height: u32) -> Vec<u8> {
    vec![0u8; (width * height * 4) as usize]
}
