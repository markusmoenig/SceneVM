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
