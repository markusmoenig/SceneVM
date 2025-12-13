use uuid::Uuid;
use vek::Vec4;

use crate::ui::{
    drawable::Drawable,
    workspace::{UiView, ViewContext},
};

#[derive(Debug, Clone)]
pub struct Label {
    pub id: Uuid,
    pub text: String,
    pub origin: [f32; 2],
    pub px_size: f32,
    pub color: Vec4<f32>,
    pub layer: i32,
}

impl Label {
    pub fn new<T: Into<String>>(text: T, origin: [f32; 2], px_size: f32, color: Vec4<f32>) -> Self {
        Self {
            id: Uuid::new_v4(),
            text: text.into(),
            origin,
            px_size,
            color,
            layer: 0,
        }
    }

    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }
}

impl UiView for Label {
    fn build(&mut self, ctx: &mut ViewContext) {
        ctx.push(Drawable::Text {
            id: self.id,
            text: self.text.clone(),
            origin: self.origin,
            px_size: self.px_size,
            color: self.color,
            layer: self.layer,
        });
    }
}
