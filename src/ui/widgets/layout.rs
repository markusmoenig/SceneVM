use crate::ui::workspace::{NodeId, UiView, ViewContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Start,
    Center,
    End,
}

#[derive(Debug, Clone)]
pub struct VStack {
    pub children: Vec<NodeId>,
    pub rect: [f32; 4], // x, y, w, h
    pub spacing: f32,
    pub padding: f32,
    pub alignment: Alignment,
}

impl VStack {
    pub fn new(rect: [f32; 4]) -> Self {
        Self {
            children: Vec::new(),
            rect,
            spacing: 4.0,
            padding: 8.0,
            alignment: Alignment::Start,
        }
    }

    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn add_child(&mut self, child: NodeId) {
        self.children.push(child);
    }
}

impl UiView for VStack {
    fn build(&mut self, _ctx: &mut ViewContext) {
        // VStack doesn't draw anything itself, it just arranges children
        // Child layout is handled by the workspace when building the tree
    }
}

#[derive(Debug, Clone)]
pub struct HStack {
    pub children: Vec<NodeId>,
    pub rect: [f32; 4], // x, y, w, h
    pub spacing: f32,
    pub padding: f32,
    pub alignment: Alignment,
}

impl HStack {
    pub fn new(rect: [f32; 4]) -> Self {
        Self {
            children: Vec::new(),
            rect,
            spacing: 4.0,
            padding: 8.0,
            alignment: Alignment::Center,
        }
    }

    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn add_child(&mut self, child: NodeId) {
        self.children.push(child);
    }
}

impl UiView for HStack {
    fn build(&mut self, _ctx: &mut ViewContext) {
        // HStack doesn't draw anything itself, it just arranges children
        // Child layout is handled by the workspace when building the tree
    }
}
