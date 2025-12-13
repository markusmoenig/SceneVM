use uuid::Uuid;
use vek::Vec4;

use crate::ui::workspace::NodeId;
use crate::ui::{Drawable, UiView, ViewContext};

/// Orientation for the toolbar layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolbarOrientation {
    Horizontal,
    Vertical,
}

/// Style properties for a toolbar widget.
#[derive(Debug, Clone)]
pub struct ToolbarStyle {
    pub rect: [f32; 4],    // x, y, w, h in pixels
    pub fill: Vec4<f32>,   // Background color
    pub border: Vec4<f32>, // Border color
    pub radius_px: f32,    // Corner radius
    pub border_px: f32,    // Border width
    pub layer: i32,        // Rendering layer
}

/// Separator style for toolbars.
#[derive(Debug, Clone)]
pub struct ToolbarSeparator {
    pub color: Vec4<f32>,
    pub thickness: f32,
    pub length: f32, // Length of the separator (perpendicular to orientation)
}

impl Default for ToolbarSeparator {
    fn default() -> Self {
        Self {
            color: Vec4::new(0.3, 0.3, 0.35, 1.0),
            thickness: 1.0,
            length: 24.0,
        }
    }
}

/// A toolbar widget that draws a background and lays out child items.
#[derive(Debug, Clone)]
pub struct Toolbar {
    pub id: String,
    render_id: Uuid,
    pub style: ToolbarStyle,
    pub orientation: ToolbarOrientation,
    pub spacing: f32, // Space between items
    pub offset: f32,  // Initial offset from the start
    pub children: Vec<NodeId>,
    pub extra_spacing: Vec<(NodeId, f32)>, // Extra spacing after specific children
    pub separators: Vec<(NodeId, ToolbarSeparator)>, // Separators after specific children
    separator_ids: Vec<Uuid>,              // IDs for separator drawables
    pub manual_separators: Vec<(f32, ToolbarSeparator)>, // Manually positioned separators (position, style)
}

impl Toolbar {
    /// Create a new toolbar widget.
    pub fn new(style: ToolbarStyle, orientation: ToolbarOrientation) -> Self {
        Self {
            id: String::new(),
            render_id: Uuid::new_v4(),
            style,
            orientation,
            spacing: 4.0,
            offset: 8.0,
            children: Vec::new(),
            extra_spacing: Vec::new(),
            separators: Vec::new(),
            separator_ids: Vec::new(),
            manual_separators: Vec::new(),
        }
    }

    /// Set the widget ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the spacing between items.
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// Set the initial offset from the start.
    pub fn with_offset(mut self, offset: f32) -> Self {
        self.offset = offset;
        self
    }

    /// Add a child to the toolbar.
    pub fn add_child(&mut self, child: NodeId) {
        self.children.push(child);
    }

    /// Add extra spacing after a specific child.
    pub fn add_extra_spacing(&mut self, child: NodeId, spacing: f32) {
        self.extra_spacing.push((child, spacing));
    }

    /// Get extra spacing for a child, if any.
    pub fn get_extra_spacing(&self, child: NodeId) -> f32 {
        self.extra_spacing
            .iter()
            .find(|(id, _)| *id == child)
            .map(|(_, spacing)| *spacing)
            .unwrap_or(0.0)
    }

    /// Add a separator after a specific child.
    /// The separator uses default styling unless you provide a custom ToolbarSeparator.
    pub fn add_separator(&mut self, child: NodeId) {
        self.separators.push((child, ToolbarSeparator::default()));
        self.separator_ids.push(Uuid::new_v4());
    }

    /// Add a separator with custom styling after a specific child.
    pub fn add_separator_with_style(&mut self, child: NodeId, separator: ToolbarSeparator) {
        self.separators.push((child, separator));
        self.separator_ids.push(Uuid::new_v4());
    }

    /// Calculate the position for a separator after a child at the given index.
    /// Returns (x, y, width, height) for the separator drawable.
    /// This is a helper for manual layout - call this to get separator positions.
    pub fn calculate_separator_position(
        &self,
        index: usize,
        item_size: f32,
        extra_spacing_before: f32,
    ) -> Option<[f32; 4]> {
        // Check if there's a separator at this index
        let separator_opt = self
            .separators
            .iter()
            .enumerate()
            .find_map(
                |(sep_idx, (_, sep))| {
                    if sep_idx == index { Some(sep) } else { None }
                },
            );

        let sep = separator_opt?;
        let [x, y, w, h] = self.style.rect;

        match self.orientation {
            ToolbarOrientation::Horizontal => {
                // Position separator after the item
                let sep_x = x
                    + self.offset
                    + (index as f32 * (item_size + self.spacing))
                    + item_size
                    + self.spacing
                    + extra_spacing_before
                    + (self.spacing / 2.0);
                let sep_y = y + (h - sep.length) / 2.0;
                Some([sep_x, sep_y, sep.thickness, sep.length])
            }
            ToolbarOrientation::Vertical => {
                // Position separator after the item
                let sep_x = x + (w - sep.length) / 2.0;
                let sep_y = y
                    + self.offset
                    + (index as f32 * (item_size + self.spacing))
                    + item_size
                    + self.spacing
                    + extra_spacing_before
                    + (self.spacing / 2.0);
                Some([sep_x, sep_y, sep.length, sep.thickness])
            }
        }
    }

    /// Add a separator at the given position that will be drawn automatically.
    /// For horizontal toolbars: position is x coordinate, separator is vertical
    /// For vertical toolbars: position is y coordinate, separator is horizontal
    pub fn add_separator_at(&mut self, position: f32, separator_style: Option<ToolbarSeparator>) {
        let sep = separator_style.unwrap_or_default();
        self.manual_separators.push((position, sep));
    }

    /// Create a separator drawable at the given position.
    /// For horizontal toolbars: position is x coordinate, separator is vertical
    /// For vertical toolbars: position is y coordinate, separator is horizontal
    pub fn create_separator_at(
        &self,
        position: f32,
        separator_style: Option<ToolbarSeparator>,
    ) -> Drawable {
        let sep = separator_style.unwrap_or_default();
        let [x, y, w, h] = self.style.rect;

        let rect = match self.orientation {
            ToolbarOrientation::Horizontal => {
                // Vertical separator at x position
                let sep_y = y + (h - sep.length) / 2.0;
                [position, sep_y, sep.thickness, sep.length]
            }
            ToolbarOrientation::Vertical => {
                // Horizontal separator at y position
                let sep_x = x + (w - sep.length) / 2.0;
                [sep_x, position, sep.length, sep.thickness]
            }
        };

        Drawable::Rect {
            id: Uuid::new_v4(),
            rect,
            fill: sep.color,
            border: Vec4::new(0.0, 0.0, 0.0, 0.0),
            radius_px: 0.0,
            border_px: 0.0,
            layer: self.style.layer + 1,
        }
    }
}

impl UiView for Toolbar {
    fn build(&mut self, ctx: &mut ViewContext) {
        // Draw the background
        ctx.push(Drawable::Rect {
            id: self.render_id,
            rect: self.style.rect,
            fill: self.style.fill,
            border: self.style.border,
            radius_px: self.style.radius_px,
            border_px: self.style.border_px,
            layer: self.style.layer,
        });

        // Draw manual separators
        for (position, sep) in &self.manual_separators {
            let drawable = self.create_separator_at(*position, Some(sep.clone()));
            ctx.push(drawable);
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn view_id(&self) -> &str {
        &self.id
    }
}
