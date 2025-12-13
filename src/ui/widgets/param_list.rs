use uuid::Uuid;
use vek::Vec4;

use crate::ui::workspace::NodeId;
use crate::ui::{Drawable, UiView, ViewContext};

/// Style properties for a parameter list widget.
#[derive(Debug, Clone)]
pub struct ParamListStyle {
    pub rect: [f32; 4],    // x, y, w, h in pixels
    pub fill: Vec4<f32>,   // Background color
    pub border: Vec4<f32>, // Border color
    pub radius_px: f32,    // Corner radius
    pub border_px: f32,    // Border width
    pub layer: i32,        // Rendering layer
}

/// A parameter list widget that displays labels on the left and widgets on the right.
/// Arranges items vertically with automatic layout.
#[derive(Debug, Clone)]
pub struct ParamList {
    pub id: String,
    render_id: Uuid,
    pub style: ParamListStyle,
    pub item_height: f32,             // Height of each row
    pub spacing: f32,                 // Vertical spacing between rows
    pub label_width: f32,             // Width of the label column
    pub padding: f32,                 // Padding inside the list
    pub label_offset: f32,            // Horizontal offset for labels from left edge
    pub items: Vec<(String, NodeId)>, // (label_text, widget_node_id)
    pub label_color: Vec4<f32>,       // Color for labels
    pub label_size: f32,              // Font size for labels
}

impl ParamList {
    /// Create a new parameter list widget.
    pub fn new(style: ParamListStyle) -> Self {
        Self {
            id: String::new(),
            render_id: Uuid::new_v4(),
            style,
            item_height: 32.0,
            spacing: 4.0,
            label_width: 100.0,
            padding: 8.0,
            label_offset: 8.0,
            items: Vec::new(),
            label_color: Vec4::new(0.9, 0.9, 0.95, 1.0),
            label_size: 14.0,
        }
    }

    /// Set the widget ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the height of each row.
    pub fn with_item_height(mut self, height: f32) -> Self {
        self.item_height = height;
        self
    }

    /// Set the spacing between rows.
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// Set the width of the label column.
    pub fn with_label_width(mut self, width: f32) -> Self {
        self.label_width = width;
        self
    }

    /// Set the padding inside the list.
    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// Set the horizontal offset for labels.
    pub fn with_label_offset(mut self, offset: f32) -> Self {
        self.label_offset = offset;
        self
    }

    /// Set the label color.
    pub fn with_label_color(mut self, color: Vec4<f32>) -> Self {
        self.label_color = color;
        self
    }

    /// Set the label font size.
    pub fn with_label_size(mut self, size: f32) -> Self {
        self.label_size = size;
        self
    }

    /// Add a parameter item (label and widget).
    pub fn add_item(&mut self, label: impl Into<String>, widget: NodeId) {
        self.items.push((label.into(), widget));
    }

    /// Get the position for a label at the given index.
    /// Returns [x, y] for the label origin.
    pub fn get_label_position(&self, index: usize) -> [f32; 2] {
        let [x, y, _, _] = self.style.rect;
        let label_x = x + self.padding + self.label_offset;
        // Calculate the center of the row
        let row_center_y = y
            + self.padding
            + (index as f32 * (self.item_height + self.spacing))
            + (self.item_height / 2.0);
        // Position text so its vertical center aligns with row center
        // Text origin is at top-left, so we subtract half the font size
        let label_y = row_center_y - (self.label_size / 2.0);
        [label_x, label_y]
    }

    /// Get the rect for a widget at the given index.
    /// Returns [x, y, w, h] for the widget.
    pub fn get_widget_rect(&self, index: usize, widget_width: f32) -> [f32; 4] {
        let [x, y, w, _] = self.style.rect;
        let widget_x = x + self.padding + self.label_width;
        let widget_y = y + self.padding + (index as f32 * (self.item_height + self.spacing));
        let widget_h = self.item_height;
        // Reserve space for value text that appears to the right of widgets (like sliders)
        // Subtract ~40px to account for 8px gap + ~30px text + margin
        let available_width = w - self.padding * 2.0 - self.label_width - 40.0;
        let final_width = widget_width.min(available_width);
        [widget_x, widget_y, final_width, widget_h]
    }

    /// Calculate the total height needed for all items.
    pub fn calculate_total_height(&self) -> f32 {
        if self.items.is_empty() {
            self.padding * 2.0
        } else {
            self.padding * 2.0
                + (self.items.len() as f32 * self.item_height)
                + ((self.items.len() - 1) as f32 * self.spacing)
        }
    }

    /// Set the position of the ParamList (useful for popups).
    pub fn set_position(&mut self, x: f32, y: f32) {
        self.style.rect[0] = x;
        self.style.rect[1] = y;
    }

    /// Get the size of the ParamList [width, height].
    pub fn get_size(&self) -> [f32; 2] {
        [self.style.rect[2], self.style.rect[3]]
    }
}

impl UiView for ParamList {
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

        // Draw labels
        for (index, (label, _)) in self.items.iter().enumerate() {
            let [label_x, label_y] = self.get_label_position(index);
            ctx.push(Drawable::Text {
                id: Uuid::new_v4(),
                text: label.clone(),
                origin: [label_x, label_y],
                px_size: self.label_size,
                color: self.label_color,
                layer: self.style.layer + 1,
            });
        }

        // Note: Widgets are positioned manually in the demo/user code
        // They need to use get_widget_rect() to calculate their positions
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
