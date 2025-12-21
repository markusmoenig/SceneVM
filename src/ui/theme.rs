//! Theme system for consistent UI styling

use vek::Vec4;

use crate::ui::{
    ButtonGroupStyle, ButtonStyle, DropdownListStyle, ParamListStyle, SliderStyle, ToolbarStyle,
};

/// A UI theme that defines colors and styling for all widgets
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,

    // Background colors
    pub background: Vec4<f32>,
    pub surface: Vec4<f32>,
    pub surface_variant: Vec4<f32>,

    // Interactive element colors
    pub primary: Vec4<f32>,
    pub primary_hover: Vec4<f32>,
    pub primary_active: Vec4<f32>,

    // Border colors
    pub border: Vec4<f32>,
    pub border_subtle: Vec4<f32>,

    // Text colors
    pub text: Vec4<f32>,
    pub text_secondary: Vec4<f32>,

    // Accent colors
    pub accent: Vec4<f32>,
    pub accent_hover: Vec4<f32>,

    // Spacing and sizing
    pub radius_px: f32,
    pub border_px: f32,
}

impl Theme {
    /// Dark theme - distinctive look with rich blacks and vibrant accents
    pub fn dark() -> Self {
        Self {
            name: "Dark".into(),

            // Backgrounds - deep blacks
            background: Vec4::new(0.015, 0.015, 0.02, 1.0), // Near black
            surface: Vec4::new(0.06, 0.06, 0.07, 1.0),      // Deep charcoal
            surface_variant: Vec4::new(0.1, 0.1, 0.11, 1.0), // Slight lift for panels

            // Primary interactive elements - darker, more subtle
            primary: Vec4::new(0.16, 0.16, 0.18, 1.0),
            primary_hover: Vec4::new(0.2, 0.2, 0.22, 1.0),
            primary_active: Vec4::new(0.25, 0.25, 0.27, 1.0), // Brighter for pressed state in dark mode

            // Borders - very subtle, almost invisible
            border: Vec4::new(0.16, 0.16, 0.18, 1.0),
            border_subtle: Vec4::new(0.1, 0.1, 0.11, 1.0),

            // Text - crisp white
            text: Vec4::new(1.0, 1.0, 1.0, 1.0),
            text_secondary: Vec4::new(0.72, 0.72, 0.75, 1.0),

            // Accent - deeper blue
            accent: Vec4::new(0.08, 0.42, 0.9, 1.0), // Slightly darker
            accent_hover: Vec4::new(0.18, 0.5, 0.98, 1.0),

            // Rounded corners
            radius_px: 10.0, // Rounded corners
            border_px: 0.0,  // No visible borders
        }
    }

    /// Light theme - clean, bright appearance with high contrast
    pub fn light() -> Self {
        Self {
            name: "Light".into(),

            // Backgrounds - lighter base so UI elements pop
            background: Vec4::new(0.72, 0.72, 0.74, 1.0), // Light canvas
            surface: Vec4::new(0.66, 0.66, 0.68, 1.0),    // Darker cards/panels
            surface_variant: Vec4::new(0.62, 0.62, 0.64, 1.0), // Inputs/secondary surfaces

            // Primary interactive elements
            primary: Vec4::new(0.5, 0.5, 0.52, 1.0), // Darker to separate from surface
            primary_hover: Vec4::new(0.46, 0.46, 0.48, 1.0),
            primary_active: Vec4::new(0.42, 0.42, 0.44, 1.0),

            // Borders - clearly visible
            border: Vec4::new(0.44, 0.44, 0.46, 1.0), // Outline elements against darker surfaces
            border_subtle: Vec4::new(0.58, 0.58, 0.6, 1.0),

            // Text - black-leaning for maximum contrast on light surfaces
            text: Vec4::new(0.0, 0.0, 0.0, 1.0),
            text_secondary: Vec4::new(0.12, 0.12, 0.14, 1.0),

            // Accent - deeper blue for titles/active states
            accent: Vec4::new(0.02, 0.5, 0.94, 1.0), // Darker but saturated to stand out
            accent_hover: Vec4::new(0.0, 0.58, 1.0, 1.0),

            // Rounded corners
            radius_px: 8.0,
            border_px: 2.0,
        }
    }

    // Style factory methods

    /// Create a button style with the given rect
    pub fn button(&self, rect: [f32; 4]) -> ButtonStyle {
        ButtonStyle {
            rect,
            fill: self.surface_variant,
            border: self.border,
            pressed_fill: self.primary_active,
            pressed_border: self.border,
            radius_px: self.radius_px,
            border_px: self.border_px,
            layer: 15, // Higher layer than toolbar to ensure buttons draw on top
            text_color: self.text,
        }
    }

    /// Create a toolbar style with the given rect
    pub fn toolbar(&self, rect: [f32; 4]) -> ToolbarStyle {
        ToolbarStyle {
            rect,
            fill: self.surface,
            border: self.border_subtle,
            radius_px: self.radius_px,
            border_px: self.border_px,
            layer: 10,
        }
    }

    /// Create a button group style with the given rect and button dimensions
    pub fn button_group(
        &self,
        rect: [f32; 4],
        button_width: f32,
        button_height: f32,
    ) -> ButtonGroupStyle {
        // For dark theme: dark background with light text
        // For light theme: light background with dark text
        let text_bg_color = if self.name == "Light" {
            Vec4::new(1.0, 1.0, 1.0, 0.85) // Light semi-transparent background
        } else {
            Vec4::new(0.0, 0.0, 0.0, 0.85) // Dark semi-transparent background
        };

        ButtonGroupStyle {
            rect,
            button_width,
            button_height,
            spacing: 4.0,
            fill: self.surface_variant,
            border: self.border,
            active_fill: self.accent,
            active_border: self.accent_hover,
            radius_px: self.radius_px,
            border_px: self.border_px,
            layer: 15, // Higher layer than toolbar
            text_color: self.text,
            text_bg_color,
        }
    }

    /// Create a slider style with the given rect
    pub fn slider(&self, rect: [f32; 4]) -> SliderStyle {
        let track_color = if self.name == "Light" {
            self.primary_active // Darker track on light panels
        } else {
            Vec4::new(0.05, 0.05, 0.06, 1.0) // Dark mode: clearly darker than panel
        };

        SliderStyle {
            rect,
            track_color,
            fill_color: self.accent,
            thumb_color: self.accent_hover,
            thumb_radius: 6.0,
            track_height: 4.0,
            layer: 11,
        }
    }

    /// Create a param list style with the given rect
    pub fn param_list(&self, rect: [f32; 4]) -> ParamListStyle {
        // Title can be accent color in dark mode for emphasis
        let title_color = if self.name == "Light" {
            self.text
        } else {
            self.accent
        };

        ParamListStyle {
            rect,
            fill: self.surface_variant, // Match other widget surfaces for consistency
            border: self.border_subtle,
            radius_px: self.radius_px,
            border_px: self.border_px,
            layer: 10,
            title_color,
            title_size: 16.0,
            label_color: self.text, // Labels should always use theme text color
        }
    }

    /// Create a dropdown list style with the given rect
    pub fn dropdown_list(&self, rect: [f32; 4]) -> DropdownListStyle {
        DropdownListStyle {
            rect,
            fill: self.surface_variant,
            border: self.border,
            hover_fill: self.primary_hover,
            text_color: self.text,
            text_size: 14.0,
            radius_px: self.radius_px,
            border_px: self.border_px,
            layer: 15,
            item_height: 36.0,
            max_visible_items: 8,
        }
    }

    /// Get the background color for this theme (for VM clear color)
    pub fn background_color(&self) -> [f32; 4] {
        [
            self.background.x,
            self.background.y,
            self.background.z,
            self.background.w,
        ]
    }
}
