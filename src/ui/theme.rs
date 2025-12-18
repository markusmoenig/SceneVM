//! Theme system for consistent UI styling

use vek::Vec4;

use crate::ui::{ButtonGroupStyle, ButtonStyle, ParamListStyle, SliderStyle, ToolbarStyle};

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
            background: Vec4::new(0.02, 0.02, 0.02, 1.0), // Almost black
            surface: Vec4::new(0.08, 0.08, 0.08, 1.0),    // Very dark gray
            surface_variant: Vec4::new(0.12, 0.12, 0.13, 1.0), // Slightly lighter

            // Primary interactive elements - darker, more subtle
            primary: Vec4::new(0.18, 0.18, 0.19, 1.0),
            primary_hover: Vec4::new(0.22, 0.22, 0.24, 1.0),
            primary_active: Vec4::new(0.08, 0.08, 0.09, 1.0), // Much darker for pressed state

            // Borders - very subtle, almost invisible
            border: Vec4::new(0.15, 0.15, 0.16, 1.0),
            border_subtle: Vec4::new(0.1, 0.1, 0.11, 1.0),

            // Text - crisp white
            text: Vec4::new(1.0, 1.0, 1.0, 1.0),
            text_secondary: Vec4::new(0.7, 0.7, 0.72, 1.0),

            // Accent - vibrant blue
            accent: Vec4::new(0.0, 0.48, 1.0, 1.0), // Bright blue
            accent_hover: Vec4::new(0.2, 0.58, 1.0, 1.0),

            // Rounded corners
            radius_px: 10.0, // Rounded corners
            border_px: 0.0,  // No visible borders
        }
    }

    /// Light theme (placeholder for future)
    pub fn light() -> Self {
        Self {
            name: "Light".into(),

            background: Vec4::new(0.95, 0.95, 0.97, 1.0),
            surface: Vec4::new(1.0, 1.0, 1.0, 1.0),
            surface_variant: Vec4::new(0.93, 0.93, 0.95, 1.0),

            primary: Vec4::new(0.88, 0.88, 0.9, 1.0),
            primary_hover: Vec4::new(0.85, 0.85, 0.88, 1.0),
            primary_active: Vec4::new(0.8, 0.8, 0.85, 1.0),

            border: Vec4::new(0.8, 0.8, 0.82, 1.0),
            border_subtle: Vec4::new(0.9, 0.9, 0.92, 1.0),

            text: Vec4::new(0.1, 0.1, 0.12, 1.0),
            text_secondary: Vec4::new(0.4, 0.4, 0.45, 1.0),

            accent: Vec4::new(0.2, 0.4, 0.7, 1.0),
            accent_hover: Vec4::new(0.25, 0.45, 0.75, 1.0),

            radius_px: 6.0,
            border_px: 1.0,
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
            pressed_border: self.accent,
            radius_px: self.radius_px,
            border_px: self.border_px,
            layer: 15, // Higher layer than toolbar to ensure buttons draw on top
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
        }
    }

    /// Create a slider style with the given rect
    pub fn slider(&self, rect: [f32; 4]) -> SliderStyle {
        SliderStyle {
            rect,
            track_color: self.primary, // Use primary instead of surface for better contrast
            fill_color: self.accent,
            thumb_color: self.accent_hover,
            thumb_radius: 6.0,
            track_height: 4.0,
            layer: 11,
        }
    }

    /// Create a param list style with the given rect
    pub fn param_list(&self, rect: [f32; 4]) -> ParamListStyle {
        ParamListStyle {
            rect,
            fill: self.surface,
            border: self.border_subtle,
            radius_px: self.radius_px,
            border_px: self.border_px,
            layer: 10,
            title_color: self.accent,
            title_size: 16.0,
        }
    }
}
