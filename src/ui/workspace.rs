use crate::ui::{
    drawable::Drawable, event::UiAction, event::UiEvent, event::UiEventOutcome, text::TextCache,
};
use rustc_hash::FxHashMap;
use std::any::Any;
use uuid::Uuid;

/// Identifier for nodes in the UI workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Context passed to views during build; collects drawables.
pub struct ViewContext<'a> {
    drawables: &'a mut Vec<Drawable>,
    current_layer: i32,
    text_cache: &'a TextCache,
}

impl<'a> ViewContext<'a> {
    pub fn push(&mut self, drawable: Drawable) {
        self.drawables.push(drawable);
    }

    pub fn with_layer(&mut self, layer: i32) -> ViewContext<'_> {
        ViewContext {
            drawables: self.drawables,
            current_layer: layer,
            text_cache: self.text_cache,
        }
    }

    pub fn layer(&self) -> i32 {
        self.current_layer
    }

    pub fn text_cache(&self) -> &TextCache {
        self.text_cache
    }
}

/// Trait implemented by UI views to emit drawables.
pub trait UiView: Any {
    fn build(&mut self, ctx: &mut ViewContext);
    fn handle_event(&mut self, _evt: &UiEvent) -> UiEventOutcome {
        UiEventOutcome::none()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn as_any(&self) -> &dyn Any;
    fn view_id(&self) -> &str {
        ""
    }
}

struct Node {
    view: Box<dyn UiView>,
    children: Vec<NodeId>,
}

/// Node-driven UI workspace: holds a tree of views and produces drawables.
pub struct Workspace {
    nodes: FxHashMap<NodeId, Node>,
    roots: Vec<NodeId>,
    dirty: bool,
    pending_actions: Vec<UiAction>,
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            nodes: FxHashMap::default(),
            roots: Vec::new(),
            dirty: true,
            pending_actions: Vec::new(),
        }
    }

    /// Insert a view as a new node and return its id.
    pub fn add_view<V: UiView + 'static>(&mut self, view: V) -> NodeId {
        let id = NodeId::new();
        self.nodes.insert(
            id,
            Node {
                view: Box::new(view),
                children: Vec::new(),
            },
        );
        id
    }

    /// Mark a node as a root in the workspace.
    pub fn add_root(&mut self, id: NodeId) {
        if self.nodes.contains_key(&id) && !self.roots.contains(&id) {
            self.roots.push(id);
        }
    }

    /// Attach a child node under a parent.
    pub fn attach(&mut self, parent: NodeId, child: NodeId) {
        let child_exists = self.nodes.contains_key(&child);
        if child_exists {
            if let Some(p) = self.nodes.get_mut(&parent) {
                if !p.children.contains(&child) {
                    p.children.push(child);
                }
            }
        }
    }

    /// Traverse roots and collect drawables.
    pub fn build(&mut self, text_cache: &TextCache) -> Vec<Drawable> {
        let mut drawables = Vec::new();
        let roots = self.roots.clone();
        for root in roots {
            self.build_node(root, &mut drawables, 0, text_cache);
        }

        // After rendering all normal views, render popups on top
        self.build_popups(&mut drawables, text_cache);

        self.dirty = false;
        drawables
    }

    fn build_node(
        &mut self,
        id: NodeId,
        out: &mut Vec<Drawable>,
        layer: i32,
        text_cache: &TextCache,
    ) {
        let Some(node) = self.nodes.get_mut(&id) else {
            return;
        };

        // Check if this is a Canvas and if it's visible
        let is_visible_canvas =
            if let Some(canvas) = node.view.as_any().downcast_ref::<crate::ui::Canvas>() {
                canvas.is_visible()
            } else {
                true // Not a canvas, always visible
            };

        if !is_visible_canvas {
            return; // Skip this canvas and its children
        }

        let mut ctx = ViewContext {
            drawables: out,
            current_layer: layer,
            text_cache,
        };
        node.view.build(&mut ctx);
        for child in node.children.clone() {
            self.build_node(child, out, layer, text_cache);
        }
    }

    /// Dispatch a UI event to all views; collects actions and marks dirty when a view changes.
    pub fn handle_event(&mut self, evt: &UiEvent) {
        let mut outcome = UiEventOutcome::none();
        let roots = self.roots.clone();
        for root in roots {
            outcome.merge(self.dispatch_node(root, evt));
        }

        // Also dispatch events to visible popup contents
        let popup_nodes = self.get_visible_popup_nodes();
        for popup_id in popup_nodes {
            outcome.merge(self.dispatch_node(popup_id, evt));
        }

        if outcome.dirty {
            self.dirty = true;
        }
        if !outcome.actions.is_empty() {
            self.pending_actions.extend(outcome.actions);
        }
    }

    fn dispatch_node(&mut self, id: NodeId, evt: &UiEvent) -> UiEventOutcome {
        let mut merged = UiEventOutcome::none();
        if let Some(node) = self.nodes.get_mut(&id) {
            // Check if this is a Canvas and if it's visible
            let is_visible_canvas =
                if let Some(canvas) = node.view.as_any().downcast_ref::<crate::ui::Canvas>() {
                    canvas.is_visible()
                } else {
                    true // Not a canvas, always visible
                };

            if !is_visible_canvas {
                return merged; // Skip event dispatch for invisible canvas and its children
            }

            merged.merge(node.view.handle_event(evt));
            let children = node.children.clone();
            for child in children {
                merged.merge(self.dispatch_node(child, evt));
            }
        }
        merged
    }

    /// Returns whether any view changed state since the last build.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Drain and return pending UI actions generated by views.
    pub fn take_actions(&mut self) -> Vec<UiAction> {
        std::mem::take(&mut self.pending_actions)
    }

    /// Replace a node's view with a new one and mark workspace as dirty.
    pub fn update_view<V: UiView + 'static>(&mut self, id: NodeId, view: V) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.view = Box::new(view);
            self.dirty = true;
        }
    }

    /// Get mutable access to a view and mark workspace as dirty.
    /// Returns None if the node doesn't exist or the type doesn't match.
    pub fn get_view_mut<V: UiView + 'static>(&mut self, id: NodeId) -> Option<&mut V> {
        if let Some(node) = self.nodes.get_mut(&id) {
            self.dirty = true;
            node.view.as_any_mut().downcast_mut::<V>()
        } else {
            None
        }
    }

    /// Find a view by its string ID and return mutable access.
    /// Returns None if no view with that ID exists or the type doesn't match.
    pub fn find_view_mut<V: UiView + 'static>(&mut self, id: &str) -> Option<&mut V> {
        for node in self.nodes.values_mut() {
            if node.view.view_id() == id {
                self.dirty = true;
                return node.view.as_any_mut().downcast_mut::<V>();
            }
        }
        None
    }

    /// Check if a point is inside any button with an open popup
    /// Returns true if the point is inside a button with popup or inside the popup itself
    pub fn is_inside_popup_area(&self, pos: [f32; 2], popup_rect: [f32; 4]) -> bool {
        let [px, py, pw, ph] = popup_rect;
        pos[0] >= px && pos[0] <= px + pw && pos[1] >= py && pos[1] <= py + ph
    }

    /// Get list of visible popup node IDs
    fn get_visible_popup_nodes(&self) -> Vec<NodeId> {
        use crate::ui::Button;

        let mut popup_nodes = Vec::new();
        for node in self.nodes.values() {
            if let Some(button) = node.view.as_any().downcast_ref::<Button>() {
                if button.is_popup_visible() {
                    if let Some(popup_id) = button.popup_content {
                        popup_nodes.push(popup_id);
                    }
                }
            }
        }
        popup_nodes
    }

    /// Build popups for all buttons that have visible popups
    fn build_popups(&mut self, out: &mut Vec<Drawable>, text_cache: &TextCache) {
        use crate::ui::Button;

        // Collect popup info first to avoid borrow checker issues
        let mut popups_to_render = Vec::new();

        for (_node_id, node) in &self.nodes {
            if let Some(button) = node.view.as_any().downcast_ref::<Button>() {
                if button.is_popup_visible() {
                    if let Some(popup_content_id) = button.popup_content {
                        // We need the popup size to calculate position
                        // For now, use a placeholder - in real implementation,
                        // the popup widget should provide its size
                        // For ParamList, we can estimate from its rect
                        if self.nodes.contains_key(&popup_content_id) {
                            // Try to get size from the popup view
                            // This is a simplified approach - ideally views would report their size
                            popups_to_render.push((
                                popup_content_id,
                                button.style.rect,
                                button.popup_alignment,
                            ));
                        }
                    }
                }
            }
        }

        // Now position and render the popups
        for (popup_id, button_rect, alignment) in popups_to_render {
            // Collect widget update info in a separate scope
            let widget_updates = {
                let Some(popup_node) = self.nodes.get_mut(&popup_id) else {
                    continue;
                };

                // Try to get size from ParamList
                let Some(param_list) = popup_node
                    .view
                    .as_any_mut()
                    .downcast_mut::<crate::ui::ParamList>()
                else {
                    continue;
                };

                let popup_size = param_list.get_size();

                // Calculate position (simplified bounds checking - assumes screen is large enough)
                let [btn_x, btn_y, btn_w, btn_h] = button_rect;
                let gap = 4.0;

                let (x, y) = match alignment {
                    crate::ui::PopupAlignment::Right => (btn_x + btn_w + gap, btn_y),
                    crate::ui::PopupAlignment::Left => (btn_x - popup_size[0] - gap, btn_y),
                    crate::ui::PopupAlignment::Bottom => (btn_x, btn_y + btn_h + gap),
                    crate::ui::PopupAlignment::Top => (btn_x, btn_y - popup_size[1] - gap),
                };

                param_list.set_position(x, y);

                // Collect child widget rects
                let children = popup_node.children.clone();
                let popup_x = param_list.style.rect[0];
                let popup_y = param_list.style.rect[1];
                let num_items = param_list.items.len();

                // First N children match param_list.items - get their rects from ParamList
                let mut updates = Vec::new();
                for (index, child_id) in children.iter().enumerate() {
                    if index < num_items {
                        // This is a ParamList item - get its rect from ParamList
                        let widget_rect = param_list.get_widget_rect(index, 180.0);
                        updates.push((*child_id, Some(widget_rect), popup_x, popup_y));
                    } else {
                        // This is an additional child (not a ParamList item)
                        updates.push((*child_id, None, popup_x, popup_y));
                    }
                }
                updates
            }; // Borrow of popup_node ends here

            // Now update child widgets
            for (child_id, widget_rect_opt, popup_x, popup_y) in widget_updates {
                if let Some(child_node) = self.nodes.get_mut(&child_id) {
                    if let Some(widget_rect) = widget_rect_opt {
                        // This is a ParamList item - position it using the rect from ParamList
                        if let Some(slider) = child_node
                            .view
                            .as_any_mut()
                            .downcast_mut::<crate::ui::Slider>()
                        {
                            slider.set_rect(widget_rect);
                        } else if let Some(btn_group) = child_node
                            .view
                            .as_any_mut()
                            .downcast_mut::<crate::ui::ButtonGroup>()
                        {
                            // ButtonGroup as ParamList item - use the rect from ParamList
                            btn_group.style.rect = widget_rect;
                        }
                    } else {
                        // This is an additional child (not a ParamList item)
                        if let Some(btn_group) = child_node
                            .view
                            .as_any_mut()
                            .downcast_mut::<crate::ui::ButtonGroup>()
                        {
                            // Store original relative position on first use
                            if btn_group.original_rect.is_none() {
                                btn_group.original_rect = Some(btn_group.style.rect);
                            }

                            // Position ButtonGroup relative to popup using original coordinates
                            let [rel_x, rel_y, w, h] = btn_group.original_rect.unwrap();
                            btn_group.style.rect = [popup_x + rel_x, popup_y + rel_y, w, h];
                        }
                    }
                }
            }

            self.build_node(popup_id, out, 100, text_cache); // High layer for popups
        }
    }

    /// Close all open popups (call this when clicking outside)
    pub fn close_all_popups(&mut self) {
        use crate::ui::Button;

        for node in self.nodes.values_mut() {
            if let Some(button) = node.view.as_any_mut().downcast_mut::<Button>() {
                if button.is_popup_visible() {
                    button.hide_popup();
                    self.dirty = true;
                }
            }
        }
    }

    /// Check if a click is inside any button with a popup or its popup content
    /// Returns true if inside, false if outside (should close popups)
    pub fn is_click_inside_popup_system(&self, pos: [f32; 2]) -> bool {
        use crate::ui::{Button, ParamList};

        for node in self.nodes.values() {
            if let Some(button) = node.view.as_any().downcast_ref::<Button>() {
                if button.is_popup_visible() {
                    // Check if click is on the button itself
                    let [bx, by, bw, bh] = button.style.rect;
                    if pos[0] >= bx && pos[0] <= bx + bw && pos[1] >= by && pos[1] <= by + bh {
                        return true;
                    }

                    // Check if click is inside the popup content
                    if let Some(popup_id) = button.popup_content {
                        if let Some(popup_node) = self.nodes.get(&popup_id) {
                            // Check if it's a ParamList and if click is inside
                            if let Some(param_list) =
                                popup_node.view.as_any().downcast_ref::<ParamList>()
                            {
                                let [px, py, pw, ph] = param_list.style.rect;
                                if pos[0] >= px
                                    && pos[0] <= px + pw
                                    && pos[1] >= py
                                    && pos[1] <= py + ph
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }
}
