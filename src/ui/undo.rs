use crate::SceneVM;
use crate::ui::Workspace;

/// Trait for undoable/redoable commands
pub trait UndoCommand: std::fmt::Debug {
    /// Execute the command
    /// `is_new` is true when the command is first executed, false when redoing
    /// This prevents re-applying UI actions that were just performed
    fn execute(&mut self, vm: &mut SceneVM, workspace: &mut Workspace, is_new: bool);

    /// Reverse the command (for undo)
    fn undo(&mut self, vm: &mut SceneVM, workspace: &mut Workspace);

    /// Optional: merge with next command if they're related (e.g., consecutive slider drags)
    /// Returns true if merge was successful
    fn try_merge(&mut self, _other: &dyn UndoCommand) -> bool {
        false
    }

    /// Command description for UI display (e.g., "Change Slider" or "Select Tool")
    fn description(&self) -> &str;

    /// Helper for downcasting in try_merge
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Undo/Redo stack manager
pub struct UndoStack {
    commands: Vec<Box<dyn UndoCommand>>,
    current_index: usize, // Points to the next command to redo
    max_size: usize,
    dirty: bool,
}

impl UndoStack {
    /// Create a new undo stack with a maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            commands: Vec::new(),
            current_index: 0,
            max_size,
            dirty: false,
        }
    }

    /// Add a new command and execute it
    pub fn execute(
        &mut self,
        mut cmd: Box<dyn UndoCommand>,
        vm: &mut SceneVM,
        workspace: &mut Workspace,
    ) {
        // Truncate any commands after current position (user did undo then new action)
        self.commands.truncate(self.current_index);

        // Try to merge with previous command (e.g., consecutive slider drags)
        if let Some(last) = self.commands.last_mut() {
            if last.try_merge(cmd.as_ref()) {
                self.dirty = true;
                return;
            }
        }

        // Execute the command (is_new = true, don't re-apply the UI action)
        cmd.execute(vm, workspace, true);

        // Add to stack
        self.commands.push(cmd);
        self.current_index += 1;
        self.dirty = true;

        // Enforce max size
        if self.commands.len() > self.max_size {
            self.commands.remove(0);
            self.current_index = self.current_index.saturating_sub(1);
        }
    }

    /// Undo the last command
    pub fn undo(&mut self, vm: &mut SceneVM, workspace: &mut Workspace) -> bool {
        if self.current_index == 0 {
            return false;
        }

        self.current_index -= 1;
        self.commands[self.current_index].undo(vm, workspace);
        self.dirty = true;
        workspace.set_dirty();
        true
    }

    /// Redo the next command
    pub fn redo(&mut self, vm: &mut SceneVM, workspace: &mut Workspace) -> bool {
        if self.current_index >= self.commands.len() {
            return false;
        }

        // is_new = false for redo (apply the UI action)
        self.commands[self.current_index].execute(vm, workspace, false);
        self.current_index += 1;
        self.dirty = true;
        workspace.set_dirty();
        true
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        self.current_index > 0
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        self.current_index < self.commands.len()
    }

    /// Clear the entire undo stack
    pub fn clear(&mut self) {
        self.commands.clear();
        self.current_index = 0;
        self.dirty = false;
    }

    /// Get description of next undo action
    pub fn undo_description(&self) -> Option<&str> {
        if self.can_undo() {
            Some(self.commands[self.current_index - 1].description())
        } else {
            None
        }
    }

    /// Get description of next redo action
    pub fn redo_description(&self) -> Option<&str> {
        if self.can_redo() {
            Some(self.commands[self.current_index].description())
        } else {
            None
        }
    }

    /// Check if stack has unsaved changes
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark stack as saved (clear dirty flag)
    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    /// Get number of commands in stack
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if stack is empty
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

// ============================================================================
// Concrete Command Implementations
// ============================================================================

/// Command for slider value changes (supports merging)
#[derive(Debug, Clone)]
pub struct SliderChangeCommand {
    widget_id: String,
    old_value: f32,
    new_value: f32,
    description: String,
}

impl SliderChangeCommand {
    pub fn new(widget_id: String, old_value: f32, new_value: f32) -> Self {
        Self {
            description: format!("Change {}", widget_id),
            widget_id,
            old_value,
            new_value,
        }
    }
}

impl UndoCommand for SliderChangeCommand {
    fn execute(&mut self, _vm: &mut SceneVM, workspace: &mut Workspace, is_new: bool) {
        // Only apply if this is a redo (is_new = false)
        // For new commands, the UI already updated the slider
        if !is_new {
            workspace.set_slider_value(&self.widget_id, self.new_value);
        }
    }

    fn undo(&mut self, _vm: &mut SceneVM, workspace: &mut Workspace) {
        workspace.set_slider_value(&self.widget_id, self.old_value);
    }

    fn try_merge(&mut self, other: &dyn UndoCommand) -> bool {
        // Try to merge consecutive slider changes for the same widget
        if let Some(other_slider) = other.as_any().downcast_ref::<SliderChangeCommand>() {
            if self.widget_id == other_slider.widget_id {
                self.new_value = other_slider.new_value;
                return true;
            }
        }
        false
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Command for button group selection changes
#[derive(Debug, Clone)]
pub struct ButtonGroupChangeCommand {
    group_id: String,
    old_index: usize,
    new_index: usize,
}

impl ButtonGroupChangeCommand {
    pub fn new(group_id: String, old_index: usize, new_index: usize) -> Self {
        Self {
            group_id,
            old_index,
            new_index,
        }
    }
}

impl UndoCommand for ButtonGroupChangeCommand {
    fn execute(&mut self, _vm: &mut SceneVM, workspace: &mut Workspace, is_new: bool) {
        if !is_new {
            workspace.set_buttongroup_index(&self.group_id, self.new_index);
        }
    }

    fn undo(&mut self, _vm: &mut SceneVM, workspace: &mut Workspace) {
        workspace.set_buttongroup_index(&self.group_id, self.old_index);
    }

    fn description(&self) -> &str {
        "Change Button Group"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Command for dropdown selection changes
#[derive(Debug, Clone)]
pub struct DropdownChangeCommand {
    dropdown_id: String,
    old_index: usize,
    new_index: usize,
}

impl DropdownChangeCommand {
    pub fn new(dropdown_id: String, old_index: usize, new_index: usize) -> Self {
        Self {
            dropdown_id,
            old_index,
            new_index,
        }
    }
}

impl UndoCommand for DropdownChangeCommand {
    fn execute(&mut self, _vm: &mut SceneVM, workspace: &mut Workspace, is_new: bool) {
        if !is_new {
            workspace.set_dropdown_index(&self.dropdown_id, self.new_index);
        }
    }

    fn undo(&mut self, _vm: &mut SceneVM, workspace: &mut Workspace) {
        workspace.set_dropdown_index(&self.dropdown_id, self.old_index);
    }

    fn description(&self) -> &str {
        "Change Dropdown"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Command for button toggle changes
#[derive(Debug, Clone)]
pub struct ButtonToggleCommand {
    #[allow(dead_code)]
    button_id: String,
    #[allow(dead_code)]
    old_state: bool,
    #[allow(dead_code)]
    new_state: bool,
}

impl ButtonToggleCommand {
    pub fn new(button_id: String, old_state: bool, new_state: bool) -> Self {
        Self {
            button_id,
            old_state,
            new_state,
        }
    }
}

impl UndoCommand for ButtonToggleCommand {
    fn execute(&mut self, _vm: &mut SceneVM, _workspace: &mut Workspace, _is_new: bool) {
        // Note: Workspace doesn't have a set_button_state method yet
        // Apps will need to implement this or handle it in their own state
    }

    fn undo(&mut self, _vm: &mut SceneVM, _workspace: &mut Workspace) {
        // Apps will need to implement button state restoration
    }

    fn description(&self) -> &str {
        "Toggle Button"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Command for color changes
#[derive(Debug, Clone)]
pub struct ColorChangeCommand {
    widget_id: String,
    #[allow(dead_code)]
    old_color: [f32; 4],
    new_color: [f32; 4],
}

impl ColorChangeCommand {
    pub fn new(widget_id: String, old_color: [f32; 4], new_color: [f32; 4]) -> Self {
        Self {
            widget_id,
            old_color,
            new_color,
        }
    }
}

impl UndoCommand for ColorChangeCommand {
    fn execute(&mut self, _vm: &mut SceneVM, _workspace: &mut Workspace, _is_new: bool) {
        // Apps will need to implement color state restoration
    }

    fn undo(&mut self, _vm: &mut SceneVM, _workspace: &mut Workspace) {
        // Apps will need to implement color state restoration
    }

    fn try_merge(&mut self, other: &dyn UndoCommand) -> bool {
        // Merge consecutive color changes for the same widget
        if let Some(other_color) = other.as_any().downcast_ref::<ColorChangeCommand>() {
            if self.widget_id == other_color.widget_id {
                self.new_color = other_color.new_color;
                return true;
            }
        }
        false
    }

    fn description(&self) -> &str {
        "Change Color"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Full state snapshot command for complex operations
/// Uses JSON serialization from app_trait
#[derive(Debug, Clone)]
pub struct StateSnapshotCommand {
    description: String,
    #[allow(dead_code)]
    old_state: String, // JSON snapshot
    #[allow(dead_code)]
    new_state: String, // JSON snapshot
}

impl StateSnapshotCommand {
    pub fn new(description: String, old_state: String, new_state: String) -> Self {
        Self {
            description,
            old_state,
            new_state,
        }
    }
}

impl UndoCommand for StateSnapshotCommand {
    fn execute(&mut self, _vm: &mut SceneVM, _workspace: &mut Workspace, _is_new: bool) {
        // Note: This needs to be called through the app instance
        // The app's load_from_json method should be called here
        // This is a placeholder - apps will need to implement the actual loading
    }

    fn undo(&mut self, _vm: &mut SceneVM, _workspace: &mut Workspace) {
        // Note: This needs to be called through the app instance
        // The app's load_from_json method should be called with old_state
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
