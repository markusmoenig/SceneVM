use uuid::Uuid;

/// Pointer/mouse/touch events in logical coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UiEventKind {
    PointerDown,
    PointerUp,
    PointerMove,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiEvent {
    pub kind: UiEventKind,
    pub pos: [f32; 2],
    /// Identifier for the pointer/gesture (0 for mouse primary).
    pub pointer_id: u32,
}

impl UiEvent {
    pub fn new(kind: UiEventKind, pos: [f32; 2]) -> Self {
        Self {
            kind,
            pos,
            pointer_id: 0,
        }
    }
}

/// Actions emitted by UI elements.
#[derive(Debug, Clone, PartialEq)]
pub enum UiAction {
    ButtonPressed(Uuid),
    ButtonToggled(Uuid, bool),
}

/// Result of handling an event: whether a view dirtied itself and any actions.
#[derive(Debug, Default)]
pub struct UiEventOutcome {
    pub dirty: bool,
    pub actions: Vec<UiAction>,
}

impl UiEventOutcome {
    pub fn none() -> Self {
        Self {
            dirty: false,
            actions: Vec::new(),
        }
    }

    pub fn dirty() -> Self {
        Self {
            dirty: true,
            actions: Vec::new(),
        }
    }

    pub fn with_action(action: UiAction) -> Self {
        Self {
            dirty: true,
            actions: vec![action],
        }
    }

    pub fn merge(&mut self, other: UiEventOutcome) {
        self.dirty |= other.dirty;
        self.actions.extend(other.actions);
    }
}
