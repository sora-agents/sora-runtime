# Tool Metadata
category: Lab / Environment Control
id: blinds

# Functional Description
Motorized blinds covering the workbench's window, controlling ambient light.

# Observable Properties
- position (integer, 0-100): current blind position; 0 is fully closed, 100 is fully open.

# Signals
(none)

# Operations
- set_position(level: integer 0-100): moves the blinds to the given position.

# Usage Protocols & Safety
set_position completes synchronously; no suspension needed. Check `position` to confirm the move.
