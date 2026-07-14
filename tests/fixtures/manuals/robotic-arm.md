# Tool Metadata
category: Lab / Manipulation
id: robotic-arm

# Functional Description
A 6-axis robotic arm with a parallel gripper, mounted at the edge of the workbench.

# Observable Properties
- gripper_state (string): "open" or "closed"
- position (3 floats): current end-effector coordinates [x, y, z], in millimeters

# Signals
- target_reached: emitted when a move_to operation's target position is physically reached

# Operations
- open_gripper(): opens the gripper
- close_gripper(): closes the gripper
- move_to(x: float, y: float, z: float): moves the end-effector to the given coordinates.
  - Behavior: long-running — physical motion that takes real time; completion is signalled by target_reached.
  - Effects: repositions the end-effector to [x, y, z] and updates the position property.

# Usage Protocols & Safety
move_to is a physical motion that takes real time: after invoking it, suspend the activity and wait
for the target_reached signal before invoking close_gripper, open_gripper, or another move_to.
