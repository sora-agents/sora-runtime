# Tool Metadata
category: Lab / Manipulation
id: robotic-arm
wot_td: urn:cherrybot

# Functional Description
A 6-axis robotic arm (the cherryBot) with a parallel gripper, mounted at the edge of the workbench.
Its tool-center point (TCP) is a full 6-DOF pose: position plus orientation. This manual is the
protocol-agnostic, semantic half of the tool's description; the matching WoT Thing Description
(urn:cherrybot) carries the protocol binding — HTTP forms and API-key security. The two are
complementary, reconciled by tool type — an adapter maps urn:cherrybot to this manual's id
robotic-arm. See ADR-0015.

# Observable Properties
- tcp (object): current TCP pose — coordinate [x, y, z] in millimetres (x, y in -720..720; z in
  -178.3..1010) and rotation [roll, pitch, yaw] in degrees (each in -180..180).
- gripper (integer, 0-800): gripper aperture; 0 is fully closed, 800 is fully open.

# Signals
- target_reached: emitted when a move_to operation's target pose is physically reached. This is a
  semantic affordance the manual adds: the cherryBot reports completion via a webhook the WoT TD
  does not yet model, and the runtime surfaces it as this signal.

# Operations
- move_to(speed, target): moves the TCP to the given 6-DOF target pose at the given speed. speed is
  an integer in 10..400; target is a pose object with the same coordinate + rotation shape as the
  tcp property.
  - Behavior: long-running — physical motion that takes real time; completion is signalled by
    target_reached.
  - Effects: repositions the TCP to target and updates the tcp property.
- open_gripper(): opens the gripper fully (aperture 800).
  - Effects: sets gripper to 800.
- close_gripper(): closes the gripper fully (aperture 0).
  - Effects: sets gripper to 0.

# Usage Protocols & Safety
An operator must be registered before the arm accepts motion commands (the cherryBot TD exposes
registerOperator / removeOperator for this). move_to is a physical motion that takes real time:
after invoking it, suspend the activity and wait for the target_reached signal before invoking
close_gripper, open_gripper, or another move_to.
