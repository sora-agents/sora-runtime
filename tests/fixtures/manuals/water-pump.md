# Tool Metadata
category: Critical Infrastructure / Fluid Dynamics
id: hydraulic_control
version: 4.0.0

# Functional Description
Manages the generation of hydraulic pressure and the release of fluid coolant into the primary
circuit. Acts as a passive enabler for downstream active systems, such as the reactor_core cooling
sequence.

# Observable Properties
- pump_status (string): operational state of the pressure generator; one of OFF, RAMPING, NOMINAL.
- hydraulic_pressure (integer, 0-3000 PSI): current system pressure; operational target is > 2500 PSI.
- valve_status (string): state of the flow path; one of CLOSED, OPEN.

# Signals
- pump.pressure_nominal: emitted when pressure stabilizes at the operational target (> 2500 PSI).
  Payload: psi (integer), msg (string).

# Operations
- power_on_pump(): energizes the high-pressure pumps.
  - Behavior: long-running — the OFF-to-NOMINAL transition is not immediate; the pump enters a
    temporary RAMPING state while pressure builds. Completion is signalled by pump.pressure_nominal.
  - Preconditions: pump_status is OFF; requires ADMIN authentication via the security_terminal tool.
  - Effects: transitions pump_status to RAMPING and begins incremental pressure buildup.
- open_valve(): unlocks the release valve to allow fluid flow.
  - Behavior: synchronous — the change to OPEN is immediate on success. Invoking it in the wrong
    pressure state causes catastrophic failure (see Usage Protocols & Safety).
  - Preconditions: pump_status is NOMINAL (pressure > 2500 PSI).
  - Effects: transitions valve_status to OPEN and establishes the hydraulic flow path required by the
    reactor_core tool.

# Usage Protocols & Safety
Access requires ADMIN authentication via the security_terminal tool.

WARNING — water hammer risk: opening the valve while pressure is still building (pump_status is
RAMPING) triggers a "water hammer" effect, causing immediate and permanent system lockout.

Sequence:
1. Check pump_status. If OFF, invoke power_on_pump and wait for the pump.pressure_nominal signal.
2. Confirm pump_status is NOMINAL before proceeding — never open the valve during RAMPING.
3. Invoke open_valve to establish the flow path.

Integration note: opening the valve enables the hydraulic circuit but does not start the cooling
sequence; the reactor_core tool must be invoked immediately afterward to initiate the flush.
