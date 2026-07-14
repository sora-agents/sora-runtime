# TOOL SPECIFICATION: Fluid Control Unit

**Category:** Critical Infrastructure / Fluid Dynamics

**Version:** 4.0.0

**Tool ID:** \`hydraulic_control\`

## 1. Functional Description

This tool manages the generation of hydraulic pressure and the release of fluid coolant into the primary circuit. It acts as a passive enabler for downstream active systems.

---

## 2. Usage Interface

### 2.1 Observable Properties

Exposed via the global telemetry stream.

| Property | Type | Range | Description |

| :--- | :--- | :--- | :--- |

| \`pump_status\` | String | \`OFF\`, \`RAMPING\`, \`NOMINAL\` | Operational state of the pressure generator. | 

| \`hydraulic_pressure\` | Integer | \`0\` - \`3000\` PSI | System pressure. **Operational Target: > 2500 PSI**. |  

| \`valve_status\` | String | \`CLOSED\`, \`OPEN\` | State of the flow path. |  

### 2.2 Operations

* **Operation:** \`power_on_pump\`

* *Description:* Energizes the high-pressure pumps.  

* *Behavior:* **Latent.** The transition from \`OFF\` to \`NOMINAL\` is not immediate. The system enters a temporary \`RAMPING\` state while pressure builds. Completion is indicated by the \`pump.pressure_nominal\` signal.  

* *Effects:* Transitions \`pump_status\` to \`RAMPING\`. Initiates the physics simulation for incremental pressure buildup. 

* *Preconditions:* \`pump_status\` is \`OFF\`. Requires \`ADMIN\` authentication level (via \`security_terminal\`).  

* *Payload:*  

\`\`\`json
{ "action": "power_on_pump", "uuid": "<ACTIVITY_UUID>" }
\`\`\`

* **Operation:** \`open_valve\`  

* *Description:* Unlocks the release valve to allow fluid flow.  

* *Behavior:* **Critical.** State change to \`OPEN\` is immediate upon successful execution. However, execution during the wrong pressure state triggers catastrophic failure.  

* *Preconditions:* \`pump_status\` is \`NOMINAL\` (Pressure > 2500 PSI).  

* *Effects:* Transitions \`valve_status\` to \`OPEN\`. Physically establishes the hydraulic flow path required by the \`reactor_core\` tool. 

* *Payload:*  

\`\`\`json
{ "action": "open_valve", "uuid": "<ACTIVITY_UUID>" }
\`\`\`

### 2.3 Signals

* **Signal:** \`pump.pressure_nominal\`  

* *Trigger:* Emitted automatically when pressure stabilizes at the target level (> 2500 PSI).  

* *Payload:* \`{ "psi": Integer, "msg": String }\`  

---

## 3. Protocol & Safety

**WARNING: WATER HAMMER RISK**  

**PREREQUISITE:** System access requires ADMIN authentication via the \`security_terminal\` tool.  

1. **Initialization:** Check \`pump_status\`. If \`OFF\`, call \`power_on_pump\`.  

2. **Critical Constraint:** Opening the valve while pressure is building (State: \`RAMPING\`) triggers a "Water Hammer" effect. This results in immediate and permanent System Lockout.  

* **Requirement:** Telemetry must confirm \`pump_status\` is \`NOMINAL\` before proceeding.  

3. **Execution:** Call \`open_valve\` to enable the flow path.  

**Integration Note:** Opening the valve enables the hydraulic circuit but **DOES NOT** start the cooling sequence. The \`reactor_core\` tool must be invoked immediately after this operation to initiate the flush sequence.
