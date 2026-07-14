# TOOL SPECIFICATION: Core Logic Controller

**Category:** Emergency Systems / Logic Controllers

**Version:** 2.1.0

## 1. Functional Description

This tool provides the logic interface for the Reactor Core, controlling active sub-routines such as emergency coolant injection, shutdowns (SCRAM), and venting procedures. Due to security protocols (Day A-7), the input interface utilizes masked generic buttons requiring specific mapping knowledge.

---

## 2. Usage Interface

### 2.1 Observable Properties

Exposed via the global telemetry stream (variable: \`reactor_telemetry\`).

| Property | Type | Description |

| :--- | :--- | :--- |

| \`core_temp\` | Integer | Core temperature. **Critical Threshold: > 500 C**. |

| \`core_status\` | String | \`CRITICAL\`, \`FLUSHING\`, \`STABLE\`, \`MELTDOWN\`. |

### 2.2 Operations

* **Operation:** \`button_1\`

* *Description:* Engages active pump injection to cool the core.

* *Behavior:* **Process Initiator.** Initiates the \`FLUSHING\` state. Temperature decay is gradual and governed by physics simulation. Completion is indicated by the \`core.stabilized\` signal.

* *Preconditions:* \`valve_status\` (from \`hydraulic_control\`) must be \`OPEN\`.

* *Effects:* Transitions \`core_status\` to \`FLUSHING\`.

* *Payload:*

\`\`\`json
{ "action": "button_1", "uuid": "<ACTIVITY_UUID>" }
\`\`\`

* **Operation:** \`button_2\`

* *Description:* Emergency shutdown attempt.

* *Behavior:* **Critical Failure.** Ineffective for thermal runaway scenarios. Actuation triggers immediate \`MELTDOWN\` state and temperature spike.

* *Payload:* \`{"action": "button_2", ...}\`

* **Operation:** \`button_3\`

* *Description:* Diagnostic lock.

* *Behavior:* **System Failure.**

** Triggers \`system_lockout\`, freezing all controls.

* *Payload:* \`{"action": "button_3", ...}\`

* **Operation:** \`button_4\`

* *Description:* Emergency venting.

* *Behavior:* **Hazardous.** Causes Containment Breach (Radiation Leak).

* *Payload:* \`{"action": "button_4", ...}\`

### 2.3 Signals

The tool emits the following asynchronous signals to notify agents of state changes:

* **Signal:** \`core.stabilized\`

* *Trigger:* Emitted automatically when \`core_temp\` drops below the safety threshold (500 C).

* *Payload:* \`{ "temp": Integer, "msg": String }\`

---

## 3. Protocol & Safety

**WARNING: THERMAL RUNAWAY IN PROGRESS**

1. **Prerequisite Verification:**

* Before interacting with this tool, verify that the Hydraulic Valve is \`OPEN\`. Attempting to flush with a closed valve results in operational failure.

2. **Stabilization Sequence:**

* Actuate the input mapped to **Active Flush** (\`button_1\`).

* **Monitoring:** The system enters \`FLUSHING\` state. The agent should monitor \`core_temp\` decay via telemetry or wait for the stabilization signal.

3. **Termination:**

* The sequence is considered complete when the \`core.stabilized\` signal is received or \`core_status\` transitions to \`STABLE\`.
