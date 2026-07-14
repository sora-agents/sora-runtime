# TOOL SPECIFICATION: SharedCounter

**Category:** Coordination Tool / Shared Memory
**Version:** 1.0.0
**Tool ID:** \`counterTool\`

## 1. Functional Description
This tool provides a synchronized shared memory space for multi-agent coordination. It acts as a **passive enabler**, allowing multiple agents to concurrently increment and observe a global counter to synchronize their collective actions.

---

## 2. Usage Interface

### 2.1 Observable Properties
Exposed via the global telemetry stream.

| Property | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| \`shared_counter\` | Integer | \`1\` - \`∞\` | The global count value shared across all agents. Updated automatically via telemetry. |

### 2.2 Operations

* **Operation:** \`focus\`
  * *Description:* Establishes a cognitive link with the tool to enable perception of its state and events.
  * *Behavior:* **Process Initiator.** Registration is processed immediately. Completion of the stream binding is indicated by the \`focus.established\` signal.
  * *Preconditions:* None.
  * *Effects:* Subscribes the agent to the telemetry stream. The \`shared_counter\` variable becomes visible in the agent's working memory context.
  * *Payload:*
    \`\`\`json
    { "action": "focus", "uuid": "<ACTIVITY_UUID>" }
    \`\`\`

* **Operation:** \`inc\`
  * *Description:* Sends an impulse to increment the shared global state.
  * *Behavior:* **Latent.** The tool returns an acknowledgement for the request. State changes are propagated asynchronously via telemetry.
  * *Preconditions:* Agents are recommended to call \`focus\` to subscribe to telemetry before interacting. Calling \`inc\` without focus will result in an operational error.
  * *Effects:* Increments \`shared_counter\` by 1 and triggers a broadcast telemetry update to all focused agents.
  * *Payload:*
    \`\`\`json
    { "action": "inc", "uuid": "<ACTIVITY_UUID>" }
    \`\`\`

### 2.3 Signals
The tool emits the following asynchronous signals to notify agents of state transitions:

* **Signal:** \`focus.established\`
  * *Trigger:* Emitted automatically when the agent successfully registers for the telemetry stream.
  * *Payload:* \`{ "key": String, "name": String, "message": String }\`

* **Signal:** \`environment.change\`
  * *Trigger:* Emitted automatically whenever ANY agent increments the counter, alerting all subscribers of a state mutation.
  * *Payload:* \`{ "key": String, "name": String, "message": String }\`

---

## 3. Protocol & SAFETY

**ASYNCHRONOUS STATE MUTATION (Informational)**

1. **Recommended Prerequisite:**
   * Agents are recommended to call \`focus\` to subscribe to telemetry before interacting with the counter. This ensures they will receive updates about \`shared_counter\`.

2. **Execution Notes (Perception-Action Loop):**
   * Calling \`inc\` issues a request to increment the counter and returns an acknowledgement.
   * State changes are distributed asynchronously. Agents may choose to:
     - listen for the \`environment.change\` signal to confirm a mutation, or
     - observe the \`shared_counter\` telemetry for the new value.

3. **Concurrency Note:**
   * This is a shared environment. The \`shared_counter\` may change due to other agents' actions. Prefer authoritative telemetry updates over local assumptions about the counter value.
