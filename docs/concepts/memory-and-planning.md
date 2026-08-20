# Memory & Planning

## Memory

The [CoALA framework](https://arxiv.org/abs/2309.02427) distinguishes between short and long-term memory.

Short-term memory, or **working memory**, maintains the agent's ongoing activities, perceptual input, and other contextual knowledge relevant to the current decision cycle. It is transient and optimized for speed (in-process).

Long-term memory modules are optional, persistent, and pluggable (e.g., a file-backed implementation to start, with database or vector-store backends as drop-in alternatives). They have their well-defined place in the S-ORA decision cycle:

- semantic memory: captures the agent's long-term knowledge about the world and itself, including tool manuals and discovered tool/workspace records — new kinds of durable "world knowledge" belong here, not in a new memory module
- procedural memory: captures the procedural knowledge the agent can query to derive or revise a plan for its current activity; this includes implicit knowledge encoded in LLM weights, and explicit knowledge captured as skills or plans — a plan is a multi-step, goal-indexed artifact, deliberately reusable across activities with similar goals, not something regenerated every cycle
- episodic memory: stores relevant experiences, such as successful activity completions, which may be retrieved for guidance in future activities

## Action space

Two types of actions: **internal actions**, for interacting with memory modules; and **external actions**, for interacting with the external world. The action space is extensible — agents and downstream frameworks can register additional internal or external actions beyond the predefined set below.

Predefined internal actions:

- **semantic memory**: _retrieve_ and _store_ tool manuals
- **working memory**: _load_ and _unload_ tool manuals from semantic memory; _filter_ perceptual input relevant to the current activity; _create_ a new activity from an unhandled message; _suspend_ and _resume_ an activity
- **procedural memory**: _retrieve_ a plan of action for the current activity, _infer_ one if a suitable one is not already known, or _store_ one that was actually followed to a successful completion (auto-store/reuse is currently disabled — the default cycle infers a fresh plan each activity, since replaying a stored plan verbatim is unsound; the operations remain for distilling reusable procedures from episodes)
- **episodic memory**: _learn_ from experience by saving a summary of an activity completion, or _consult_ previous experiences

Predefined external actions:

- _invoke_ a tool operation
- _join_ a configured workspace (connects and registers its tools) and _leave_ one (closes the connection)
- _retrieve_ manuals from external repositories
- _focus_ on and _unfocus_ from tools to perceive observable properties and signals
- _send_ messages to other agents, via a pluggable protocol (e.g., A2A, plain HTTP)
