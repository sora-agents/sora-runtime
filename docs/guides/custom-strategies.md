# Custom Strategies

## Customizing the planning/grounding prompts

`agent.yaml`'s `procedural:` block ([Your First Agent](../getting-started/first-agent.md#anatomy-of-agentyaml)) points at your own `PlanPrompt`/`GroundPrompt`
callables — each fully *replaces* the corresponding built-in (`default_plan_prompt` /
`default_ground_prompt`), it doesn't patch pieces of it; write one to change wording, tone, or the
cost/quality tradeoff of planning and grounding. For example, the built-in prompt's default
guidance for a `send` step reporting a not-yet-known result is a `$decide` reference — a natural
sentence, but it costs one extra `ProceduralMemory.ground()` model call at run time (see
`PlanPrompt` in the [API Sketch](https://github.com/sora-agents/sora-runtime/blob/main/README.md#api-sketch)). A stricter prompt can trade that phrasing for a free, mechanical
`$from` copy:

    # my_agent/prompts.py
    from sora.memory import PLAN_SYSTEM_PROMPT, render_tools

    def cheap_plan_prompt(activity, tools, observed, messages):
        system = PLAN_SYSTEM_PROMPT + (
            "\nPrefer a bare $from copy over $decide phrasing for send content, even if it reads "
            "less like a sentence — minimizing model calls matters more than prose here."
        )
        user = f"Goal: {activity.goal}\n\nAvailable tools:\n{render_tools(tools)}"
        return system, user

    # my_agent/agent.yaml
      procedural:
        plan_prompt: my_agent.prompts.cheap_plan_prompt

`ground_prompt` follows the same shape (`GroundPrompt` in the [API Sketch](https://github.com/sora-agents/sora-runtime/blob/main/README.md#api-sketch)), for customizing the
grounding escalation itself rather than what the plan asks it to do.

## See also

- [Design note — Reasoning strategy extension](../architecture/notes/reasoning-strategy-extension.md)
- [ADR-0010 — Pluggable phase strategies](../architecture/adrs/0010-pluggable-phase-strategies.md)
