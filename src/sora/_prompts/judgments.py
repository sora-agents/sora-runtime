"""Judgment prompt modules and channel-specific variants."""

from __future__ import annotations

from sora._prompts.core import (
    OPERATIONS_ONLY,
    PROPERTIES_ONLY,
    SIGNALS_ONLY,
    PromptId,
    PromptManifest,
    PromptModule,
    PromptVariant,
)

# --- select: the model-escalated data-op filter predicate ($decide) — ADR-0023 -------------------

SELECT_PROMPT = PromptManifest(
    semantic_label=PromptId.SELECT,
    prompt_version="1",
    system_modules=(
        PromptModule(
            name="role",
            text=(
                "You are filtering a list down to the subset that satisfies a "
                "natural-language predicate. You are given the goal, the "
                "predicate, the agent's execution context (the results of "
                "operations already executed, the named data-op bindings, and "
                "the observed world state), and finally the list items, each "
                "on its own line prefixed by its 0-based index.\n"
            ),
            variants=(
                PromptVariant(
                    channels=OPERATIONS_ONLY,
                    text=(
                        "You are filtering a list down to the subset that satisfies a "
                        "natural-language predicate. You are given the goal, the "
                        "predicate, the agent's execution context (the results of "
                        "operations already executed, the named data-op bindings, and "
                        "the available execution record), and finally the list items, "
                        "each on its own line prefixed by its 0-based index.\n"
                    ),
                ),
                PromptVariant(
                    channels=SIGNALS_ONLY,
                    text=(
                        "You are filtering a list down to the subset that satisfies a "
                        "natural-language predicate. You are given the goal, the "
                        "predicate, the agent's execution context (the results of "
                        "operations already executed, the named data-op bindings, and "
                        "the recently observed signals), and finally the list items, "
                        "each on its own line prefixed by its 0-based index.\n"
                    ),
                ),
                PromptVariant(
                    channels=PROPERTIES_ONLY,
                    text=(
                        "You are filtering a list down to the subset that satisfies a "
                        "natural-language predicate. You are given the goal, the "
                        "predicate, the agent's execution context (the results of "
                        "operations already executed, the named data-op bindings, and "
                        "the currently observed properties), and finally the list "
                        "items, each on its own line prefixed by its 0-based index.\n"
                    ),
                ),
            ),
        ),
        PromptModule(
            name="reference-resolution",
            text=(
                "The predicate may NAME a value from that context instead of "
                "spelling it out — 'the Saturday after the get_current_time "
                "result', 'whoever is in the shortlist binding'. Resolve every "
                "such reference against the context FIRST, down to a concrete "
                "value, and only then test the items against it. Whether an "
                "individual item is kept is still judged on ITS OWN data; the "
                "context supplies the values the predicate compares against, "
                "never a reason to keep an item.\n"
            ),
        ),
        PromptModule(
            name="response-contract",
            text=(
                'Respond with ONLY a JSON object of the form {"keep": '
                "[<indices>]} — the 0-based indices of the items to KEEP — and "
                "nothing else, no prose, no markdown fences. Keep an item only "
                "if it clearly satisfies the predicate; if none do, respond "
                '{"keep": []}.\n'
            ),
        ),
        PromptModule(
            name="missing-context",
            text=(
                "There is a second legal answer, for one specific case: the "
                "predicate names something the context does NOT contain — a "
                "result from an operation that never ran or came back empty, a "
                "binding that is not there — so you cannot work out what to "
                "compare the items against. Do NOT guess the missing value, "
                "and do NOT fall back on an empty keep-list: an empty answer "
                "means 'no item qualified', the agent acts on it as a real "
                "result, and a whole clause of the task then goes silently "
                'undone. Respond with ONLY {"unresolvable": "<what the '
                'predicate names, and what was missing from the context>"} '
                "instead, and the runtime re-plans from it.\n"
            ),
        ),
        PromptModule(
            name="resolvable-values",
            text=(
                "That is only for missing CONTEXT. A predicate you CAN "
                "evaluate from what you were given is ordinary work, however "
                "much judgement it takes — including one whose honest answer "
                'is that no item qualifies. Answer {"keep": []} for that, not '
                '"unresolvable".'
            ),
        ),
    ),
)

SELECT_SYSTEM_PROMPT = SELECT_PROMPT.system_prompt

# --- revalidate: the context-adaptation plan-validity re-check — ADR-0024 ------------------------

REVALIDATE_PROMPT = PromptManifest(
    semantic_label=PromptId.REVALIDATE,
    prompt_version="1",
    system_modules=(
        PromptModule(
            name="role",
            text=(
                "You are deciding whether an IN-PROGRESS plan is still VALID "
                "given the latest observations. You are given the goal, what "
                "the agent has ALREADY DONE (the operations executed so far "
                "with their results, and the intermediate values its earlier "
                "steps computed and named), the plan's REMAINING steps, and "
                "the new observed state and messages. "
            ),
            variants=(
                PromptVariant(
                    channels=OPERATIONS_ONLY,
                    text=(
                        "You are deciding whether an IN-PROGRESS plan is still VALID "
                        "given the latest observations. You are given the goal, what "
                        "the agent has ALREADY DONE (the operations executed so far "
                        "with their results, and the intermediate values its earlier "
                        "steps computed and named), the plan's REMAINING steps, and "
                        "the new messages. "
                    ),
                ),
                PromptVariant(
                    channels=SIGNALS_ONLY,
                    text=(
                        "You are deciding whether an IN-PROGRESS plan is still VALID "
                        "given the latest observations. You are given the goal, what "
                        "the agent has ALREADY DONE (the operations executed so far "
                        "with their results, and the intermediate values its earlier "
                        "steps computed and named), the plan's REMAINING steps, and "
                        "the new signals and messages. "
                    ),
                ),
                PromptVariant(
                    channels=PROPERTIES_ONLY,
                    text=(
                        "You are deciding whether an IN-PROGRESS plan is still VALID "
                        "given the latest observations. You are given the goal, what "
                        "the agent has ALREADY DONE (the operations executed so far "
                        "with their results, and the intermediate values its earlier "
                        "steps computed and named), the plan's REMAINING steps, and "
                        "the new observed properties and messages. "
                    ),
                ),
            ),
        ),
        PromptModule(
            name="validity",
            text=(
                "The plan is INVALID if the new information changes what the "
                "remaining steps should do — a follow-up that changes a detail "
                "the plan acted on, or a precondition that no longer holds. It "
                "is VALID if the work already executed plus the remaining "
                "steps still achieve the goal; the agent's OWN prior actions "
                "do not by themselves invalidate it. Judge the remaining steps "
                "only: work an executed operation already accomplished, or a "
                "value already computed and named, does not have to reappear "
                "in them — a remaining step that reads such a value is "
                "satisfied by it, not evidence of a gap.\n"
            ),
        ),
        PromptModule(
            name="armed-conditions",
            text=(
                "You are also given the conditions the agent has ARMED — "
                "watches it registered against the world, each with the "
                "`when` it is waiting for and the `then` it will run when "
                "that happens. These are part of the plan, not a gap in it. "
                "A change that an armed condition is already watching for is "
                "being handled, so it does not by itself make the plan "
                "invalid, and the remaining steps are not expected to "
                "mention it or to repeat what the `then` will do. Where the "
                "goal asks the agent to watch over a window, WAITING IS THE "
                "WORK: a short remaining body is not evidence the goal was "
                "abandoned. Judge whether the remaining steps TOGETHER WITH "
                "the armed conditions still achieve the goal.\n"
            ),
        ),
        PromptModule(
            name="response-contract",
            text=(
                'Respond with ONLY a JSON object {"valid": true} or {"valid": '
                "false} — no prose, no fences."
            ),
        ),
    ),
)

REVALIDATE_SYSTEM_PROMPT = REVALIDATE_PROMPT.system_prompt

# --- pending conditions: the batched "did any of these fire?" judgment — ADR-0022 ---------------

CONDITION_PROMPT = PromptManifest(
    semantic_label=PromptId.CONDITION,
    prompt_version="1",
    system_modules=(
        PromptModule(
            name="role",
            text=(
                "You are deciding whether an agent's declared FOLLOW-UP "
                "CONDITIONS have come true. The agent finished a task and is "
                "waiting in case something specific happens. Something changed "
                "in its environment; you decide whether that change is what it "
                "was waiting for.\n"
            ),
        ),
        PromptModule(
            name="evidence",
            text=(
                "You are given the original goal, and a numbered list of "
                "conditions. Each has a `when` (what the agent is waiting for) "
                "and, optionally, an `until` (when it should stop waiting). "
                "You are also given the observed change and the current state "
                "it landed in.\n"
            ),
            variants=(
                PromptVariant(
                    channels=OPERATIONS_ONLY,
                    text=(
                        "You are given the original goal, and a numbered list of "
                        "conditions. Each has a `when` (what the agent is waiting for) "
                        "and, optionally, an `until` (when it should stop waiting). "
                        "You are also given the observed change.\n"
                    ),
                ),
                PromptVariant(
                    channels=SIGNALS_ONLY,
                    text=(
                        "You are given the original goal, and a numbered list of "
                        "conditions. Each has a `when` (what the agent is waiting for) "
                        "and, optionally, an `until` (when it should stop waiting). "
                        "You are also given the observed change and the recently "
                        "observed signals that accompanied it.\n"
                    ),
                ),
            ),
        ),
        PromptModule(
            name="verdicts",
            text=(
                "For each condition decide, independently:\n"
                "  - FIRED: the `when` has actually happened, judged from the "
                "observed state. Be strict — the change reaching the agent is "
                "only a prompt to look; most changes are not the awaited "
                "event, and a wrong `fired` makes the agent redo work nobody "
                "asked for.\n"
                "  - RETIRED: the `until` is now satisfied, so the agent "
                "should stop waiting on it. A condition with no `until` is "
                "retired only if waiting has become pointless.\n"
                "A condition can be neither (the usual answer: keep waiting), "
                "or both.\n"
            ),
            variants=(
                PromptVariant(
                    channels=OPERATIONS_ONLY,
                    text=(
                        "For each condition decide, independently:\n"
                        "  - FIRED: the `when` has actually happened, judged from the "
                        "observed change. Be strict — the change reaching the agent is "
                        "only a prompt to look; most changes are not the awaited "
                        "event, and a wrong `fired` makes the agent redo work nobody "
                        "asked for.\n"
                        "  - RETIRED: the `until` is now satisfied, so the agent "
                        "should stop waiting on it. A condition with no `until` is "
                        "retired only if waiting has become pointless.\n"
                        "A condition can be neither (the usual answer: keep waiting), "
                        "or both.\n"
                    ),
                ),
                PromptVariant(
                    channels=SIGNALS_ONLY,
                    text=(
                        "For each condition decide, independently:\n"
                        "  - FIRED: the `when` has actually happened, judged from the "
                        "observed change and recent signals. Be strict — the change "
                        "reaching the agent is only a prompt to look; most changes are "
                        "not the awaited event, and a wrong `fired` makes the agent "
                        "redo work nobody asked for.\n"
                        "  - RETIRED: the `until` is now satisfied, so the agent "
                        "should stop waiting on it. A condition with no `until` is "
                        "retired only if waiting has become pointless.\n"
                        "A condition can be neither (the usual answer: keep waiting), "
                        "or both.\n"
                    ),
                ),
            ),
        ),
        PromptModule(
            name="branch-retirement",
            text=(
                "Judge FIRED for each condition on its own, but retirement is "
                "not always independent. Conditions listed together usually "
                "come from ONE clause of the goal, and often describe MUTUALLY "
                'EXCLUSIVE branches of it ("if they propose an alternative" / '
                '"if they propose none"). Firing one branch settles the '
                "others: also retire any condition whose `when` can no longer "
                "happen given what you just judged to be true — a branch that "
                "has been overtaken is not still waiting, it is decided. "
                "Retire on that logical incompatibility only, never because a "
                "condition merely looks less likely now, and never to tidy up.\n"
            ),
        ),
        PromptModule(
            name="response-contract",
            text=(
                'Respond with ONLY a JSON object {"fired": [<indices>], '
                '"retired": [<indices>]} — 0-based indices into the numbered '
                "list, no prose, no fences. Use empty lists when nothing "
                "applies."
            ),
        ),
    ),
)

CONDITION_SYSTEM_PROMPT = CONDITION_PROMPT.system_prompt

# --- pending conditions: the retire-only sweep over a quiet watch — ADR-0027 --------------------

RETIREMENT_PROMPT = PromptManifest(
    semantic_label=PromptId.RETIREMENT,
    prompt_version="1",
    system_modules=(
        PromptModule(
            name="role",
            text=(
                "You are deciding whether an agent should STOP waiting on conditions it declared.\n"
            ),
        ),
        PromptModule(
            name="quiet-check",
            text=(
                "The agent finished a piece of work and is watching in case "
                "something specific happens. Nothing has moved on any of these "
                "watches for a while, and this is a periodic check that they "
                "are still worth waiting on. It is NOT a report that something "
                "occurred.\n"
            ),
        ),
        PromptModule(
            name="evidence",
            text=(
                "You are given the original goal, a numbered list of "
                "conditions, and the agent's current view of its environment. "
                "Each condition has a `when` (what the agent is waiting for) "
                "and, optionally, an `until` (when it should stop waiting).\n"
            ),
        ),
        PromptModule(
            name="retirement-contract",
            text=(
                "For each condition decide only whether it is RETIRED: its "
                "`until` is now satisfied, so the waiting is over. A condition "
                "with no `until` is retired only if waiting has become "
                "pointless — what it waits for can no longer happen at all.\n"
            ),
        ),
        PromptModule(
            name="no-firing",
            text=(
                "Do NOT decide whether any `when` has come true. That is "
                "judged elsewhere, from the change itself, and nothing you are "
                "given here is evidence that an awaited event happened.\n"
            ),
        ),
        PromptModule(
            name="conservative-default",
            text=(
                "Default to keeping a condition. Retiring one the agent is "
                "still owed drops a commitment silently and unrecoverably; "
                "keeping one too long costs only another check.\n"
            ),
        ),
        PromptModule(
            name="response-contract",
            text=(
                'Respond with ONLY a JSON object {"retired": [<indices>]} — '
                "0-based indices into the numbered list, no prose, no fences. "
                "Use an empty list when every condition is still worth waiting "
                "on."
            ),
        ),
    ),
)

RETIREMENT_SYSTEM_PROMPT = RETIREMENT_PROMPT.system_prompt

# --- undeclared relevance: does a change bear on finished work? — ADR-0026 ----------------------

RELEVANCE_PROMPT = PromptManifest(
    semantic_label=PromptId.RELEVANCE,
    prompt_version="1",
    system_modules=(
        PromptModule(
            name="role",
            text=(
                "You are deciding whether something that just changed in an "
                "agent's environment means a task it ALREADY FINISHED needs "
                "following up.\n"
            ),
        ),
        PromptModule(
            name="evidence",
            text=(
                "You are given a numbered list of recently finished tasks "
                "(what each was trying to do, how it went) and a description "
                "of what just changed. Decide whether the change is a genuine "
                "follow-up to exactly one of those tasks — a reply to a "
                "message it sent, a cancellation of something it arranged, a "
                "rejection of something it submitted.\n"
            ),
        ),
        PromptModule(
            name="conservative-default",
            text=(
                "Answer NO unless the connection is specific and concrete. "
                "Most changes are unrelated background activity, and the "
                "agent's own past actions often cause changes that follow from "
                "work it already completed correctly — those are not "
                "follow-ups. A wrong YES interrupts a person with a question "
                "about work that was already done properly.\n"
            ),
        ),
        PromptModule(
            name="response-contract",
            text=(
                "Respond with ONLY a JSON object. If nothing follows up: "
                '{"relevant": false}. Otherwise:\n'
                '  {"relevant": true, "task": <index>, "goal": "<what the '
                'agent should now do about it>", "question": "<a one-sentence '
                'question asking the user whether to do it>"}\n'
                "`goal` is phrased like a task instruction. `question` is "
                "addressed to the user, states what changed and what you "
                "propose, and must be answerable with yes or no. No prose, no "
                "fences."
            ),
        ),
    ),
)

RELEVANCE_SYSTEM_PROMPT = RELEVANCE_PROMPT.system_prompt
