"""Grounding prompt modules and channel-specific variants."""

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

GROUND_PROMPT = PromptManifest(
    semantic_label=PromptId.GROUND,
    prompt_version="1",
    system_modules=(
        PromptModule(
            name="role",
            text=(
                "You are grounding the parameters of a SINGLE tool operation "
                "about to be invoked. You are given the goal, the operation "
                "and its parameter schema, a partial set of parameters (some "
                "values may still be references to earlier results), the "
                "agent's currently observed properties and recently observed "
                "signals, the named data-op bindings (collections an earlier "
                "step computed), and the results of the operations already "
                "executed. Produce the final, concrete parameters: fill every "
                "value that depends on a prior result, a named binding, or an "
                "already-observed property/signal from the ACTUAL data given, "
                "and keep already-concrete values as given.\n"
            ),
            variants=(
                PromptVariant(
                    channels=OPERATIONS_ONLY,
                    text=(
                        "You are grounding the parameters of a SINGLE tool operation "
                        "about to be invoked. You are given the goal, the operation "
                        "and its parameter schema, a partial set of parameters (some "
                        "values may still be references to earlier results), the named "
                        "data-op bindings (collections an earlier step computed), and "
                        "the results of the operations already executed. Produce the "
                        "final, concrete parameters: fill every value that depends on "
                        "a prior result or a named binding from the ACTUAL data given, "
                        "and keep already-concrete values as given.\n"
                    ),
                ),
                PromptVariant(
                    channels=SIGNALS_ONLY,
                    text=(
                        "You are grounding the parameters of a SINGLE tool operation "
                        "about to be invoked. You are given the goal, the operation "
                        "and its parameter schema, a partial set of parameters (some "
                        "values may still be references to earlier results), the "
                        "agent's recently observed signals, the named data-op bindings "
                        "(collections an earlier step computed), and the results of "
                        "the operations already executed. Produce the final, concrete "
                        "parameters: fill every value that depends on a prior result, "
                        "a named binding, or a recently observed signal from the "
                        "ACTUAL data given, and keep already-concrete values as given.\n"
                    ),
                ),
                PromptVariant(
                    channels=PROPERTIES_ONLY,
                    text=(
                        "You are grounding the parameters of a SINGLE tool operation "
                        "about to be invoked. You are given the goal, the operation "
                        "and its parameter schema, a partial set of parameters (some "
                        "values may still be references to earlier results), the "
                        "agent's currently observed properties, the named data-op "
                        "bindings (collections an earlier step computed), and the "
                        "results of the operations already executed. Produce the "
                        "final, concrete parameters: fill every value that depends on "
                        "a prior result, a named binding, or an already-observed "
                        "property from the ACTUAL data given, and keep "
                        "already-concrete values as given.\n"
                    ),
                ),
            ),
        ),
        PromptModule(
            name="response-contract",
            text=(
                'Respond with ONLY a JSON object of the form {"params": { ... '
                "}} and nothing else — no prose, no markdown fences. Use only "
                "parameter names from the schema.\n"
            ),
        ),
        PromptModule(
            name="missing-data",
            text=(
                "If a reference names data that is NOT in what you were given "
                "— the operation it names returned an empty list, the field is "
                "absent, the binding is empty — then that value does not exist "
                "yet and you must NOT invent one. Do NOT substitute a nearby "
                "value that merely looks plausible: the user's own name or "
                "address, some other contact, a guessed date. The agent ACTS "
                "on what you return and many operations are irreversible, so a "
                "wrong-but-plausible recipient is far worse than an admitted "
                "gap — a gap can be recovered from, a sent email cannot. In "
                'that case respond with ONLY {"unresolvable": "<which '
                'parameter, and what was missing from the data>"} instead, and '
                "the runtime re-plans from it.\n"
            ),
        ),
        PromptModule(
            name="list-integrity",
            text=(
                "This applies element by element inside a LIST parameter too: "
                "if an element's reference names data that is not there, "
                "report the gap for that parameter — do NOT quietly drop the "
                "element and return a shorter list. A list that comes back "
                "shorter than it went in is silently doing less than the step "
                "asked for, which reads as success and is not; the runtime "
                "rejects it.\n"
            ),
        ),
        PromptModule(
            name="resolvable-values",
            text=(
                "That is only for missing DATA. A value you can compute or "
                "phrase from what you WERE given is resolvable, so produce it: "
                "a $decide asking for a sentence about a result that is "
                "present, or a date derived from a clock reading in the "
                "history, are ordinary work, not gaps.\n"
            ),
        ),
        PromptModule(
            name="execution-record",
            text=(
                "Some parameters are natural-language TEXT you are asked to "
                "phrase — a report back to the user, an email body. Phrase "
                "those from the EXECUTION RECORD: the results of operations "
                "already executed, plus the note of planned steps that "
                "performed no operation. The goal tells you what was INTENDED, "
                "never what happened. Do NOT state that an action was "
                "performed unless the record shows the operation that "
                "performed it — a planned step can legitimately expand to "
                "nothing when the collection it iterates turns out empty, and "
                "then the thing it would have done was not done by anyone. "
                "Where the record shows such a step, say so plainly and say "
                "which work it was, rather than omitting it or reporting the "
                "intent as achieved. The user has no other view of what the "
                "agent did, so a report that overstates it is not something "
                "they can catch or recover from."
            ),
        ),
    ),
)

GROUND_SYSTEM_PROMPT = GROUND_PROMPT.system_prompt
