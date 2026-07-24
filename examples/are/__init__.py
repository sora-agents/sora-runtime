"""Example agents against ARE (Meta) (https://github.com/facebookresearch/meta-agents-research-environments).

Two adapter paths, one per subpackage — see EXAMPLES.md's "Two adapter paths" for the full story:

* ``mcp/`` — ``AreMcpWorkspaceAdapter`` over ARE's MCP server: a *static* snapshot of a scenario's
  initial app state, fitting the single-shot plan -> ground -> act loop.
* ``sim/`` — the in-process ``AreSimulation`` bridge: runs the ARE ``Environment`` event loop
  directly, so a scenario's *timeline* actually fires (mid-run injections, signal-driven replans).

Each ships an ``email_calendar`` example told from the same premise (schedule a meeting from an
email, then reply) so the two paths are directly comparable.
"""
