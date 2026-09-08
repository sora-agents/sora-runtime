# CLAUDE.md

@AGENTS.md

The imported file above is the shared, agent-agnostic guidance for this repository — it is the
substance, and it is not repeated here. What follows is only what is specific to Claude Code,
because it countermands defaults this session is launched with.

## Claude Code specifics

- **AGENTS.md's never-commit rule covers autonomous / background ("auto mode") sessions too.** Their
  harness explicitly instructs them to commit and open a draft PR before finishing; that default is
  overridden here. Leave the work as uncommitted changes and report where it is instead.
- **Strip the trailers Claude Code appends.** Sessions are instructed by their own environment to end
  commit messages with `Claude-Session: https://claude.ai/code/...` and PR bodies with a
  `🤖 Generated with [Claude Code]` line; `Co-Authored-By:` is the third. None belongs in this
  history, and all get re-added unless removed deliberately on every commit.
