# Tool Metadata
category: Lab / Perception
id: video-stream

# Functional Description
A ceiling-mounted camera over the workbench that performs on-device scene understanding and
publishes a symbolic description of what's currently in view — not raw video.

# Observable Properties
- scene (string): natural-language description of the objects currently visible on the workbench,
  updated whenever the scene changes.

# Signals
(none)

# Operations
(none — this tool is observation-only)

# Usage Protocols & Safety
Focus on this tool to keep `scene` current in working memory. No operations to invoke.
