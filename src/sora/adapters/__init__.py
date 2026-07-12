"""Protocol adapters that import externally-defined tools into the S-ORA usage interface.

Each adapter lives behind an optional extra (e.g. ``sora-runtime[mcp]``) and is imported only when
that protocol is actually used, so the core package stays dependency-light.
"""
