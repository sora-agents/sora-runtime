# Workspace & Tool Integration

## Connecting to an MCP server: remote vs. local

An `mcp` (or `are-mcp`) workspace connects over whichever transport its entry describes — the runtime
does **not** have to deploy the server itself:

    workspaces:
      # Remote: connect to an already-running server (nothing is spawned). `address` is the URL;
      # SSE is the default, or add `transport: streamable-http`.
      - origin: {adapter: mcp, address: "http://localhost:8080/sse"}
        workspace_id: remote-tools

      # Local: the adapter spawns and owns a stdio subprocess. Give it a `command` (+ `args`);
      # `address` is then just a nominal label. `mcp-server-time` is the official MCP project's
      # reference time server (github.com/modelcontextprotocol/servers/tree/main/src/time).
      - origin: {adapter: mcp, address: "stdio:time"}
        workspace_id: time
        command: uvx
        args: ["mcp-server-time"]

The rule is simply: an entry with a `command` runs a local stdio subprocess; otherwise `address` is
treated as the URL of an existing server to connect to. Either way `discover()` enumerates the
server's tools and `restore()` reconnects the same way — the transport is the only thing that differs.

## See also

- [Concepts — Environment Model](../concepts/environment-model.md)
- [EXAMPLES.md — The lab workspace](https://github.com/sora-agents/sora-runtime/blob/main/EXAMPLES.md#the-lab-workspace)
