"""Runtime-IO adapter: surfaces the agent's own I/O channels as S-ORA tools, so *all* messaging is
a uniform tool invocation rather than a special ``send`` action.

Today it exposes exactly one channel — the reply to the user — as a ``send_message_to_user``
operation. The tool is transport-backed: it delegates to the injected ``MessageTransport``, whatever
bootstrap wired (``InProcessTransport`` for a bare runtime, ``AreTransport`` under ARE). So one
always-joined tool is environment-correct everywhere, and there is nothing to suppress: an
environment framework like ARE already routes the user channel through *its* transport, not a
competing tool (ARE deliberately keeps ``AgentUserInterface`` out of its tool catalog). Additional
runtime channels (e.g. an agent-to-agent ``send_message_to_agent``) are foreseen tools in this same
workspace, deferred until peer transport is wired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sora.manual import Manual, OperationSpecification
from sora.types import OperationAck

if TYPE_CHECKING:
    from sora.environment import Tool, Workspace, WorkspaceOrigin
    from sora.manual import ToolRecord, WorkspaceRecord
    from sora.perception import SignalSink
    from sora.transport import MessageTransport
    from sora.types import ObservableProperty

RUNTIME_IO_ADAPTER = "runtime-io"  # matches WorkspaceOrigin.adapter
RUNTIME_IO_ADDRESS = "runtime"  # the origin address; also the tool-id namespace
RUNTIME_IO_WORKSPACE = "runtime-io"
SEND_MESSAGE_TO_USER = "send_message_to_user"

_USER_CHANNEL_MANUAL = "RuntimeUserChannel"
_USER_CHANNEL_TOOL = f"{RUNTIME_IO_ADDRESS}/UserChannel"


def user_channel_manual() -> Manual:
    """The synthesized manual for the user-reply tool. Recipient-in-the-name (``..._to_user``) is
    deliberate: the operation asserts *who* it reaches, so a plan can't mistake it for a way to
    message some other person (emailing/chatting a third party is that domain tool's own op)."""
    return Manual(
        id=_USER_CHANNEL_MANUAL,
        metadata={"source": RUNTIME_IO_ADAPTER},
        description="The agent's own reply channel to the user.",
        observable_properties=[],
        signals=[],
        operations=[
            OperationSpecification(
                name=SEND_MESSAGE_TO_USER,
                description=(
                    "Send a natural-language message to the user — the agent's own reply channel. "
                    "Use it to report an outcome or answer; the recipient is always the user, so "
                    "never use it to message anyone else (emailing/chatting another person is that "
                    "domain tool's own operation)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The natural-language message to show the user.",
                        }
                    },
                    "required": ["text"],
                },
                returns=None,
                side_effecting=True,  # sending a message to the user is an outward-facing write
            )
        ],
        raw_text=None,
    )


class _RuntimeIOTool:  # satisfies the Tool Protocol
    def __init__(self, tool_id: str, manual: Manual, transport: MessageTransport) -> None:
        self.id = tool_id
        self.manual = manual
        self.address: str | None = None
        self._transport = transport

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        if operation_name != SEND_MESSAGE_TO_USER:
            return OperationAck(ok=False, result=f"unknown operation {operation_name!r}")
        # Deliver on the agent's own channel. Keep the {"text": ...} content shape the transport and
        # presentation layer already stream; pass any extra params through untouched.
        await self._transport.send("user", dict(params))
        return OperationAck(ok=True)

    async def focus(self, sink: SignalSink) -> None:  # no observable state to subscribe to
        return None

    async def unfocus(self) -> None:
        return None

    def observe(self) -> list[ObservableProperty]:
        return []


class _RuntimeIOWorkspace:  # satisfies the Workspace Protocol
    def __init__(self, ws_id: str, origin: WorkspaceOrigin, tools: list[Tool]) -> None:
        self.id = ws_id
        self.origin = origin
        self._tools = tools

    def tools(self) -> list[Tool]:
        return self._tools

    async def close(self) -> None:  # the transport lifecycle is owned by bootstrap, not the tool
        return None


class RuntimeIOAdapter:  # satisfies the WorkspaceAdapter Protocol
    """Always joined by bootstrap, wired with the same ``MessageTransport`` handed to the cycle.
    Its one workspace holds the runtime's own I/O tools (today just the user-reply channel)."""

    name = RUNTIME_IO_ADAPTER  # matches WorkspaceOrigin.adapter

    def __init__(
        self,
        *,
        origin: WorkspaceOrigin,
        transport: MessageTransport,
        workspace_id: str = RUNTIME_IO_WORKSPACE,
    ) -> None:
        self._origin = origin
        self._transport = transport
        self._workspace_id = workspace_id

    async def discover(self) -> list[Workspace]:
        return [self._build_workspace(self._workspace_id, self._origin)]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        # One synthetic tool, rebuilt directly — there is no external connection to re-establish.
        return self._build_workspace(workspace_record.id, workspace_record.origin)

    def _build_workspace(self, ws_id: str, origin: WorkspaceOrigin) -> Workspace:
        tool = _RuntimeIOTool(_USER_CHANNEL_TOOL, user_channel_manual(), self._transport)
        return _RuntimeIOWorkspace(ws_id, origin, [tool])
