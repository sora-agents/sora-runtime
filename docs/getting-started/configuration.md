# Configuration

## Technology Stack & Requirements

- **Runtime — Python 3.12+ (asyncio)**: the decision cycle, activities, memory modules, and action
  registry live here. Chosen for async I/O concurrency without a blocking scheduler, and because the
  LLM/tool ecosystem (MCP, A2A, provider SDKs) is Python-first.
- **CLI**: the runtime ships a minimal terminal interface — output is streamed as the decision cycle
  runs, and the user can type input at any point, which is queued as a `Message` (sender `"user"`) for
  the next Observe phase — terminal input is user communication, not environment stimuli, so it's never
  a `Percept`. Similar in spirit to existing coding-agent CLIs: a persistent terminal session, not a
  one-shot command. Richer UIs are out of scope for the runtime and belong to whatever agent consumes it.
- **LLM access**: the runtime's one seam onto a model is the wire-format-neutral `LLMClient`
  Protocol (`sora/llm.py`) — system + prompt in, text out, committing to no provider shape — so no
  hard dependency on a single provider SDK. The shipped default client targets the Anthropic
  Messages API via the optional `[llm]` extra (model id is a config value, never hardcoded; the
  provider API key is supplied through the environment — e.g. `ANTHROPIC_API_KEY` — never committed
  to `agent.yaml`, see [Configuring the LLM](#configuring-the-llm-and-its-api-key)). Provider SDKs
  are optional extras. The model call itself lives in `ProceduralMemory.infer`, behind the default
  `ReasonStrategy`; the text→`Plan` conversion there is the anti-corruption boundary.
- **Protocol adapters** (MCP, A2A, OpenAPI) ship as optional extras (e.g. `pip install sora-runtime[mcp]`)
  so the core package stays dependency-light.
- **Manual parsing**: pluggable `ManualParser` per format — Markdown by default, XML as an alternative.
- **Memory backends**: pluggable `MemoryBackend` — file-based by default, database/vector-store as
  drop-in alternatives.
- **Tooling**: `pytest` + `mypy` for testing and type-checking; `uv` for packaging.
- **Zero manual wiring for the common case**: implementing an agent means writing `agent.yaml` and,
  typically, one `ReasonStrategy` — never constructing `Agent`/`DecisionCycle`/memory modules by hand.
  All wiring is centralized in `sora/bootstrap.py` (see the
  [Python API Reference](../reference/python-api.md#sora.bootstrap)).

## Configuring the LLM and its API key

The default `ReasonStrategy` is **model-backed** — Reason is the one phase with no mechanical default,
since planning needs a model — so running an agent with it requires an LLM. Install the provider extra
and supply credentials through the **environment**, never through `agent.yaml`:

    $ uv sync --extra llm                    # the default AnthropicLLMClient (official Anthropic SDK)
    $ export ANTHROPIC_API_KEY=sk-ant-...     # the secret lives in the environment, not in any file
    $ uv run sora run

The API key is a **secret — keep it out of version control.** `agent.yaml` is committed, so it names
the *model* (a config value you can swap freely, e.g. `claude-opus-4-8`) but never the key. The
shipped `AnthropicLLMClient` reads the key from the environment via the Anthropic SDK's standard
resolution — `ANTHROPIC_API_KEY`, or an `ant auth login` profile for local dev — so the client needs
no key in code or config. Only pass one explicitly (`AnthropicLLMClient(api_key=...)`) when you must
inject a specific key programmatically. In production, load the key from a secrets manager or a
`.gitignore`d `.env` at start-up; never paste keys into `agent.yaml`, manuals, prompts, or source.

For local development, **copy `.env.example` to `.env`** and set `ANTHROPIC_API_KEY` there — `sora run`
loads a local `.env` automatically when present, so you don't need to `export` it each time. `.env`
is gitignored, and **real environment variables still take precedence** (a `.env` value is used only
when the variable isn't already set), so it never silently overrides a key you exported deliberately.

For OpenAI proper, `OpenAICompatLLMClient` likewise uses the SDK-standard `OPENAI_API_KEY` when
`base_url` is absent. For a hosted OpenAI-compatible endpoint, name a dedicated environment
variable explicitly; the variable name is safe to commit, while bootstrap resolves its value only
when constructing the client:

```yaml
agent:
  llm:
    client: sora.adapters.openai_llm.OpenAICompatLLMClient
    model: moonshotai/kimi-k2.5
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
```

```console
$ export OPENROUTER_API_KEY=sk-or-v1-...
```

`api_key_env` and an explicit `api_key` are mutually exclusive, and a named but unset variable is a
startup error. A configured `base_url` never inherits `OPENAI_API_KEY`: without `api_key_env` (or a
programmatically supplied `api_key`) it receives a non-secret placeholder, which is suitable for
unauthenticated local runtimes such as Ollama.

## See also

- [Quickstart](quickstart.md) — installing and running your first agent
- [Your First Agent](first-agent.md) — `agent.yaml` anatomy beyond LLM configuration
