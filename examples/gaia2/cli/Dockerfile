# S-ORA on the containerised Gaia2 CLI harness.
#
# Built with two BuildKit contexts so neither checkout has to be copied into the other:
#   docker build --platform linux/amd64 \
#     --build-context gaia2cli=<path to gaia2-cli> \
#     -f examples/gaia2/cli/Dockerfile -t localhost/gaia2-sora:latest .
# The default context is this repository; `gaia2cli` supplies the shared C wrapper and adapter
# base. See the Makefile next to this file.
#
# The image name matters: the runner picks a container profile by substring match, and
# `openclaw`/`gaia2-oc`/`hermes`/`gaia2-mini`/`mini-swe-agent`/`oracle` each select a profile that
# would hand this image the wrong environment. `gaia2-sora` matches none of them and falls through
# to the default profile, which passes PROVIDER/MODEL/API_KEY/BASE_URL — what the worker reads.

ARG GAIA2_CLI_IMAGE=localhost/gaia2-cli
ARG GAIA2_CLI_VERSION=local

# ── Compile the setuid wrapper from upstream source ──────────────────────────────────────────────
FROM debian:bookworm-slim AS gaia2-exec-builder
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=gaia2cli shared/gaia2-exec.c /tmp/gaia2-exec.c
RUN gcc -O2 -o /tmp/gaia2-exec /tmp/gaia2-exec.c && strip /tmp/gaia2-exec

# ── Final image: the gaia2-cli base + the S-ORA runtime ──────────────────────────────────────────
FROM ${GAIA2_CLI_IMAGE}:${GAIA2_CLI_VERSION}

ARG http_proxy
ARG https_proxy

# Only the packaging inputs, so a change to tests or docs does not invalidate the install layer.
COPY pyproject.toml README.md /opt/sora-runtime/
COPY src /opt/sora-runtime/src
RUN http_proxy=${http_proxy} https_proxy=${https_proxy} \
    pip install --no-cache-dir "/opt/sora-runtime[llm,openai]"

# ── Gaia2 state: privilege-separated via the "gaia2" user ────────────────────────────────────────
RUN useradd -r -s /usr/sbin/nologin -d /var/gaia2 gaia2 && \
    mkdir -p /var/gaia2/state
ENV GAIA2_STATE_DIR=/var/gaia2/state
RUN chown -R gaia2:gaia2 /var/gaia2 && chmod 700 /var/gaia2

# ── Non-root agent user ──────────────────────────────────────────────────────────────────────────
RUN useradd -m -s /bin/bash -u 1000 agent

# ── Setuid Gaia2 wrapper ─────────────────────────────────────────────────────────────────────────
COPY --from=gaia2-exec-builder /tmp/gaia2-exec /usr/local/bin/gaia2-exec
RUN chown gaia2:gaia2 /usr/local/bin/gaia2-exec && chmod u+s /usr/local/bin/gaia2-exec

# ── The agent's sandbox PATH: the ten app symlinks and nothing else ──────────────────────────────
# No shell utilities: operations are typed and invoked as argv, so there is no shell for the model
# to write into. The init entrypoint deletes the symlinks this scenario does not use, and the
# adapter discovers exactly what survives.
RUN mkdir -p /home/agent/bin && \
    for cmd in calendar contacts emails messages chats rent-a-flat city cabs shopping cloud-drive; do \
        ln -s /usr/local/bin/gaia2-exec /home/agent/bin/$cmd; \
    done && \
    ln -s /usr/local/bin/python3 /home/agent/bin/python3 && \
    chown -R agent:agent /home/agent/bin

# ── Adapter + worker + entrypoints ───────────────────────────────────────────────────────────────
COPY --from=gaia2cli shared/gaia2_adapter_base.py /opt/gaia2_adapter_base.py
COPY examples/gaia2/cli/gaia2_adapter.py /opt/gaia2_adapter.py
COPY examples/gaia2/cli/sora_worker.py /opt/sora_worker.py
COPY examples/gaia2/cli/agent.yaml /opt/agent.yaml
COPY examples/gaia2/cli/entrypoint.sh /opt/entrypoint.sh
COPY examples/gaia2/cli/gaia2-init-entrypoint.sh /opt/gaia2-init-entrypoint.sh
RUN chmod +x /opt/entrypoint.sh /opt/gaia2-init-entrypoint.sh

ENV HOME=/home/agent
ENV PATH="/home/agent/bin"

# The launcher overrides the image entrypoint with `bash /opt/gaia2-init-entrypoint.sh`, so this
# path is fixed by the harness, not by us.
USER root
ENTRYPOINT []
CMD ["/usr/bin/bash", "/opt/gaia2-init-entrypoint.sh"]
