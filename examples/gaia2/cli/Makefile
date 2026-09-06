# Build and smoke-test the S-ORA Gaia2 container image.
#
#   make -C examples/gaia2/cli build GAIA2_CLI_DIR=/path/to/gaia2-cli
#   make -C examples/gaia2/cli verify
#
# GAIA2_CLI_DIR is the upstream gaia2-cli checkout: its base image must be built first
# (`docker build -f cli/Dockerfile -t localhost/gaia2-cli:local .` there), and its `shared/`
# directory is passed in as a second BuildKit context so neither tree has to be copied into the
# other. PLATFORM is pinned because the base image hardcodes an x86_64 libfaketime path; on an
# arm64 host this runs under emulation.

SORA_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../../..)
GAIA2_CLI_DIR ?= $(error set GAIA2_CLI_DIR to the upstream gaia2-cli checkout)
IMAGE ?= localhost/gaia2-sora:latest
PLATFORM ?= linux/amd64
DOCKER ?= docker

.PHONY: build verify

build: ## Build the S-ORA image on top of localhost/gaia2-cli:local
	$(DOCKER) build --platform $(PLATFORM) \
		--build-context gaia2cli=$(GAIA2_CLI_DIR) \
		-f $(SORA_ROOT)/examples/gaia2/cli/Dockerfile \
		-t $(IMAGE) $(SORA_ROOT)

verify: ## Smoke-test the image: symlinks, schema->manual, sora import, faketime interposition
	$(DOCKER) run --rm --platform $(PLATFORM) $(IMAGE) /usr/bin/bash -c '\
		set -e; \
		for cmd in calendar contacts emails messages chats rent-a-flat city cabs shopping cloud-drive; do \
			test -L /home/agent/bin/$$cmd || { echo "FAIL: $$cmd symlink missing"; exit 1; }; \
			echo "  OK: $$cmd"; \
		done; \
		GAIA2_STATE_DIR=/tmp/s /usr/local/bin/emails schema > /tmp/e.json; \
		GAIA2_STATE_DIR=/tmp/s /usr/local/bin/calendar schema > /tmp/c.json; \
		/usr/local/bin/python3 -c "\
import json; \
from sora.adapters.gaia2_cli import schema_to_manual; \
m = schema_to_manual(\"emails\", json.load(open(\"/tmp/e.json\")), app_class=\"EmailClientV2\"); \
print(\"  OK: emails ->\", len(m.operations), \"operations\"); \
m = schema_to_manual(\"calendar\", json.load(open(\"/tmp/c.json\")), app_class=\"CalendarApp\"); \
print(\"  OK: calendar ->\", len(m.operations), \"operations\")"; \
		/usr/local/bin/python3 -c "import sora.bootstrap; print(\"  OK: sora importable\")"; \
		echo "1999-01-02 03:04:05" > /tmp/faketime.rc; \
		/usr/local/bin/python3 -c "\
from datetime import datetime, UTC; \
from sora.adapters.gaia2_cli import _FaketimeClock; \
now = _FaketimeClock().now(); \
assert now.year == 1999, f\"the scenario clock did not read: {now}\"; \
real = datetime.now(UTC); \
assert real.year != 1999, \"the process clock must stay real, or TLS breaks\"; \
print(\"  OK: scenario clock\", now, \"| process clock stays real\")"; \
		echo "All checks passed!"'
