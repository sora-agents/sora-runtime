"""Gaia2 benchmark driver for S-ORA.

Eval tooling, not runtime: everything benchmark-specific (judge attach, scenario iteration,
scored run-to-completion) lives here, so the core `sora` CLI stays benchmark-agnostic. Run the
driver with ``python -m examples.gaia2.run_benchmark`` (see its ``--help``).
"""
