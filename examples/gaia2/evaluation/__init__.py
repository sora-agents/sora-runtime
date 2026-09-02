"""Offline-first evaluation infrastructure for the Gaia2 example.

Campaign modules own prompt- or paper-specific policy. The package intentionally has no provider
imports at module import time. Offline commands remain usable without a credential or network
connection.
"""
