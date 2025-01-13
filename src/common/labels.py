"""Provenance labels.

Every number that appears in a table, figure caption or sentence of this
repository carries exactly one of the three source labels below. Settings chosen
by this repository carry :data:`ASSUMPTION`.
"""

from __future__ import annotations

PAPER_REPORTED = "[Paper-reported]"
"""The value is printed in the source paper. It was not produced by this code."""

REPRODUCED = "[Reproduced]"
"""The value was produced by running this repository; a raw CSV and a config copy exist."""

TARGET_TBD = "[Target / TBD]"
"""The experiment is specified but has not been run. No value may be shown, only ``TBD``."""

ASSUMPTION = "[Assumption — not in paper]"
"""A setting chosen by this repository because the source paper does not state it."""

TBD = "TBD"
"""Cell content for a value that has not been measured."""

NOT_COMPARABLE = "N/A — baseline zero / not comparable"
"""Cell content when a Delta or Improvement % must not be computed (Before / After rule)."""

SOURCE_LABELS = (PAPER_REPORTED, REPRODUCED, TARGET_TBD)
