"""Closed M1 domain taxonomies (torch-free; shared by pack + triage_head)."""
from __future__ import annotations

DOMAINS_5: tuple[str, ...] = (
    "math",
    "code",
    "knowledge",
    "commonsense",
    "instruction",
)

DOMAINS_20: tuple[str, ...] = (
    "agriculture",
    "archaeology",
    "architecture",
    "astronomy",
    "automotive",
    "aviation",
    "chemistry",
    "cooking",
    "energy",
    "finance",
    "gardening",
    "hardware",
    "marine",
    "music",
    "photography",
    "physiology",
    "sports",
    "textiles",
    "weather",
    "wildlife",
)

N_DIFFICULTY: int = 5
