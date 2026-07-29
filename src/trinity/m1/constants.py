"""Milestone-1 competition constants."""

# Miner may submit only the head (+ optional attention). Encoder is host-side.
MAX_HEAD_PARAMS: int = 1_000_000  # hard reject if trainable head params >= this

# One M1 submission attempt per miner per calendar day (UTC window = 24h).
RATE_LIMIT_MAX_SUBMISSIONS: int = 1
RATE_LIMIT_WINDOW_DAYS: int = 1

DEFAULT_D_H: int = 1024
N_DIFFICULTY: int = 5

# Composite used by eval_milestone1 / leaderboard
DOMAIN_WEIGHT: float = 0.7

# Challenger must beat king composite by >= this margin to become new king / merge PR.
WIN_MARGIN: float = 0.02

M1_LEADERBOARD_NAME: str = "m1_leaderboard.json"
