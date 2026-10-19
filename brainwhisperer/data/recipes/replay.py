"""Pins reproducing the published (seed 6-9) runs' model inputs.

These constants exist ONLY to keep loading/replaying the published
checkpoints bit-identical. They are not data about the datasets — the
truthful values differ and are derivable from the session rules.
See docs/decisions/0005-kunz-t15-calendar-replay.md: switching to the
derived values is a behavior change that rides on the planned retrain.
"""

#: Kunz T15's published embedder saw all three recording dates as calendar
#: day 0 (the pre-neuralset placeholder table mapped them to the same fake
#: session), so its month buckets are [0, 0, 0] and its sinusoidal day
#: positions are constant. The session rule derives [0, 287, 308]
#: (2024-03-03 -> 2024-12-15 -> 2025-01-05), which would size
#: `proj_*_month` to 11 slots instead of 1 and invalidate the checkpoints.
KUNZ_T15_PUBLISHED_CALENDAR_DAYS = [0, 0, 0]
