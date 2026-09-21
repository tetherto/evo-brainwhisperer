from datetime import datetime

# Helper constants holding the number of recording days per dataset

CARD_SESSIONS = [
    "t15.2023.08.11",
    "t15.2023.08.13",
    "t15.2023.08.18",
    "t15.2023.08.20",
    "t15.2023.08.25",
    "t15.2023.08.27",
    "t15.2023.09.01",
    "t15.2023.09.03",
    "t15.2023.09.24",
    "t15.2023.09.29",
    "t15.2023.10.01",
    "t15.2023.10.06",
    "t15.2023.10.08",
    "t15.2023.10.13",
    "t15.2023.10.15",
    "t15.2023.10.20",
    "t15.2023.10.22",
    "t15.2023.11.03",
    "t15.2023.11.04",
    "t15.2023.11.17",
    "t15.2023.11.19",
    "t15.2023.11.26",
    "t15.2023.12.03",
    "t15.2023.12.08",
    "t15.2023.12.10",
    "t15.2023.12.17",
    "t15.2023.12.29",
    "t15.2024.02.25",
    "t15.2024.03.03",
    "t15.2024.03.08",
    "t15.2024.03.15",
    "t15.2024.03.17",
    "t15.2024.04.25",
    "t15.2024.04.28",
    "t15.2024.05.10",
    "t15.2024.06.14",
    "t15.2024.07.19",
    "t15.2024.07.21",
    "t15.2024.07.28",
    "t15.2025.01.10",
    "t15.2025.01.12",
    "t15.2025.03.14",
    "t15.2025.03.16",
    "t15.2025.03.30",
    "t15.2025.04.13",
]

WILLETT_SESSIONS = [
    "t12.2022.04.28",
    "t12.2022.05.05",
    "t12.2022.05.17",
    "t12.2022.05.19",
    "t12.2022.05.24",
    "t12.2022.05.26",
    "t12.2022.06.02",
    "t12.2022.06.07",
    "t12.2022.06.14",
    "t12.2022.06.16",
    "t12.2022.06.21",
    "t12.2022.06.23",
    "t12.2022.06.28",
    "t12.2022.07.05",
    "t12.2022.07.14",
    "t12.2022.07.21",
    "t12.2022.07.27",
    "t12.2022.07.29",
    "t12.2022.08.02",
    "t12.2022.08.11",
    "t12.2022.08.13",
    "t12.2022.08.18",
    "t12.2022.08.23",
    "t12.2022.08.25",
]

# Placeholder only.
KUNZ_T12_SESSIONS = [
    "t12.2023.11.28",
]

KUNZ_T15_SESSIONS = [
    "t15.2024.03.03",
    "t15.2024.03.03",
    "t15.2024.03.03",
]

# Placeholder only. These 10 arbitrary-but-parseable dates exist only to satisfy
# sessions_to_calendar_days's structural requirement (every dataset needs an entry).
WAIRAGKAR_SESSIONS = [
    "t15.2024.01.01",
    "t15.2024.01.02",
    "t15.2024.01.03",
    "t15.2024.01.04",
    "t15.2024.01.05",
    "t15.2024.01.06",
    "t15.2024.01.07",
    "t15.2024.01.08",
    "t15.2024.01.09",
    "t15.2024.01.10",
]

JUDE_SESSIONS = [
    "t17.2024.04.15",
    "t17.2024.04.16",
    "t17.2024.04.17",
    "t17.2024.05.06",
    "t17.2024.05.07",
]


def session_to_day(sessions: list[str]) -> dict[str, int]:
    return {s: i for i, s in enumerate(sessions)}


def sessions_to_calendar_days(sessions: list[str]) -> list[int]:
    """Convert session name strings to days elapsed since the first session."""

    def _parse(s: str) -> datetime:
        return datetime.strptime(s[-10:], "%Y.%m.%d")

    dates = [_parse(s) for s in sessions]
    first = min(dates)
    return [(d - first).days for d in dates]


SESSIONS_BY_DATASET: dict[str, list[str]] = {
    "card": CARD_SESSIONS,
    "willet": WILLETT_SESSIONS,
    "kunz_t12": KUNZ_T12_SESSIONS,
    "kunz_t15": KUNZ_T15_SESSIONS,
    "wairagkar": WAIRAGKAR_SESSIONS,
    "jude": JUDE_SESSIONS,
}
