from __future__ import annotations


def smooth_main_view_segments(
    segments: list[dict],
    min_main_duration: float = 3.0,
    max_gap: float = 2.0,
) -> list[dict]:
    main_segments = [
        dict(segment)
        for segment in segments
        if segment.get("label") == "MAIN_LIVE_VIEW"
        and float(segment.get("end", 0)) - float(segment.get("start", 0)) >= min_main_duration
    ]
    merged: list[dict] = []
    for segment in main_segments:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        gap = float(segment["start"]) - float(previous["end"])
        if gap <= max_gap:
            previous["end"] = segment["end"]
            previous["confidence"] = max(
                float(previous.get("confidence", 0)),
                float(segment.get("confidence", 0)),
            )
        else:
            merged.append(segment)
    return merged

