from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CATEGORIES = ("C1", "C2", "C3")
CATEGORY_PAIR_COMBOS = (("C1", "C2"), ("C1", "C3"), ("C2", "C3"))


@dataclass
class HarmInterval:
    start_sec: float
    end_sec: float
    category: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateClip:
    category: str
    video_path: str
    duration_sec: float
    benchmark_name: str
    subcategories: list[str] = field(default_factory=list)
    source_labels: list[int] = field(default_factory=list)
    source_description: str | None = None
    violate_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InsertedSegment:
    category: str
    source_path: str
    source_benchmark: str
    source_subcategories: list[str]
    source_duration_sec: float
    base_insert_position_sec: float
    interval: HarmInterval

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["interval"] = self.interval.to_dict()
        return payload


@dataclass
class SyntheticSample:
    sample_id: str
    insert_count: int
    planned_categories: list[str]
    synthetic_video_path: str
    safe_source_path: str
    safe_source_duration_sec: float
    safe_clip_start_sec: float
    base_safe_clip_duration_sec: float
    clip_duration_sec: float
    max_duration_sec: float
    was_tail_trimmed: bool
    inserted_segments: list[InsertedSegment] = field(default_factory=list)
    gt_categories: list[str] = field(default_factory=list)
    gt_intervals: list[HarmInterval] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["inserted_segments"] = [segment.to_dict() for segment in self.inserted_segments]
        payload["gt_intervals"] = [interval.to_dict() for interval in self.gt_intervals]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SyntheticSample":
        inserted_segments = [
            InsertedSegment(
                category=segment["category"],
                source_path=segment["source_path"],
                source_benchmark=segment["source_benchmark"],
                source_subcategories=list(segment.get("source_subcategories", [])),
                source_duration_sec=float(segment["source_duration_sec"]),
                base_insert_position_sec=float(segment["base_insert_position_sec"]),
                interval=HarmInterval(**segment["interval"]),
            )
            for segment in payload.get("inserted_segments", [])
        ]
        gt_intervals = [HarmInterval(**interval) for interval in payload.get("gt_intervals", [])]
        return cls(
            sample_id=payload["sample_id"],
            insert_count=int(payload["insert_count"]),
            planned_categories=list(payload.get("planned_categories", [])),
            synthetic_video_path=payload["synthetic_video_path"],
            safe_source_path=payload["safe_source_path"],
            safe_source_duration_sec=float(payload["safe_source_duration_sec"]),
            safe_clip_start_sec=float(payload["safe_clip_start_sec"]),
            base_safe_clip_duration_sec=float(payload["base_safe_clip_duration_sec"]),
            clip_duration_sec=float(payload["clip_duration_sec"]),
            max_duration_sec=float(payload["max_duration_sec"]),
            was_tail_trimmed=bool(payload["was_tail_trimmed"]),
            inserted_segments=inserted_segments,
            gt_categories=list(payload.get("gt_categories", [])),
            gt_intervals=gt_intervals,
        )
