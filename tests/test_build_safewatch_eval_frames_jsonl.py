import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "data_curation"))

from build_safewatch_eval_frames_jsonl import (  # pyright: ignore[reportMissingImports]
    build_image_eval_record,
    frame_directory_name,
    sample_times_1fps,
)


def test_sample_times_1fps_handles_short_and_fractional_durations():
    assert sample_times_1fps(0.0) == [0.0]
    assert sample_times_1fps(0.5) == [0.0]
    assert sample_times_1fps(2.2) == [0.0, 1.0, 2.0]


def test_build_image_eval_record_converts_video_prompt_to_image_prompt():
    record = {
        "id": "sample_1",
        "video": "/tmp/video.mp4",
        "conversations": [
            {"from": "human", "value": "<video>\nDescribe this video."},
            {"from": "gpt", "value": "[]"},
        ],
        "metadata": {"source_category": "abuse_1"},
    }
    frame_paths = ["/tmp/sample_1/frame_0001.jpg", "/tmp/sample_1/frame_0002.jpg"]

    output = build_image_eval_record(record, frame_paths, "/tmp/sample_1")

    assert output["images"] == frame_paths
    assert output["conversations"][0]["value"] == "<image> <image>\nDescribe this video."
    assert output["conversations"][1]["value"] == "[]"
    assert output["metadata"]["frame_dir"] == "/tmp/sample_1"
    assert output["metadata"]["num_frames"] == 2
    assert output["metadata"]["source_video"] == "/tmp/video.mp4"


def test_frame_directory_name_shortens_and_hashes_long_ids():
    sample_id = "misinformation_2__নিউজিল্যান্ডের আগ্নেয়গিরি গোলাপি অগ্ন্যুৎপাত Pink Volcano 🌋"

    name = frame_directory_name(sample_id)

    assert len(name) < len(sample_id)
    assert name.endswith("_" + name.split("_")[-1])
    assert len(name.split("_")[-1]) == 12
