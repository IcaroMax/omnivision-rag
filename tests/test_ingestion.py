import os
import cv2
import numpy as np
import pytest
from src.ingestion.video_processor import KeyframeExtractor


def test_keyframe_extractor_missing_file():
    extractor = KeyframeExtractor(output_dir="storage/test_keyframes")
    with pytest.raises(FileNotFoundError):
        extractor.extract_keyframes("nonexistent_video.mp4")


def test_keyframe_extractor_synthetic_video(tmp_path):
    output_dir = tmp_path / "keyframes"
    extractor = KeyframeExtractor(output_dir=str(output_dir))

    # Create dummy 1-second video (10 frames at 10 fps)
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))

    for i in range(10):
        frame = np.full((240, 320, 3), i * 20, dtype=np.uint8)
        out.write(frame)
    out.release()

    extracted = extractor.extract_keyframes(str(video_path), fps_sample=1.0)
    assert len(extracted) >= 1
    assert os.path.exists(extracted[0]["storage_path"])
    assert "timestamp_sec" in extracted[0]


def test_detect_scene_change():
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = np.full((100, 100, 3), 255, dtype=np.uint8)

    # Extreme visual difference must trigger scene change
    assert KeyframeExtractor.detect_scene_change(frame_a, frame_b, threshold=30.0) is True
    # Identical frames must not trigger scene change
    assert KeyframeExtractor.detect_scene_change(frame_a, frame_a, threshold=30.0) is False
