import os
from typing import Dict, List, Tuple
import cv2
import numpy as np


class KeyframeExtractor:
    """Extracts high-information keyframes from video files for multimodal indexing.
    
    Supports periodic sampling (e.g., 1 frame per second) and dynamic scene-change
    detection via frame difference histograms.
    """

    def __init__(self, output_dir: str = "storage/keyframes"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_keyframes(
        self,
        video_path: str,
        fps_sample: float = 1.0,
        max_frames: int = 100,
    ) -> List[Dict[str, any]]:
        """Extracts periodic representative keyframes from a video.
        
        Args:
            video_path: Absolute or relative path to the input video file.
            fps_sample: How many frames to sample per second of video.
            max_frames: Safety limit to prevent memory saturation on long videos.

        Returns:
            List of dictionaries containing frame index, timestamp, and saved file path.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / fps_sample))
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        extracted = []
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or saved_count >= max_frames:
                break

            if frame_count % frame_interval == 0:
                timestamp_sec = round(frame_count / video_fps, 2)
                filename = f"{video_name}_t{timestamp_sec:.2f}s_f{frame_count}.jpg"
                save_path = os.path.join(self.output_dir, filename)

                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                extracted.append({
                    "frame_index": frame_count,
                    "timestamp_sec": timestamp_sec,
                    "storage_path": save_path,
                    "filename": filename,
                })
                saved_count += 1

            frame_count += 1

        cap.release()
        return extracted

    @staticmethod
    def detect_scene_change(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: float = 30.0) -> bool:
        """Determines if a significant visual transition occurred between two frames."""
        gray_a = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_a, gray_b)
        mean_diff = float(np.mean(diff))
        return mean_diff > threshold
