import csv
import os
from typing import Optional

import cv2
import numpy as np


class VideoAnnotatorCore:
    """Video IO and label persistence."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.path: Optional[str] = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame = 0
        self.annotations: dict[int, dict[str, str]] = {}
        self.frame_cache: list[np.ndarray] = []

    # -----------------------------
    # Video load/save
    # -----------------------------
    def load_video(self, filepath: str) -> bool:
        if self.cap:
            self.cap.release()

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return False

        self.cap = cap
        self.path = filepath
        self.frame_count = self._probe_frame_count(cap)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 30.0
        self.current_frame = 0
        self.frame_cache = []

        self.annotations.clear()
        self.load_csv(self.derive_csv_path())

        # Cache first frames for snappier seeking.
        prime_count = min(60, self.frame_count)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(prime_count):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_cache.append(frame.copy())
        if len(self.frame_cache) < self.frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, len(self.frame_cache))
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return True

    def _probe_frame_count(self, cap: cv2.VideoCapture) -> int:
        """
        OpenCV's CAP_PROP_FRAME_COUNT can overreport. Try to seek to the last
        frame; if that fails, rely on the position the backend landed on.
        Fall back to a lightweight grab-count when needed.
        """
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, reported - 1))
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return reported

            fallback = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if fallback > 0:
                return fallback

        count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            grabbed = cap.grab()
            if not grabbed:
                break
            count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return max(count, reported)

    def derive_csv_path(self):
        if not self.path:
            return ""
        base, _ = os.path.splitext(self.path)
        return base + "_labels.csv"

    def load_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row["frame"])
                mode_val = row.get("mode") or row.get("tag") or row.get("name")
                group_val = row.get("group") or row.get("label", "")
                if not mode_val:
                    continue
                self.annotations[frame] = {
                    "mode": mode_val,
                    "group": group_val,
                }

    def save_csv(self):
        if not self.path:
            return
        csv_path = self.derive_csv_path()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "mode", "group"])
            writer.writeheader()
            for frame in sorted(self.annotations.keys()):
                writer.writerow(
                    {
                        "frame": frame,
                        "mode": self.annotations[frame]["mode"],
                        "group": self.annotations[frame].get("group", ""),
                    }
                )

    # -----------------------------
    # Frame access
    # -----------------------------
    def read_next_frame(self):
        """Sequential read for playback."""
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.current_frame += 1
        return frame

    def _read_from_cap(self, frame_index: int):
        """Random read with fallback for flaky streams."""
        if not self.cap:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            return frame
        if frame_index > 0:
            backup_idx = max(0, frame_index - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, backup_idx)
            self.cap.read()  # discard
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def get_frame(self, frame_index: int):
        if not self.cap:
            return None
        if frame_index < 0 or frame_index >= self.frame_count:
            return None

        if frame_index < len(self.frame_cache):
            frame = self.frame_cache[frame_index]
        else:
            frame = self._read_from_cap(frame_index)

        if frame is None:
            return None

        self.current_frame = frame_index
        return frame

    # -----------------------------
    # Labels
    # -----------------------------
    def set_label(self, frame, mode, group):
        self.annotations[frame] = {"mode": mode, "group": group}
        self.save_csv()

    def get_label(self, frame=None):
        if frame is None:
            frame = self.current_frame
        return self.annotations.get(frame)
