import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from moviepy import VideoFileClip


@dataclass
class VideoSection:
    section_id: int
    label: str
    group_id: str
    start_frame: int
    end_frame: int

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame

    def to_json(self) -> dict:
        return {
            "section_ID": self.section_id,
            "section_label": self.label,
            "group_ID": self.group_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


class VideoAnnotatorCore:
    """Video IO and label persistence."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.path: Optional[str] = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame = 0
        self.annotations: dict[int, dict[str, str]] = {}
        self.sections: list[VideoSection] = []
        self.frame_cache: list[np.ndarray] = []
        self.dirty = False
        self.sections_dirty = False

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
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 30.0
        self.current_frame = 0
        self.frame_cache = []
        self.frame_count = self._read_frame_count(cap)

        self.annotations.clear()
        self.sections.clear()
        self.load_csv(self.derive_csv_path())
        self.load_sections(self.derive_sections_path())
        self.dirty = False
        self.sections_dirty = False

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

    def _read_frame_count(self, cap: cv2.VideoCapture) -> int:
        frame_count = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        if frame_count > 0:
            return frame_count

        count = 0
        while True:
            ret, _frame = cap.read()
            if not ret:
                break
            count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return count

    def derive_csv_path(self):
        if not self.path:
            return ""
        base, _ = os.path.splitext(self.path)
        return base + "_labels.csv"

    def derive_sections_path(self):
        if not self.path:
            return ""
        base, _ = os.path.splitext(self.path)
        return base + "_sections.json"

    def load_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row["frame"])
                mode_val = row.get("mode") or row.get("tag") or row.get("name")
                group_val = row.get("group_ID") or row.get("group") or row.get("label", "")
                section_id = self._parse_int(row.get("section_ID"))
                if not mode_val:
                    continue
                self.annotations[frame] = {
                    "mode": mode_val,
                    "group": group_val,
                    "group_ID": group_val,
                    "section_ID": section_id,
                    "section_label": row.get("section_label", ""),
                }

    def save_csv(self):
        if not self.path:
            return
        self.sync_annotation_sections()
        csv_path = self.derive_csv_path()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame", "section_ID", "section_label", "group_ID", "mode"],
            )
            writer.writeheader()
            for frame in sorted(self.annotations.keys()):
                ann = self.annotations[frame]
                section = self._annotation_section(frame, ann)
                section_id = section.section_id if section else ann.get("section_ID")
                section_label = section.label if section else ann.get("section_label", "")
                group_id = ann.get("group_ID") or ann.get("group", "")
                if section and section.group_id:
                    group_id = section.group_id
                writer.writerow(
                    {
                        "frame": frame,
                        "section_ID": "" if section_id in ("", None) else section_id,
                        "section_label": section_label,
                        "group_ID": group_id,
                        "mode": ann["mode"],
                    }
                )
        self.dirty = False

    def load_sections(self, json_path: str):
        if not json_path or not os.path.exists(json_path):
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        items = payload.get("sections", []) if isinstance(payload, dict) else []
        for item in items:
            section_id = self._parse_int(item.get("section_ID"))
            if section_id is None:
                continue
            start_frame = self._clamp_frame(self._parse_int(item.get("start_frame")) or 0)
            end_frame = self._clamp_frame(self._parse_int(item.get("end_frame")) or start_frame)
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame
            self.sections.append(
                VideoSection(
                    section_id=section_id,
                    label=str(item.get("section_label", "")),
                    group_id=str(item.get("group_ID", "")),
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
        self._renumber_sections_by_start()

    def save_sections(self):
        if not self.path:
            return
        payload = {"sections": [section.to_json() for section in self.sections]}
        with open(self.derive_sections_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        self.sections_dirty = False

    def _parse_int(self, value):
        if value in ("", None):
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _clamp_frame(self, frame: int) -> int:
        if self.frame_count <= 0:
            return max(0, frame)
        return max(0, min(self.frame_count - 1, int(frame)))

    def _is_auto_section_label(self, label: str) -> bool:
        clean = str(label or "").strip()
        return not clean or re.fullmatch(r"Section\s+\d+", clean) is not None

    def _annotation_section(self, frame: int, ann: dict) -> VideoSection | None:
        section = self.section_by_id(ann.get("section_ID"))
        if section is not None:
            return section
        return self.section_at_frame(frame)

    def sync_annotation_sections(self) -> bool:
        changed = False
        for frame, ann in self.annotations.items():
            section = self._annotation_section(frame, ann)
            if section is None:
                continue

            if self._parse_int(ann.get("section_ID")) != section.section_id:
                ann["section_ID"] = section.section_id
                changed = True
            if ann.get("section_label", "") != section.label:
                ann["section_label"] = section.label
                changed = True

            group_id = section.group_id or ann.get("group_ID") or ann.get("group", "")
            if group_id and ann.get("group_ID", "") != group_id:
                ann["group_ID"] = group_id
                changed = True
            if group_id and ann.get("group", "") != group_id:
                ann["group"] = group_id
                changed = True

        if changed:
            self.dirty = True
        return changed

    def _renumber_sections_by_start(self):
        old_to_new: dict[int, int] = {}
        self.sections.sort(
            key=lambda item: (item.start_frame, item.end_frame, item.section_id)
        )

        for new_id, section in enumerate(self.sections, start=1):
            old_id = section.section_id
            old_to_new[old_id] = new_id
            section.section_id = new_id
            if self._is_auto_section_label(section.label):
                section.label = f"Section {new_id}"

        if not old_to_new:
            return

        changed_annotations = False
        for ann in self.annotations.values():
            old_id = self._parse_int(ann.get("section_ID"))
            if old_id not in old_to_new:
                continue
            new_id = old_to_new[old_id]
            if old_id != new_id:
                ann["section_ID"] = new_id
                changed_annotations = True
            if self._is_auto_section_label(str(ann.get("section_label", ""))):
                ann["section_label"] = f"Section {new_id}"
                changed_annotations = True

        if changed_annotations:
            self.dirty = True

    def next_section_id(self) -> int:
        if not self.sections:
            return 1
        return max(section.section_id for section in self.sections) + 1

    def add_section(self, start_frame: int, end_frame: int, group_id: str = "", label: str = "") -> VideoSection:
        start_frame = self._clamp_frame(start_frame)
        end_frame = self._clamp_frame(end_frame)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        section = VideoSection(
            section_id=self.next_section_id(),
            label=label,
            group_id=group_id,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self.sections.append(section)
        self._renumber_sections_by_start()
        self.sync_annotation_sections()
        self.sections_dirty = True
        self.save_sections()
        return section

    def replace_sections(self, section_specs: list[dict], reassign_annotations: bool = True):
        normalized = []
        for index, spec in enumerate(section_specs):
            start_frame = self._parse_int(spec.get("start_frame"))
            end_frame = self._parse_int(spec.get("end_frame"))
            if start_frame is None or end_frame is None:
                continue
            start_frame = self._clamp_frame(start_frame)
            end_frame = self._clamp_frame(end_frame)
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame
            normalized.append(
                {
                    "sort_index": index,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "label": str(spec.get("label", "")).strip(),
                    "group_id": str(spec.get("group_id", "")).strip(),
                }
            )

        normalized.sort(
            key=lambda item: (
                item["start_frame"],
                item["end_frame"],
                item["sort_index"],
            )
        )

        self.sections = []
        for section_id, item in enumerate(normalized, start=1):
            label = item["label"] or f"Section {section_id}"
            self.sections.append(
                VideoSection(
                    section_id=section_id,
                    label=label,
                    group_id=item["group_id"],
                    start_frame=item["start_frame"],
                    end_frame=item["end_frame"],
                )
            )

        self.sections_dirty = True
        if reassign_annotations:
            self._reassign_annotation_sections()
        self.save_sections()

    def _reassign_annotation_sections(self):
        changed = False
        for frame, ann in self.annotations.items():
            section = self.section_at_frame(frame)
            new_section_id = section.section_id if section else ""
            new_section_label = section.label if section else ""
            if ann.get("section_ID") != new_section_id:
                ann["section_ID"] = new_section_id
                changed = True
            if ann.get("section_label", "") != new_section_label:
                ann["section_label"] = new_section_label
                changed = True
            if section and section.group_id:
                if ann.get("group_ID") != section.group_id:
                    ann["group_ID"] = section.group_id
                    ann["group"] = section.group_id
                    changed = True
        if changed:
            self.dirty = True

    def update_section(
        self,
        section_id: int,
        *,
        label: str | None = None,
        group_id: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> VideoSection | None:
        section = self.section_by_id(section_id)
        if section is None:
            return None
        if label is not None:
            section.label = label
        if group_id is not None:
            section.group_id = group_id
        if start_frame is not None:
            section.start_frame = self._clamp_frame(start_frame)
        if end_frame is not None:
            section.end_frame = self._clamp_frame(end_frame)
        if section.end_frame < section.start_frame:
            section.start_frame, section.end_frame = section.end_frame, section.start_frame
        self._renumber_sections_by_start()
        self.sync_annotation_sections()
        self.sections_dirty = True
        self.save_sections()
        return section

    def preview_update_section(
        self,
        section_id: int,
        start_frame: int,
        end_frame: int,
    ) -> VideoSection | None:
        section = self.section_by_id(section_id)
        if section is None:
            return None
        section.start_frame = self._clamp_frame(start_frame)
        section.end_frame = self._clamp_frame(end_frame)
        if section.end_frame < section.start_frame:
            section.start_frame, section.end_frame = section.end_frame, section.start_frame
        return section

    def delete_section(self, section_id: int):
        for ann in self.annotations.values():
            if self._parse_int(ann.get("section_ID")) == section_id:
                ann["section_ID"] = ""
                ann["section_label"] = ""
                self.dirty = True
        self.sections = [section for section in self.sections if section.section_id != section_id]
        self._renumber_sections_by_start()
        self.sync_annotation_sections()
        self.sections_dirty = True
        self.save_sections()

    def section_by_id(self, section_id) -> VideoSection | None:
        parsed = self._parse_int(section_id)
        if parsed is None:
            return None
        for section in self.sections:
            if section.section_id == parsed:
                return section
        return None

    def section_at_frame(self, frame: int) -> VideoSection | None:
        matches = [section for section in self.sections if section.contains(frame)]
        if not matches:
            return None
        return sorted(matches, key=lambda section: (section.start_frame, section.section_id))[0]

    def next_section_after(self, frame: int) -> VideoSection | None:
        future = [section for section in self.sections if section.start_frame > frame]
        if not future:
            return None
        return sorted(future, key=lambda section: (section.start_frame, section.section_id))[0]

    def annotation_count_for_section_id(self, section_id: int) -> int:
        section = self.section_by_id(section_id)
        if section is None:
            return 0
        return sum(
            1
            for frame, ann in self.annotations.items()
            if self._parse_int(ann.get("section_ID")) == section_id
            or (
                self._parse_int(ann.get("section_ID")) is None
                and section.contains(frame)
            )
        )

    def export_calculated_csv(self, csv_path: str):
        self.sync_annotation_sections()
        rows: dict[tuple, int] = {}
        for frame, ann in self.annotations.items():
            mode = ann.get("mode")
            if not mode:
                continue
            section = self._annotation_section(frame, ann)
            section_id = section.section_id if section else ann.get("section_ID")
            section_label = section.label if section else ann.get("section_label", "")
            group_id = ann.get("group_ID") or ann.get("group", "")
            if section:
                if section_id in ("", None):
                    section_id = section.section_id
                if section.group_id:
                    group_id = section.group_id
            key = (section_id or "", section_label, group_id, mode)
            rows[key] = rows.get(key, 0) + 1

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "section_ID",
                    "section_label",
                    "group_ID",
                    "mode",
                    "mode_count",
                    "total_frames",
                    "total_seconds",
                    "fps",
                ],
            )
            writer.writeheader()
            for key, count in sorted(rows.items(), key=lambda item: tuple(str(part) for part in item[0])):
                section_id, section_label, group_id, mode = key
                writer.writerow(
                    {
                        "section_ID": section_id,
                        "section_label": section_label,
                        "group_ID": group_id,
                        "mode": mode,
                        "mode_count": count,
                        "total_frames": count,
                        "total_seconds": round(count / max(self.fps, 1.0), 6),
                        "fps": self.fps,
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
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index + 1)
        else:
            frame = self._read_from_cap(frame_index)

        if frame is None:
            return None

        self.current_frame = frame_index
        return frame

    # -----------------------------
    # Audio
    # -----------------------------
    def get_audio_samples(self) -> tuple["np.ndarray | None", int]:
        """Return (mono float32 samples, sample_rate), or (None, 0) if no audio."""
        if not self.path:
            return None, 0
        try:
            clip = VideoFileClip(self.path)
            if clip.audio is None:
                clip.close()
                return None, 0
            sample_rate = int(clip.audio.fps)
            samples = clip.audio.to_soundarray()
            clip.close()
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            samples = samples.astype(np.float32)
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples /= peak
            return samples, sample_rate
        except Exception:
            return None, 0

    # -----------------------------
    # Labels
    # -----------------------------
    def set_label(self, frame, mode, group, section: VideoSection | None = None, autosave: bool = True):
        section_id = section.section_id if section else ""
        section_label = section.label if section else ""
        group_id = section.group_id if section and section.group_id else group
        self.annotations[frame] = {
            "mode": mode,
            "group": group_id,
            "group_ID": group_id,
            "section_ID": section_id,
            "section_label": section_label,
        }
        self.dirty = True
        if autosave:
            self.save_csv()

    def get_label(self, frame=None):
        if frame is None:
            frame = self.current_frame
        return self.annotations.get(frame)
