import json
import os


DEFAULT_SETTINGS_PATH = os.path.join(
    os.path.dirname(__file__),
    "encode_settings.json",
)


class SettingsLoader:
    """Utility to read encode_settings.json into a python object."""

    def __init__(self, path: str = DEFAULT_SETTINGS_PATH):
        self.path = path
        self.reload()

    def reload(self):
        self.data = self._load(self.path)
        self.labels = self.data.get("labels", {})
        self.app_keys = self.data.get("app_keys", {})
        self.mouse = self.data.get("mouse", {})
        self.timings = self.data.get("timings", {})
        self.timeline = self.data.get("timeline", {})
        self.sections = self.data.setdefault("sections", {})
        self.groups = self._load_groups()

    def _load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"encode_settings.json が見つかりません: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_groups(self) -> list[str]:
        groups = self.data.get("groups")
        if isinstance(groups, list):
            cleaned = [str(group).strip() for group in groups if str(group).strip()]
            if cleaned:
                return cleaned

        derived: list[str] = []
        for data in self.labels.values():
            group = str(data.get("group", "")).strip()
            if group and group not in derived:
                derived.append(group)
        self.data["groups"] = derived
        return derived

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        self.reload()

    def default_section_seconds(self) -> float:
        try:
            return float(self.sections.get("default_duration_seconds", 10))
        except Exception:
            return 10.0
