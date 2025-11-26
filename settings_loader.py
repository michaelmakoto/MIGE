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
        self.data = self._load(path)

        self.labels = self.data.get("labels", {})
        self.app_keys = self.data.get("app_keys", {})
        self.mouse = self.data.get("mouse", {})
        self.timings = self.data.get("timings", {})
        self.timeline = self.data.get("timeline", {})

    def _load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"encode_settings.json が見つかりません: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
