import base64
import hashlib
import json
import logging
import os
import tempfile
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


class PersistentCache:
    """Simple file-based cache with text/audio buckets and basic stats."""

    def __init__(self, cache_dir: Optional[str] = None, language: str = "en"):
        self.language = language
        base_dir = cache_dir or os.environ.get("CACHE_DIR", "cache")
        self.cache_dir = Path(base_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.text_cache_dir = self.cache_dir / "text"
        self.audio_cache_dir = self.cache_dir / "audio"
        self.stats_file = self.cache_dir / "stats.json"

        self.text_cache_dir.mkdir(exist_ok=True)
        self.audio_cache_dir.mkdir(exist_ok=True)

        self._init_stats()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def get(self, prompt: str, context: str = "", cache_type: str = "text") -> Optional[Any]:
        """Retrieve cached payload."""
        key = self._generate_key(prompt, context)
        cache_path = self._cache_path_for_key(key, cache_type)
        if not cache_path.exists():
            self._update_stats("misses")
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Cache read failure for %s: %s", cache_path, exc)
            self._update_stats("misses")
            return None

        self._update_stats("hits")
        return self._deserialize(payload)

    def set(self, prompt: str, response: Any, context: str = "", cache_type: str = "text") -> None:
        """Persist payload in cache."""
        key = self._generate_key(prompt, context)
        cache_path = self._cache_path_for_key(key, cache_type)

        serialized = self._serialize(response)
        data = {
            "prompt": prompt,
            "context": context,
            "response": serialized["payload"],
            "encoding": serialized["encoding"],
            "language": self.language,
        }

        try:
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Cache write failure for %s: %s", cache_path, exc)
            return

        self._update_stats("total_entries", set_count=True)

    def memoize(self, cache_type: str = "text") -> Callable:
        """Lightweight memoization decorator using PersistentCache."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                context = self._memo_context(func, args, kwargs)
                cached = self.get(func.__name__, context=context, cache_type=cache_type)
                if cached is not None:
                    return cached

                result = func(*args, **kwargs)
                self.set(func.__name__, result, context=context, cache_type=cache_type)
                return result

            return wrapper

        return decorator

    def get_stats(self) -> Dict[str, Any]:
        stats = self._load_stats()
        stats["cache_size_mb"] = self._cache_size_mb()
        stats["hit_rate"] = self._hit_rate(stats)
        stats["language"] = self.language
        return stats

    def clear(self, cache_type: Optional[str] = None) -> None:
        if cache_type == "text":
            self._clear_dir(self.text_cache_dir)
        elif cache_type == "audio":
            self._clear_dir(self.audio_cache_dir)
        else:
            self._clear_dir(self.text_cache_dir)
            self._clear_dir(self.audio_cache_dir)
        self._init_stats(force=True)

    def create_zip_snapshot(self) -> Tuple[Optional[str], Optional[str]]:
        """Package cache directory into a zip archive."""
        try:
            temp_dir = tempfile.gettempdir()
            archive_path = os.path.join(temp_dir, "cache_archive.zip")
            if os.path.isfile(archive_path):
                os.remove(archive_path)

            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                for path in self.cache_dir.rglob("*"):
                    if path.is_file():
                        zf.write(path, arcname=path.relative_to(self.cache_dir))

            return archive_path, None
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error("Failed to create cache archive: %s", exc, exc_info=True)
            return None, f"Error creating cache archive: {exc}"

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _init_stats(self, force: bool = False) -> None:
        if self.stats_file.exists() and not force:
            return
        stats = {"hits": 0, "misses": 0, "total_entries": 0, "language": self.language}
        self._save_stats(stats)

    def _generate_key(self, prompt: str, context: str) -> str:
        content = f"{self.language}:{prompt}:{context}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _cache_path_for_key(self, key: str, cache_type: str) -> Path:
        directory = self.text_cache_dir if cache_type == "text" else self.audio_cache_dir
        return directory / f"{key}.json"

    def _load_stats(self) -> Dict[str, Any]:
        try:
            with self.stats_file.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:  # pragma: no cover - fallback
            return {"hits": 0, "misses": 0, "total_entries": 0, "language": self.language}

    def _save_stats(self, stats: Dict[str, Any]) -> None:
        with self.stats_file.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)

    def _update_stats(self, field: str, set_count: bool = False) -> None:
        stats = self._load_stats()
        if set_count and field == "total_entries":
            text_entries = len(list(self.text_cache_dir.glob("*.json")))
            audio_entries = len(list(self.audio_cache_dir.glob("*.json")))
            stats[field] = text_entries + audio_entries
        else:
            stats[field] = stats.get(field, 0) + 1
        self._save_stats(stats)

    def _cache_size_mb(self) -> float:
        total = 0
        for path in self.cache_dir.rglob("*.json"):
            total += path.stat().st_size
        return round(total / (1024 * 1024), 2)

    @staticmethod
    def _hit_rate(stats: Dict[str, Any]) -> float:
        hits = stats.get("hits", 0)
        misses = stats.get("misses", 0)
        total = hits + misses
        return round((hits / total) * 100, 2) if total else 0.0

    @classmethod
    def _serialize(cls, response: Any) -> Dict[str, Any]:
        encoded = cls._encode_value(response)
        return {"payload": encoded, "encoding": "structured"}

    @classmethod
    def _deserialize(cls, payload: Dict[str, Any]) -> Any:
        data = payload.get("response")
        return cls._decode_value(data)

    @classmethod
    def _encode_value(cls, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        if isinstance(value, (bytes, bytearray)):
            return {"__type": "bytes", "data": base64.b64encode(value).decode("utf-8")}

        if isinstance(value, list):
            return {"__type": "list", "data": [cls._encode_value(item) for item in value]}

        if isinstance(value, tuple):
            return {"__type": "tuple", "data": [cls._encode_value(item) for item in value]}

        if isinstance(value, dict):
            return {"__type": "dict", "data": {key: cls._encode_value(val) for key, val in value.items()}}

        return {"__type": "repr", "data": repr(value)}

    @classmethod
    def _decode_value(cls, value: Any) -> Any:
        if isinstance(value, dict) and "__type" in value:
            tag = value["__type"]
            data = value.get("data")
            if tag == "bytes" and isinstance(data, str):
                return base64.b64decode(data.encode("utf-8"))
            if tag == "list":
                return [cls._decode_value(item) for item in data]
            if tag == "tuple":
                return tuple(cls._decode_value(item) for item in data)
            if tag == "dict":
                return {key: cls._decode_value(val) for key, val in data.items()}
            if tag == "repr":
                return data
        return value

    @staticmethod
    def _memo_context(func: Callable, args: tuple, kwargs: dict) -> str:
        try:
            serialized = json.dumps({"args": args, "kwargs": kwargs}, ensure_ascii=False, default=str)
        except TypeError:
            serialized = str((args, kwargs))
        return f"{func.__module__}.{func.__name__}:{serialized}"

    @staticmethod
    def _clear_dir(directory: Path) -> None:
        for file_path in directory.glob("*.json"):
            try:
                file_path.unlink()
            except OSError as exc:  # pragma: no cover - defensive logging
                logging.warning("Failed to delete cache file %s: %s", file_path, exc)


# ------------------------------------------------------------------------- #
# Module-level helpers
# ------------------------------------------------------------------------- #

cache = PersistentCache()


def create_cache_zip() -> Tuple[Optional[str], Optional[str]]:
    """Compatibility wrapper for API endpoint."""
    return cache.create_zip_snapshot()
