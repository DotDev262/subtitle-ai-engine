from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[4] / "assets"


def get_demo_asset_paths() -> dict[str, Path]:
    return {
        "video": ASSETS_DIR / "demo.mp4",
        "srt_en": ASSETS_DIR / "sample_en.srt",
        "srt_hi": ASSETS_DIR / "sample_hi.srt",
    }


def load_demo_srt(lang: str = "en") -> str:
    paths = get_demo_asset_paths()
    key = "srt_hi" if lang == "hi" else "srt_en"
    path = paths[key]
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
