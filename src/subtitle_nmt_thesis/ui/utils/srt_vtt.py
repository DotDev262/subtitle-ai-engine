import re


def srt_to_vtt(srt_text: str) -> str:
    """Converts SubRip (.srt) formatted text into WebVTT (.vtt) format."""
    clean_text = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not clean_text:
        return "WEBVTT\n"
    # Convert timestamps: 00:00:01,000 --> 00:00:01.000
    vtt_timestamps = re.sub(
        r"(\d{2}:\d{2}:\d{2}),(\d{3})",
        r"\1.\2",
        clean_text,
    )
    return f"WEBVTT\n\n{vtt_timestamps}\n"
