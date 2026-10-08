"""Formatting helpers retained for subtitle exports and tests."""


def format_timestamp(start: float, end: float) -> str:
    def to_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        whole = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{whole:02d}.{millis:03d}"

    return f"{to_timestamp(start)} --> {to_timestamp(end)}"


def get_badge_html(label: str, value: str | int | float, is_compliant: bool) -> str:
    color = "#28a745" if is_compliant else "#dc3545"
    background = "#e8f5e9" if is_compliant else "#ffebee"
    icon = "🟢" if is_compliant else "🔴"
    return (
        f"<span style='background-color: {background}; color: {color}; "
        f"border: 1px solid {color}; padding: 2px 8px; border-radius: 12px; "
        f"font-size: 12px; font-weight: 600; margin-right: 6px;'>"
        f"{icon} {label}: {value}</span>"
    )
