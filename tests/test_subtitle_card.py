from subtitle_nmt_thesis.ui.components.subtitle_card import format_timestamp, get_badge_html


def test_format_timestamp():
    assert format_timestamp(1.5, 4.25) == "00:00:01.500 --> 00:00:04.250"


def test_get_badge_html():
    green = get_badge_html("CPL", 35, is_compliant=True)
    assert "🟢" in green
    assert "35" in green

    red = get_badge_html("CPL", 55, is_compliant=False)
    assert "🔴" in red
    assert "55" in red
