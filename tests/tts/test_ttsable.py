from src.tts.ttsable import sanitize_voiceline_cache_name


def test_control_characters_are_replaced():
    name = "MaleKhajiit Дж'зарго уже слышал о\tnew\tученике"
    sanitized = sanitize_voiceline_cache_name(name)
    assert "\t" not in sanitized
    assert sanitized == "MaleKhajiit Дж'зарго уже слышал о new ученике"


def test_illegal_filename_chars_are_replaced():
    sanitized = sanitize_voiceline_cache_name('Voice a/b\\c:d*e?f"g<h>i|j.')
    for ch in '\\/:*?"<>|.':
        assert ch not in sanitized
    assert sanitized == "Voice a b c d e f g h i j"


def test_whitespace_runs_are_collapsed():
    assert sanitize_voiceline_cache_name("Voice  a \n\n b \r c") == "Voice a b c"


def test_normal_text_is_untouched():
    assert sanitize_voiceline_cache_name("MaleEvenToned Привет, Зисам!") == "MaleEvenToned Привет, Зисам!"
