import ast
import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from src.stt.stt import Transcriber

SRC_ROOT = Path(__file__).parents[2] / "src"


def _class_node(module_path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    pytest.fail(f"Class {class_name} not found in {module_path}")


def _methods(class_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in class_node.body if isinstance(n, ast.FunctionDef)}


def _init_self_attributes(init_node: ast.FunctionDef) -> set[str]:
    names = set()
    assignment_nodes: list[ast.stmt] = [n for n in ast.walk(init_node) if isinstance(n, (ast.Assign, ast.AnnAssign))]
    for node in assignment_nodes:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                names.add(target.attr)
    return names


def _transcriber_lists() -> dict[str, list]:
    """Extracts the real ignore list / hallucination markers from source without instantiating Transcriber"""
    init = _methods(_class_node(SRC_ROOT / "stt" / "stt.py", "Transcriber"))["__init__"]
    lists: dict[str, list] = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr in ("__ignore_list", "__hallucination_markers"):
                    lists[target.attr] = ast.literal_eval(node.value)
    return lists


def _bare_transcriber() -> Transcriber:
    """A Transcriber instance built without running the heavy __init__ (no Whisper / VAD loading)"""
    lists = _transcriber_lists()
    transcriber = Transcriber.__new__(Transcriber)
    transcriber._Transcriber__ignore_list = lists["__ignore_list"]
    transcriber._Transcriber__hallucination_markers = lists["__hallucination_markers"]
    return transcriber


class TestTranscriberStructure:
    """Guards against a regression where a helper method was inserted into the middle of
    Transcriber.__init__, silently truncating it (attributes like _lock were never assigned)"""

    CRITICAL_INIT_ATTRIBUTES = {
        "transcribe_model", "vad", "_audio_buffer", "_audio_queue", "_stream",
        "_lock", "_processing_thread", "_running", "_speech_detected",
        "_speech_end_time", "_transcription_ready", "_current_transcription",
        "_consecutive_empty_count", "__ignore_list", "__hallucination_markers",
    }

    def _transcriber_class(self) -> ast.ClassDef:
        return _class_node(SRC_ROOT / "stt" / "stt.py", "Transcriber")

    def test_init_assigns_all_critical_attributes(self):
        init = _methods(self._transcriber_class())["__init__"]
        missing = self.CRITICAL_INIT_ATTRIBUTES - _init_self_attributes(init)
        assert not missing, (
            f"Transcriber.__init__ no longer assigns: {sorted(missing)}. "
            "The method body was probably truncated by an edit"
        )

    def test_init_contains_no_nested_method_definitions(self):
        init = _methods(self._transcriber_class())["__init__"]
        nested_defs = [n for n in ast.walk(init) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not init]
        assert not nested_defs, (
            f"__init__ contains nested def(s) at line {[n.lineno for n in nested_defs]}: "
            "a helper method was likely inserted mid-init"
        )

    def test_hallucination_filter_is_called_for_every_whisper_backend(self):
        source = (SRC_ROOT / "stt" / "stt.py").read_text(encoding="utf-8")
        assert source.count("self.__is_whisper_hallucination(") >= 3, (
            "Every whisper_transcribe branch (local, OpenAI-compatible, custom server) must filter hallucinations"
        )

    def test_local_whisper_uses_vad_filter(self):
        source = (SRC_ROOT / "stt" / "stt.py").read_text(encoding="utf-8")
        assert "vad_filter=True" in source and "vad_filter=False" not in source, (
            "Whisper must run with vad_filter=True: without it, silence and room noise decode "
            "into deterministic hallucinations like «Редактор субтитров Н.Закомолдина»"
        )


class TestMantellaRouteStructure:
    """Guards the request serialization lock introduced to let end_conversation run
    while a blocked player_input is waiting on the microphone"""

    def test_route_lock_is_initialized(self):
        init = _methods(_class_node(SRC_ROOT / "http" / "routes" / "mantella_route.py", "mantella_route"))["__init__"]
        assert "_route_lock" in _init_self_attributes(init)

    def test_end_conversation_is_not_serialized(self):
        source = (SRC_ROOT / "http" / "routes" / "mantella_route.py").read_text(encoding="utf-8")
        end_branch = source.split("if request_type == comm_consts.KEY_REQUESTTYPE_ENDCONVERSATION:")[1].split("else:")[0]
        assert "end_conversation" in end_branch
        assert "_route_lock" not in end_branch, (
            "end_conversation must bypass _route_lock, otherwise it deadlocks on a mic-blocked player_input"
        )

    def test_conversation_end_wakes_mic_waiter(self):
        """Conversation.end must call stop_listening: it is the only code path that runs in
        parallel (unserialized) with a player_input request blocked on the microphone"""
        end_method = _methods(_class_node(SRC_ROOT / "conversation" / "conversation.py", "Conversation"))["end"]
        calls = {n.func.attr for n in ast.walk(end_method)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert "stop_listening" in calls, (
            "Conversation.end must call stop_listening to wake a player_input blocked on the mic"
        )


class TestWhisperHallucinationFilter:
    """Behavior of the anti-hallucination filter using the real lists from source"""

    def test_exact_english_hallucinations_are_dropped(self):
        transcriber = _bare_transcriber()
        assert transcriber._Transcriber__is_whisper_hallucination("Thank you for watching.")

    def test_exact_russian_hallucinations_are_dropped(self):
        transcriber = _bare_transcriber()
        for phrase in ["Продолжение следует...", "Канал.", "продолжение следует"]:
            assert transcriber._Transcriber__is_whisper_hallucination(phrase), phrase

    def test_short_phrases_with_markers_are_dropped(self):
        transcriber = _bare_transcriber()
        for phrase in [
            "Добро пожаловать на наш канал!",
            "Редактор субтитров Н.Закомолдина",
            "Субтитры субтитров А.Семкин",
            "Ставьте лайки",
            "С вами был Игорь Негода.",
            "Не переключайтесь",
        ]:
            assert transcriber._Transcriber__is_whisper_hallucination(phrase), phrase

    def test_long_legitimate_lines_mentioning_markers_pass(self):
        transcriber = _bare_transcriber()
        phrase = "Опять редактор субтитров! Их там целая армия — Бойкова, Семкин и Сухиашвили идут за мной!"
        assert not transcriber._Transcriber__is_whisper_hallucination(phrase)

    def test_normal_speech_passes(self):
        transcriber = _bare_transcriber()
        for phrase in ["Расскажи про Виндхельм", "Ну что, охотник, где тут кабан прошёл?"]:
            assert not transcriber._Transcriber__is_whisper_hallucination(phrase), phrase

    def test_empty_transcription_is_dropped(self):
        transcriber = _bare_transcriber()
        assert transcriber._Transcriber__is_whisper_hallucination("   ")


class _FakeSegment:
    def __init__(self, text: str):
        self.text = text


class _FakeWhisperModel:
    def __init__(self, text: str):
        self._text = text

    def transcribe(self, audio, **kwargs):
        return [_FakeSegment(self._text)], None


def _transcriber_with_model(text: str) -> Transcriber:
    transcriber = _bare_transcriber()
    transcriber.transcribe_model = _FakeWhisperModel(text)
    transcriber.external_whisper_service = False
    transcriber.stt_service = "whisper"
    transcriber.task = "transcribe"
    transcriber.language = "ru"
    transcriber.whisper_model = "small"
    return transcriber


class TestWhisperTranscribeWiring:
    """The filter must be wired into whisper_transcribe, not just exist as a method"""

    def test_hallucinated_transcription_returns_empty(self):
        transcriber = _transcriber_with_model("Продолжение следует...")
        result = transcriber.whisper_transcribe(np.zeros(1600, dtype=np.float32), "")
        assert result == ""

    def test_normal_transcription_returns_text(self):
        transcriber = _transcriber_with_model("Пойдём через реку, мимо мельницы.")
        result = transcriber.whisper_transcribe(np.zeros(1600, dtype=np.float32), "")
        assert "мельницы" in result


class TestActivationNameExists:
    """Keyword matching semantics used for end-conversation keywords"""

    def test_single_word_keyword_matches_inside_phrase(self):
        assert Transcriber.activation_name_exists("ну пока девочки", ["пока"])

    def test_multi_word_keyword_matches_whole_utterance_only(self):
        assert Transcriber.activation_name_exists("до встречи", ["до встречи"])
        assert not Transcriber.activation_name_exists("ладно, до встречи, девочки", ["до встречи"])

    def test_no_match_returns_false(self):
        assert not Transcriber.activation_name_exists("погнали дальше", ["пока"])


class TestListeningCancellation:
    """Regression: while a player_input request is blocked inside get_latest_transcription
    waiting on the microphone, an unserialized end_conversation request reaches
    Conversation.end -> stop_listening. That must wake the blocked waiter (which returns
    None) instead of letting it hold the HTTP route lock forever."""

    def _armed_transcriber(self) -> Transcriber:
        transcriber = _bare_transcriber()
        transcriber._speech_detected = False
        transcriber._lock = threading.Lock()
        transcriber._current_transcription = ""
        transcriber._transcription_ready = threading.Event()
        transcriber._listening_cancelled = False
        transcriber._running = True
        transcriber._stream = None
        transcriber._processing_thread = None
        transcriber._audio_queue = queue.Queue()
        transcriber.play_cough_sound = False
        transcriber.loglevel = 20
        return transcriber

    def test_already_cancelled_wait_returns_none(self, monkeypatch):
        monkeypatch.setattr(Transcriber, "_reset_state", lambda self: None)
        transcriber = self._armed_transcriber()
        transcriber._listening_cancelled = True
        transcriber._transcription_ready.set()
        assert transcriber.get_latest_transcription(silence_timeout=0) is None

    def test_stop_listening_wakes_blocked_waiter(self, monkeypatch):
        monkeypatch.setattr(Transcriber, "_reset_state", lambda self: None)
        transcriber = self._armed_transcriber()

        result: list = []
        waiter = threading.Thread(
            target=lambda: result.append(transcriber.get_latest_transcription(silence_timeout=0))
        )
        waiter.start()
        # No speech, no timeout -> the waiter must stay blocked on the event...
        waiter.join(timeout=0.3)
        assert waiter.is_alive(), "get_latest_transcription should block until speech or cancellation"

        # ...until stop_listening (e.g. from end_conversation) wakes it
        transcriber.stop_listening()
        waiter.join(timeout=2)
        assert not waiter.is_alive(), "stop_listening must wake the blocked get_latest_transcription call"
        assert result == [None]

    def test_normal_transcription_still_returned_after_reuse(self, monkeypatch):
        """A leftover cancelled event from a previous stop must not swallow the next transcription"""
        monkeypatch.setattr(Transcriber, "_reset_state", lambda self: None)
        transcriber = self._armed_transcriber()
        # simulate: previous session cancelled (event left set), then start_listening resets the flag
        transcriber._listening_cancelled = True
        transcriber._transcription_ready.set()
        transcriber._running = True
        transcriber._listening_cancelled = False
        transcriber._transcription_ready.clear()
        transcriber._current_transcription = "привет, как дела"
        transcriber._transcription_ready.set()
        assert transcriber.get_latest_transcription(silence_timeout=0) == "привет, как дела"
