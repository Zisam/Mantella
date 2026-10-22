import json
from pathlib import Path
import pytest
from src.character_manager import Character
from src.conversation.conversation_log import conversation_log


WORLD_ID = "01A2B3"


@pytest.fixture
def log_env(tmp_path: Path, example_skyrim_npc_character: Character):
    original_game_path = conversation_log.game_path
    conversation_log.game_path = str(tmp_path)
    folder = tmp_path / WORLD_ID / example_skyrim_npc_character.name
    folder.mkdir(parents=True)
    history_file = folder / f"{example_skyrim_npc_character.name}.json"
    yield history_file, example_skyrim_npc_character
    conversation_log.game_path = original_game_path


def write_history(history_file: Path, conversations: list, encoding: str) -> None:
    with open(history_file, 'w', encoding=encoding) as f:
        json.dump(conversations, f, indent=4)


def message(content: str) -> dict:
    return {"role": "user", "content": content}


def test_load_conversation_log_reads_plain_utf8(log_env):
    history_file, character = log_env
    write_history(history_file, [[message("hello")], [message("hi"), message("bye")]], encoding='utf-8')

    loaded = conversation_log.load_conversation_log(character, WORLD_ID)

    assert [m["content"] for m in loaded] == ["hello", "hi", "bye"]


def test_load_conversation_log_tolerates_utf8_bom(log_env):
    history_file, character = log_env
    write_history(history_file, [[message("hello")]], encoding='utf-8-sig')
    assert history_file.read_bytes()[:3] == b'\xef\xbb\xbf'  # sanity: BOM actually written

    loaded = conversation_log.load_conversation_log(character, WORLD_ID)

    assert [m["content"] for m in loaded] == ["hello"]


def test_get_conversation_log_length_tolerates_utf8_bom(log_env):
    history_file, character = log_env
    write_history(history_file, [[message("a"), message("b")], [message("c")]], encoding='utf-8-sig')

    assert conversation_log.get_conversation_log_length(character, WORLD_ID) == 3


def test_get_conversation_log_length_missing_file_returns_zero(log_env):
    _, character = log_env

    assert conversation_log.get_conversation_log_length(character, WORLD_ID) == 0


def test_save_appends_to_bom_file_and_normalizes_it(log_env):
    history_file, character = log_env
    write_history(history_file, [[message("old")]], encoding='utf-8-sig')

    conversation_log.save_conversation_log(character, [message("new1"), message("new2")], WORLD_ID)

    raw = history_file.read_bytes()
    assert raw[:3] != b'\xef\xbb\xbf'  # rewritten by Mantella without BOM
    with open(history_file, 'r', encoding='utf-8') as f:  # strict utf-8 must now parse
        history = json.load(f)
    assert len(history) == 2
    assert conversation_log.get_conversation_log_length(character, WORLD_ID) == 3
