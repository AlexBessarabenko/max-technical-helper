"""Yandex SpeechKit STT: распознавание голосовых сообщений (requests замокан)."""

from unittest.mock import MagicMock, patch

from src.bot.speech import transcribe


def _settings():
    s = MagicMock()
    s.yandex_api_key = "test-key"
    s.yandex_folder_id = "test-folder"
    return s


@patch("src.bot.speech.requests.post")
def test_transcribe_success(mock_post):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"result": "как настроить vpn"}
    mock_post.return_value = resp

    assert transcribe(b"ogg-bytes", _settings()) == "как настроить vpn"

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    assert kwargs["params"] == {
        "folderId": "test-folder",
        "lang": "ru-RU",
        "format": "oggopus",
    }
    assert kwargs["headers"] == {"Authorization": "Api-Key test-key"}
    assert kwargs["data"] == b"ogg-bytes"
    assert kwargs["timeout"] == 30


@patch("src.bot.speech.requests.post")
def test_transcribe_empty_result_returns_none(mock_post):
    # Тишина/неразборчиво: API отвечает 200 с пустым result.
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"result": ""}
    mock_post.return_value = resp
    assert transcribe(b"ogg-bytes", _settings()) is None


@patch("src.bot.speech.requests.post")
def test_transcribe_api_error_returns_none(mock_post):
    resp = MagicMock(status_code=400)
    resp.text = "bad request"
    mock_post.return_value = resp
    assert transcribe(b"ogg-bytes", _settings()) is None


@patch("src.bot.speech.requests.post")
def test_transcribe_network_error_returns_none(mock_post):
    mock_post.side_effect = OSError("connection refused")
    assert transcribe(b"ogg-bytes", _settings()) is None
