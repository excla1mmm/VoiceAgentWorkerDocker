import pytest

from voice_agent_worker.server import take_sentences
from voice_agent_worker.tts import readable, voice_files


@pytest.mark.parametrize(("text", "expected"), [
    ("МИК+О на связи", "мико на связи"),
    ("Подождите sil<[400]> минуту", "Подождите, минуту"),
    ("Тариф стоит 2900 рублей", "Тариф стоит две тысячи девятьсот рублей"),
])
def test_readable(text, expected):
    assert readable(text) == expected


def test_take_sentences():
    assert take_sentences("Да. Сейчас соединю") == (["Да. "], "Сейчас соединю")
    assert take_sentences("Сейчас") == ([], "Сейчас")
    assert take_sentences("Хорошо! Спасибо?") == (["Хорошо! ", "Спасибо?"], "")


def test_voice_files():
    assert voice_files("ru_RU-irina-medium")[0] == "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"
