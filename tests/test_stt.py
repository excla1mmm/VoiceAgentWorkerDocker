import numpy as np

from voice_agent_worker.stt import speech_region

RATE = 8000


def tone(seconds: float, level: int = 3000) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (np.sin(2 * np.pi * 300 * t) * level).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(RATE * seconds), dtype=np.int16)


def test_nothing_said():
    assert speech_region(silence(3), RATE) is None
    assert speech_region(np.zeros(10, dtype=np.int16), RATE) is None


def test_a_click_is_not_speech():
    assert speech_region(np.concatenate([silence(1), tone(0.04), silence(1)]), RATE) is None


def test_the_speech_before_the_end_of_utterance():
    region = speech_region(np.concatenate([silence(5), tone(1), silence(0.2)]), RATE)
    assert 1.0 <= len(region) / RATE <= 1.45


def test_an_old_noise_is_cut_off():
    region = speech_region(np.concatenate([tone(1), silence(4), tone(1), silence(0.2)]), RATE)
    assert len(region) / RATE < 1.45


def test_a_pause_inside_the_utterance_is_kept():
    region = speech_region(np.concatenate([tone(1), silence(0.6), tone(1), silence(0.2)]), RATE)
    assert len(region) / RATE > 2.5


def test_a_long_utterance_keeps_its_end():
    region = speech_region(tone(40), RATE)
    assert len(region) / RATE <= 30.1
