import numpy as np
import pytest

from chainconsumer.color_finder import ALL_COLOURS, colors


def test_colors_rgb2hex_1():
    c = np.array([1, 1, 1, 1])
    assert colors.get_formatted([c])[0] == "#ffffff"


def test_colors_rgb2hex_2():
    c = np.array([0, 0, 0.5, 1])
    assert colors.get_formatted([c])[0] == "#000080"


def test_colors_alias_works():
    assert colors.format("b") in ALL_COLOURS["blue"]


def test_colors_name_works():
    assert colors.format("blue") in ALL_COLOURS["blue"]


def test_colors_error_on_garbage():
    with pytest.raises(ValueError):
        colors.get_formatted(["java"])


def test_clamp1():
    assert colors._clamp(-10) == 0


def test_clamp2():
    assert colors._clamp(10) == 10


def test_clamp3():
    assert colors._clamp(1000) == 255
