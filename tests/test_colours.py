import numpy as np
import pytest

from chainconsumer.colors import Colors


def test_colors_rgb2hex_1():
    c = np.array([1, 1, 1, 1])
    colourmap = Colors()
    assert colourmap.get_formatted([c])[0] == "#ffffff"


def test_colors_rgb2hex_2():
    c = np.array([0, 0, 0.5, 1])
    colourmap = Colors()
    assert colourmap.get_formatted([c])[0] == "#000080"


def test_colors_alias_works():
    colourmap = Colors()
    assert colourmap.get_formatted(["b"])[0] == colourmap.color_map["blue"]


def test_colors_name_works():
    colourmap = Colors()
    assert colourmap.get_formatted(["blue"])[0] == colourmap.color_map["blue"]


def test_colors_error_on_garbage():
    colourmap = Colors()
    with pytest.raises(ValueError):
        colourmap.get_formatted(["java"])


def test_clamp1():
    assert Colors()._clamp(-10) == 0


def test_clamp2():
    assert Colors()._clamp(10) == 10


def test_clamp3():
    assert Colors()._clamp(1000) == 255
