# -*- coding: utf-8 -*-
from matplotlib.colors import rgb2hex
import matplotlib.cm as cm
import numpy as np
# Colours drawn from material designs colour pallet at https://material.io/guidelines/style/color.html


class Colors(object):
    def __init__(self):
        self.color_map = {
            "blue": "#1976D2",
            "lblue": "#4FC3F7",
            "red": "#E53935",
            "green": "#43A047",
            "lgreen": "#8BC34A",
            "purple": "#673AB7",
            "cyan": "#4DD0E1",
            "magenta": "#E91E63",
            "yellow": "#F2D026",
            "black": "#333333",
            "grey": "#9E9E9E",
            "orange": "#FB8C00",
            "amber": "#FFB300",
            "brown": "#795548"
        }
        self.aliases = {
            "b": "blue", "r": "red", "g": "green", "k": "black", "m": "magenta", "c": "cyan",
            "o": "orange", "y": "yellow", "a": "amber", "p": "purple",
            "e": "grey", "lg": "lgreen", "lb": "lblue"
        }
        self.default_colors = ["blue", "lgreen", "red", "purple", "yellow", "grey",
                               "lblue", "magenta", "green", "brown", "black", "orange"]

    def format(self, color):
        if isinstance(color, np.ndarray):
            color = rgb2hex(color)
        if color[0] == "#":
            return color
        elif color in self.color_map:
            return self.color_map[color]
        elif color in self.aliases:
            alias = self.aliases[color]
            return self.color_map[alias]
        else:
            raise ValueError("Color %s is not mapped. Please give a hex code" % color)

    def get_formatted(self, list_colors):
        return [self.format(c) for c in list_colors]

    def get_default(self):
        return self.get_formatted(self.default_colors)

    def get_colormap(self, num, scale=0.7):  # pragma: no cover
        color_list = self.get_formatted(cm.rainbow(np.linspace(0, 1, num)))
        scales = scale + (1 - scale) * np.abs(1 - np.linspace(0, 2, num))
        scaled = [self.scale_colour(c, s) for c, s in zip(color_list, scales)]
        return scaled

    def scale_colour(self, colour, scalefactor):  # pragma: no cover
        if isinstance(colour, np.ndarray):
            r, g, b = colour[:3] * 255.0
        else:
            hexx = colour.strip('#')
            if scalefactor < 0 or len(hexx) != 6:
                return hexx
            r, g, b = int(hexx[:2], 16), int(hexx[2:4], 16), int(hexx[4:], 16)
        r = self._clamp(int(r * scalefactor))
        g = self._clamp(int(g * scalefactor))
        b = self._clamp(int(b * scalefactor))
        return "#%02x%02x%02x" % (r, g, b)

    def _clamp(self, val, minimum=0, maximum=255):
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val
