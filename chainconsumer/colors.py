from matplotlib.colors import rgb2hex
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
            "yellow": "#FFEB3B",
            "black": "#333333",
            "grey": "#9E9E9E",
            "orange": "#FB8C00",
            "amber": "#FFB300",
            "brown": "#795548"
        }
        self.aliases = {
            "b": "blue", "r": "red", "g": "green", "k": "black", "m": "magenta", "c": "cyan",
            "o": "orange", "y": "yellow", "a": "amber",
            "e": "grey", "lg": "lgreen", "lb": "lblue"
        }
        self.default_colors = ["blue", "red", "green", "purple", "yellow",
                               "lblue", "magenta", "lgreen", "brown", "black", "grey"]

    def get_formatted(self, list_colors):
        formatted = []
        for c in list_colors:
            if isinstance(c, np.ndarray):
                c = rgb2hex(c)
            if c[0] == "#":
                formatted.append(c)
            elif c in self.color_map:
                formatted.append(self.color_map[c])
            elif c in self.aliases:
                alias = self.aliases[c]
                formatted.append(self.color_map[alias])
            else:
                raise ValueError("Color %s is not mapped. Please give a hex code" % c)
        return formatted

    def get_default(self):
        return self.get_formatted(self.default_colors)
