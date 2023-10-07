from collections.abc import Generator, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb2hex

# Colours drawn from tailwind: https://tailwindcss.com/docs/customizing-colors

ColorInput = str | np.ndarray | list[float]

ALL_COLOURS = {
    "rose": ["#ffe4e6", "#fecdd3", "#fda4af", "#fb7185", "#f43f5e", "#e11d48", "#be123c", "#9f1239", "#881337"],
    "pink": ["#fce7f3", "#fbcfe8", "#f9a8d4", "#f472b6", "#ec4899", "#db2777", "#be185d", "#9d174d", "#831843"],
    "fuchsia": ["#fae8ff", "#f5d0fe", "#f0abfc", "#e879f9", "#d946ef", "#c026d3", "#a21caf", "#86198f", "#701a75"],
    "purple": ["#f3e8ff", "#e9d5ff", "#d8b4fe", "#c084fc", "#a855f7", "#9333ea", "#7e22ce", "#6b21a8", "#581c87"],
    "violet": ["#ede9fe", "#ddd6fe", "#c4b5fd", "#a78bfa", "#8b5cf6", "#7c3aed", "#6d28d9", "#5b21b6", "#4c1d95"],
    "indigo": ["#e0e7ff", "#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1", "#4f46e5", "#4338ca", "#3730a3", "#312e81"],
    "blue": ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e40af", "#1e3a8a"],
    "sky": ["#e0f2fe", "#bae6fd", "#7dd3fc", "#38bdf8", "#0ea5e9", "#0284c7", "#0369a1", "#075985", "#0c4a6e"],
    "cyan": ["#cffafe", "#a5f3fc", "#67e8f9", "#22d3ee", "#06b6d4", "#0891b2", "#0e7490", "#155e75", "#164e63"],
    "teal": ["#ccfbf1", "#99f6e4", "#5eead4", "#2dd4bf", "#14b8a6", "#0d9488", "#0f766e", "#115e59", "#134e4a"],
    "emerald": ["#d1fae5", "#a7f3d0", "#6ee7b7", "#34d399", "#10b981", "#059669", "#047857", "#065f46", "#064e3b"],
    "green": ["#dcfce7", "#bbf7d0", "#86efac", "#4ade80", "#22c55e", "#16a34a", "#15803d", "#166534", "#14532d"],
    "lime": ["#ecfccb", "#d9f99d", "#bef264", "#a3e635", "#84cc16", "#65a30d", "#4d7c0f", "#3f6212", "#365314"],
    "yellow": ["#fef9c3", "#fef08a", "#fde047", "#facc15", "#eab308", "#ca8a04", "#a16207", "#854d0e", "#713f12"],
    "amber": ["#fef3c7", "#fde68a", "#fcd34d", "#fbbf24", "#f59e0b", "#d97706", "#b45309", "#92400e", "#78350f"],
    "orange": ["#ffedd5", "#fed7aa", "#fdba74", "#fb923c", "#f97316", "#ea580c", "#c2410c", "#9a3412", "#7c2d12"],
    "red": ["#fee2e2", "#fecaca", "#fca5a5", "#f87171", "#ef4444", "#dc2626", "#b91c1c", "#991b1b", "#7f1d1d"],
    "warmGray": ["#f5f5f4", "#e7e5e4", "#d6d3d1", "#a8a29e", "#78716c", "#57534e", "#44403c", "#292524", "#1c1917"],
    "trueGray": ["#f5f5f5", "#e5e5e5", "#d4d4d4", "#a3a3a3", "#737373", "#525252", "#404040", "#262626", "#171717"],
    "gray": ["#f4f4f5", "#e4e4e7", "#d4d4d8", "#a1a1aa", "#71717a", "#52525b", "#3f3f46", "#27272a", "#18181b"],
    "coolGray": ["#f3f4f6", "#e5e7eb", "#d1d5db", "#9ca3af", "#6b7280", "#4b5563", "#374151", "#1f2937", "#111827"],
    "blueGray": ["#f1f5f9", "#e2e8f0", "#cbd5e1", "#94a3b8", "#64748b", "#475569", "#334155", "#1e293b", "#0f172a"],
}


class Colors:
    def __init__(self) -> None:
        self.aliases: dict[str, str] = {
            "b": "blue",
            "r": "red",
            "g": "green",
            "k": "gray",
            "f": "gray",
            "m": "rose",
            "c": "cyan",
            "o": "orange",
            "y": "yellow",
            "a": "amber",
            "p": "purple",
            "e": "grey",
            "lg": "lime",
            "lb": "sky",
            "black": "gray",
            "white": "gray",
        }
        self.default_colors: tuple[str, ...] = (
            "blue",
            "emerald",
            "red",
            "purple",
            "amber",
            "grey",
            "cyan",
            "teal",
            "green",
            "orange",
            "indigo",
        )

    def next_colour(self) -> Generator[str, None, None]:
        """A generator to return a sequence of colors"""
        for index in [4, 7, 2]:
            for color in self.default_colors:
                yield ALL_COLOURS[color][index]

    def format(self, color: ColorInput | None) -> str:
        if color is None:
            return next(iter(self.next_colour()))
        elif isinstance(color, np.ndarray | list):
            color = rgb2hex(color)  # type: ignore
        if color[0] == "#":
            return color
        elif color in ALL_COLOURS:
            return ALL_COLOURS[color][4]
        elif color in self.aliases:
            alias = self.aliases[color]
            index = 4
            if color.lower() in ["k", "black"]:
                index = -1
            elif color.lower() in ["f", "white"]:
                index = 0
            return ALL_COLOURS[alias][index]
        else:
            raise ValueError(f"Color {color} is not mapped. Please give a hex code")

    def get_formatted(self, list_colors: Iterable[ColorInput]) -> list[str]:
        return [self.format(c) for c in list_colors]

    def get_default(self) -> list[str]:
        return self.get_formatted(self.default_colors)

    def get_colormap(self, num: int, cmap_name: str, scale: float = 0.7) -> list[str]:  # pragma: no cover
        color_list = self.get_formatted(plt.get_cmap(cmap_name)(np.linspace(0.05, 0.9, num)))
        scales = scale + (1 - scale) * np.abs(1 - np.linspace(0, 2, num))
        scaled = [self.scale_colour(c, s) for c, s in zip(color_list, scales)]
        return scaled

    def scale_colour(self, color: ColorInput, scalefactor: float) -> str:  # pragma: no cover
        hexx = self.format(color).strip("#")
        if scalefactor < 0 or len(hexx) != 6:
            return hexx
        r, g, b = int(hexx[:2], 16), int(hexx[2:4], 16), int(hexx[4:], 16)
        r = self._clamp(int(r * scalefactor))
        g = self._clamp(int(g * scalefactor))
        b = self._clamp(int(b * scalefactor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _clamp(self, val: int, minimum: int = 0, maximum: int = 255) -> int:
        if val < minimum:
            return minimum
        if val > maximum:
            return maximum
        return val


colors = Colors()
