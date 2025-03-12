import importlib.metadata as importlib_metadata

from .analysis import Bound
from .chain import Chain, ChainConfig
from .chainconsumer import ChainConsumer
from .color_finder import colors
from .examples import make_sample
from .plotting.config import PlotConfig
from .truth import Truth

__all__ = ["Bound", "Chain", "ChainConfig", "ChainConsumer", "PlotConfig", "Truth", "colors", "make_sample"]

__version__ = importlib_metadata.version(__name__)
