from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("iss_preprocess")
except PackageNotFoundError:
    # package is not installed
    pass

# Expose main submodules for iss.<submodule> usage
from . import io
from . import call
from . import cli
from . import config
from . import coppafish
from . import diagnostics
from . import image
from . import pipeline
from . import reg
from . import segment
from . import vis
