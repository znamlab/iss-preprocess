from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("iss_preprocess")
except PackageNotFoundError:
    # package is not installed
    pass

from . import io
