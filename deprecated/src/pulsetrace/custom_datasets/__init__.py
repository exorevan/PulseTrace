import glob
import importlib
import os

from .default import DefaultDataset

# Gather all .py files in this directory (excluding __init__.py)
modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
__all__ = ["DefaultDataset"]

for path in modules:
    filename = os.path.basename(path)
    module_name, ext = os.path.splitext(filename)

    # Skip non-Python files and __init__.py
    if ext.lower() != ".py" or module_name == "__init__":
        continue

    # Import the module
    imported_module = importlib.import_module(f".{module_name}", package=__name__)

    # Optionally copy symbols into the package namespace
    for attr_name in dir(imported_module):
        if not attr_name.startswith("_"):  # ignore special/__ attributes
            globals()[attr_name] = getattr(imported_module, attr_name)
            __all__.append(attr_name)
