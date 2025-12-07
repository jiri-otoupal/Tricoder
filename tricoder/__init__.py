"""TriVector Code Intelligence - Multi-view code relationship model.

Copyright (c) 2024 Jiri Otoupal
Licensed under Non-Commercial License. Commercial use requires a license.
See LICENSE file for details.
"""
from .model import SymbolModel
from .train import train_model

try:
    from .__about__ import __version__, __author__, __email__, __license__, __copyright__
except ImportError:
    __version__ = "1.1.4"
    __author__ = "Jiri Otoupal"
    __email__ = "j.f.otoupal@gmail.com"
    __license__ = "Non-Commercial License"
    __copyright__ = "Copyright (c) 2024 Jiri Otoupal"

__all__ = ['SymbolModel', 'train_model']
