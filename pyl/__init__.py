"""
PyYel packages initializer
"""

__all__ = [
    "data",
    "networks"
]

import os

if not os.path.exists(os.path.join("", "temp")):
    os.mkdir("temp")
    print("PyYel >> temp folder created under:", os.path.abspath(""))