"""
PyYel packages initializer
"""

import os

# PyYeL
DIR_PYYEL_PATH = os.path.dirname(os.path.abspath(__file__))

# PyYel/Data
DIR_DATA_PATH = os.path.join(DIR_PYYEL_PATH, 'Data')
DIR_DATA_CONFIGS_PATH = os.path.join(DIR_DATA_PATH, 'Configs')
DIR_DATA_GUIS_PATH = os.path.join(DIR_DATA_PATH, 'GUIs')

# PyYel/Networks
DIR_NETWORKS_PATH = os.path.join(DIR_PYYEL_PATH, 'Networks')
