from ._bootstrap_utf8 import install as _install_utf8

_install_utf8()

from .conversion import *
from .g2p_yoruba import convert as yoruba_to_ipa
from .english2ipa import *
