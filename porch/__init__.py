from .argument import *
from .base import *
from .debug import *
from .loss import *
from porch.models.mlp import *
from porch.models.cnn import *
from .models import *
from .modules import *
from .regularizer import *
from .optim import *

from . import _functions


"""
try:
	from .lenetA import *
	#from .mlpA import *
	pass
except ImportError:
	raise ImportError("Could not load some of the XModules...")
	pass
"""