import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_externals
from carlini.nn_robust_attacks import l2_attack, li_attack, l0_attack

