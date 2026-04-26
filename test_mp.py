import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

PROJECT_ROOT = Path("/Users/songtaowang/Documents/MTP_code")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.model.BPC.CG import MasterProblem, PatternColumn

# Print out MasterProblem source
import inspect
print(inspect.getsource(MasterProblem))
