__version__ = "1.0.0"
__author__ = "Seu Nome"
__email__ = "seu.email@example.com"

import os
import sys
from pathlib import Path

def setup_project_paths():
    """Configura os paths do projeto automaticamente"""
    project_root = Path(__file__).parent.parent.absolute()
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    return project_root, src_path

PROJECT_ROOT, SRC_PATH = setup_project_paths()