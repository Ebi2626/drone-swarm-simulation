"""Dodaje katalog projektu do `sys.path` aby notebooki w `notebooks/` mogły
importować moduły z `src/` bez instalacji pakietu."""
import sys
import os

current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Dodano do ścieżki: {project_root}")