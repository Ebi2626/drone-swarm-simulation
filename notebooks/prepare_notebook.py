import sys
import os

# Pobieramy ścieżkę do folderu, w którym jest notebook
current_dir = os.getcwd()

# Pobieramy ścieżkę do folderu nadrzędnego (główny katalog projektu)
project_root = os.path.dirname(current_dir)

# Dodajemy go do sys.path, jeśli jeszcze go tam nie ma
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Dodano do ścieżki: {project_root}")