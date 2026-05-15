"""Dodaje katalog projektu do `sys.path` aby notebooki w `notebooks/` mogły
importować moduły z `src/` bez instalacji pakietu.

Atrybut publiczny `project_root` (absolutna ścieżka do korzenia repo) jest
używany przez notebooki do lokalizowania plików konfiguracyjnych
(`configs/`), wyników (`results/`) i załącznika (`appendix/`).
"""
import sys
import os

# Bazujemy na lokalizacji pliku — odporne na zmiany `cwd` notebooka.
project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Dodano do ścieżki: {project_root}")
