#!/bin/bash

# Sprawdzenie, czy podano argument
if [ -z "$1" ]; then
    echo "❌ Błąd: Nie podano identyfikatora eksperymentu."
    echo "Użycie: ./run.sh <nazwa_wygenerowanego_pliku_bez_rozszerzenia>"
    echo "Przykład: ./run.sh exp_20260421_a1b2c3d4_complex_test"
    exit 1
fi

EXPERIMENT_ID=$1

# Optymalizacja wielowątkowości C/C++ (kluczowe dla PyBullet i SciPy/NumPy w trybie multirun)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "🚀 Uruchamianie eksperymentu: $EXPERIMENT_ID"
echo "⚙️ Tryb: Równoległy (Multirun), Wątki matematyczne zablokowane na 1 per proces"
echo "-------------------------------------------------------------------"

# Wywołanie Hydry ze wstrzyknięciem eksperymentu i flagą multirun
python main.py +experiment_generated=${EXPERIMENT_ID} -m

# Sprawdzenie kodu wyjścia
if [ $? -eq 0 ]; then
    echo "-------------------------------------------------------------------"
    echo "✅ Eksperyment '$EXPERIMENT_ID' zakończony pomyślnie!"
else
    echo "-------------------------------------------------------------------"
    echo "❌ Eksperyment '$EXPERIMENT_ID' zakończony z błędem."
    exit 1
fi