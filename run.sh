#!/bin/bash

# Sprawdzenie, czy podano argument
if [ -z "$1" ]; then
    echo "❌ Błąd: Nie podano identyfikatora eksperymentu."
    echo "Użycie: ./run.sh <exp_id> [--n-jobs N] [--override key=val ...]"
    echo "Przykład: ./run.sh exp_20260421_a1b2c3d4_complex_test --n-jobs 6"
    exit 1
fi

EXPERIMENT_ID=$1
shift  # pozostałe argumenty trafią do run_subprocess.py jako "$@"

# Optymalizacja wielowątkowości C/C++ (kluczowe dla PyBullet i SciPy/NumPy)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "🚀 Uruchamianie eksperymentu: $EXPERIMENT_ID"
echo "⚙️ Tryb: subprocess-per-job (workaround dla H1''; plan.md, Krok 5c)"
echo "    Każde zadanie w fresh procesie 'python main.py' — eliminuje"
echo "    akumulację stanu PyBullet/Numba między uruchomieniami,"
echo "    przyczynę patologii 'drony stoją w PyBullet'."
echo "-------------------------------------------------------------------"

# Wywołanie subprocess-runnera (zastępuje Hydra-multirun + joblib)
python experiments/run_subprocess.py $EXPERIMENT_ID "$@"

# Sprawdzenie kodu wyjścia
if [ $? -eq 0 ]; then
    echo "-------------------------------------------------------------------"
    echo "✅ Eksperyment '$EXPERIMENT_ID' zakończony pomyślnie!"
else
    echo "-------------------------------------------------------------------"
    echo "❌ Eksperyment '$EXPERIMENT_ID' zakończony z błędem."
    exit 1
fi
