#!/bin/bash

# Domyślny wzorzec to "main.py", ale możesz go nadpisać podając argument
# np.: ./wait_for_python_shutdown.sh "moj_eksperyment.py"
SCRIPT_NAME=${1:-"main.py"}

echo "Rozpoczęto monitorowanie procesów pasujących do wzorca: $SCRIPT_NAME"

while :; do
    # Zliczamy procesy pasujące do wzorca
    PROCESS_COUNT=$(pgrep -f "$SCRIPT_NAME" | wc -l)
    
    if [ "$PROCESS_COUNT" -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Wszystkie procesy eksperymentu zakończone!"
        break
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Eksperyment nadal trwa. Liczba aktywnych procesów: $PROCESS_COUNT"
    fi
    
    # Czekamy 10 sekund przed kolejnym sprawdzeniem
    sleep 60
done

echo "Wyłączanie systemu..."
systemctl poweroff