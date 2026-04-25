# Praca z agentem 

Mając skonfigurowany CLAUDE.md oraz szablon plan.md, Twój cykl pracy z agentem (workflow) powinien teraz wyglądać następująco:

- Faza 1: Inicjalizacja zadania
Zamiast pisać zawiłe prompty w konsoli, wpisujesz swój cel do pliku plan.md w sekcji ## 📌 Current Objective.

- Faza 2: Planowanie przez agenta
Uruchamiasz Claude Code i wydajesz krótką komendę:
`"Przeczytaj plan.md. Zbadaj repozytorium (pamiętaj o odpowiednim pliku koncepcyjnym z głównego folderu), a następnie zaktualizuj sekcję Task Breakdown w plan.md o techniczne kroki, które zamierzasz podjąć. Nie pisz jeszcze kodu."`

- Faza 3: Akceptacja i wykonanie
Przeglądasz zaktualizowany plan.md. Jeśli logika algorytmiczna lub architektoniczna jest poprawna, piszesz do Claude:
`"Plan wygląda dobrze. Zrealizuj zadanie krok po kroku, aktualizując checkboxy w plan.md po każdym sukcesie. Jeśli utkniesz na błędzie powyżej 2 prób, zatrzymaj się i opisz problem w Engineering Notes."`

- Faza 4: Sprzątanie kontekstu
Po zakończeniu zadania i zatwierdzeniu zmian (commit), wpisujesz w konsoli /clear, aby wyczyścić pamięć agenta przed nowym, niezwiązanym zadaniem. Pamięć o projekcie i tak pozostaje bezpieczna w CLAUDE.md i plikach dokumentacji.