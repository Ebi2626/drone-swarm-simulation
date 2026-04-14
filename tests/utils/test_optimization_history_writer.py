import os
import shutil
import time
import numpy as np
import h5py

# Zakładam, że klasa znajduje się w src/utils/optimization_history_writer.py
from src.utils.optimization_history_writer import OptimizationHistoryWriter

def test_optimization_history_writer():
    output_dir = "test_output_history"
    
    # Czyszczenie katalogu przed testem
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    print("Inicjalizacja writera...")
    writer = OptimizationHistoryWriter(output_dir=output_dir)
    
    num_generations = 250
    pop_size = 60      # 60 dronów/osobników w roju
    num_objectives = 3 # Np. długość trasy, błąd wysokości, kolizje
    num_decisions = 10 # Np. parametry krzywej B-sklejanej
    
    print(f"Symulacja {num_generations} generacji NSGA-III...")
    start_time = time.time()
    
    for gen in range(num_generations):
        # Symulacja ewaluacji roju: generujemy losowe macierze dla danej generacji
        gen_objectives = np.random.rand(pop_size, num_objectives)
        gen_decisions = np.random.rand(pop_size, num_decisions)
        
        # Tworzymy słownik odpowiadający stanowi populacji w danej iteracji
        data_packet = {
            "objectives": gen_objectives,
            "decisions": gen_decisions
        }
        
        # Wrzucenie do kolejki (nie blokuje głównego wątku)
        writer.put_generation_data(data_packet)
        
        # Symulacja drobnego opóźnienia obliczeniowego algorytmu (np. NSGA-III sortowania)
        time.sleep(0.005) 

    # Zakończenie pracy eksperymentu
    print("Zamykanie writera (oczekiwanie na opróżnienie bufora)...")
    writer.close()
    print(f"Czas działania symulacji logowania: {time.time() - start_time:.2f} s")
    
    # ---------------------------------------------------------
    # WERYFIKACJA OFFLINE (Odczyt Danych)
    # ---------------------------------------------------------
    hdf5_path = os.path.join(output_dir, "optimization_history.h5")
    
    assert os.path.exists(hdf5_path), "Plik HDF5 nie został utworzony!"
    
    print("\n--- WERYFIKACJA ZAPISU ---")
    with h5py.File(hdf5_path, "r") as f:
        print("Dostępne datasety w pliku:", list(f.keys()))
        
        obj_ds = f["objectives"]
        dec_ds = f["decisions"]
        
        print(f"Kształt macierzy 'objectives': {obj_ds.shape}")
        print(f"Kształt macierzy 'decisions':  {dec_ds.shape}")
        
        # Asercje weryfikujące zgodność z rygorem naukowym (prawidłowe wymiary)
        assert obj_ds.shape == (num_generations, pop_size, num_objectives), \
            "Błąd: Zły kształt macierzy celów! Prawdopodobnie użyto np.concatenate zamiast np.stack."
        assert dec_ds.shape == (num_generations, pop_size, num_decisions), \
            "Błąd: Zły kształt macierzy decyzji!"
            
        print("Wszystkie testy zaliczone pomyślnie. Dane są gotowe do analizy statystycznej!")

    # ---------------------------------------------------------
    # SPRZĄTANIE
    # ---------------------------------------------------------
    shutil.rmtree(output_dir)
    print(f"Usunięto katalog testowy: {output_dir}")

if __name__ == "__main__":
    test_optimization_history_writer()