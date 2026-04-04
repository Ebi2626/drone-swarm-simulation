import numpy as np
from scipy.interpolate import splprep, splev

# Zakładam, że klasa TrapezoidalProfile znajduje się w tym samym folderze
from src.trajectory.TrapezoidalProfile import TrapezoidalProfile

class BSplineTrajectory:
    def __init__(self, waypoints: np.ndarray, cruise_speed: float, max_accel: float):
        """
        Inicjalizuje trajektorię B-Spline na podstawie punktów zoptymalizowanych przez NSGA-III.
        
        :param waypoints: Tablica numpy o wymiarach (N, 3) zawierająca punkty trasy.
        :param cruise_speed: Maksymalna (przelotowa) prędkość drona [m/s].
        :param max_accel: Maksymalne dopuszczalne przyspieszenie drona [m/s^2].
        """
        self.waypoints = waypoints
        
        # splprep generuje parametry krzywej dla punktów w 3D.
        # s=0 oznacza, że wymuszamy dokładne przejście krzywej przez waypointy (interpolacja).
        # k=3 to stopień krzywej (cubic - zapewnia ciągłość przyspieszeń).
        self.tck, self.u_params = splprep(
            [waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], 
            s=0, 
            k=3
        )
        
        # Obliczanie przybliżonej całkowitej długości krzywej [m]
        self.arc_length = self._calculate_arc_length()
        
        # Inicjalizacja profilu prędkości z wczorajszej klasy
        self.profile = TrapezoidalProfile(
            total_distance=self.arc_length, 
            cruise_speed=cruise_speed, 
            max_accel=max_accel
        )
        
        # Całkowity czas lotu po tej trajektorii [s]
        self.total_duration = self.profile.total_duration

    def _calculate_arc_length(self, num_samples: int = 1000) -> float:
        """
        Oblicza całkowitą długość krzywej B-Spline poprzez całkowanie numeryczne
        (sumę odległości euklidesowych między gęsto próbkowanymi punktami).
        """
        u_samples = np.linspace(0, 1.0, num_samples)
        
        # splev dla tablicy 'u' zwraca listę [tablica_x, tablica_y, tablica_z]
        # Transpozycja (.T) zamienia to na wygodną macierz (num_samples, 3)
        points = np.array(splev(u_samples, self.tck)).T
        
        # Różnice między kolejnymi punktami (wektory przesunięć)
        diffs = np.diff(points, axis=0)
        
        # Suma długości wektorów (norm euklidesowych)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def get_state_at_time(self, t_flight: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Pobiera docelową pozycję i wektor prędkości dla danego czasu lotu.
        Wykorzystywane bezpośrednio przez kontroler PID w każdym kroku symulacji.
        
        :param t_flight: Aktualny czas trwania misji (po odjęciu czasu hover).
        :return: Tuple (target_pos, target_vel) jako wektory numpy (3,).
        """
        # 1. Pobierz przebyty dystans i aktualną skalarną prędkość z profilu trapezowego
        current_distance, current_speed = self.profile.get_state(t_flight)
        
        # 2. Zmapuj przebyty dystans na parametr krzywej u w przedziale [0, 1]
        if self.arc_length <= 1e-6:
            u = 1.0
        else:
            u = np.clip(current_distance / self.arc_length, 0.0, 1.0)
        
        # 3. Wylicz docelową pozycję 3D (x, y, z) na krzywej
        # splev dla skalara 'u' zwraca listę [x, y, z], rzutujemy na wektor numpy
        pos = np.array(splev(u, self.tck))
        
        # Zatrzymanie w punkcie docelowym (lub brak ruchu)
        if u >= 1.0 or current_speed <= 1e-6:
            return pos, np.zeros(3)
        
        # 4. Wylicz wektor kierunkowy (styczną) za pomocą pierwszej pochodnej krzywej (der=1)
        derivative = np.array(splev(u, self.tck, der=1))
        norm = np.linalg.norm(derivative)
        
        # 5. Normalizacja wektora kierunkowego i skalowanie przez zadaną prędkość
        if norm < 1e-6:
            target_vel = np.zeros(3)
        else:
            unit_dir = derivative / norm
            target_vel = unit_dir * current_speed 
            
        return pos, target_vel