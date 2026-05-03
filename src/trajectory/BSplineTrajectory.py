import numpy as np
from scipy.interpolate import splprep, splev

from src.trajectory.TrapezoidalProfile import TrapezoidalProfile
from src.trajectory.ConstantSpeedProfile import ConstantSpeedProfile
from src.trajectory.CruiseDecelProfile import CruiseDecelProfile

class BSplineTrajectory:
    def __init__(
        self,
        waypoints: np.ndarray,
        cruise_speed: float,
        max_accel: float,
        constant_speed: bool = False,
        decel_at_end: bool = False,
    ):
        """
        Inicjalizuje trajektorię B-Spline na podstawie punktów zoptymalizowanych przez NSGA-III
        lub punktów z lokalnego planera A*.

        :param waypoints: Tablica numpy o wymiarach (N, 3) zawierająca punkty trasy.
        :param cruise_speed: Maksymalna (przelotowa) prędkość drona [m/s].
        :param max_accel: Maksymalne dopuszczalne przyspieszenie drona [m/s^2].
        :param constant_speed: Jeśli True, używa profilu bez fazy przyspieszania
            (drone wchodzi w trajektorię z v ≈ cruise_speed). Stosowane dla
            lokalnych manewrów uniku — trapezoidalny profil z v(0)=0 powodowałby
            gwałtowne hamowanie na starcie.
        :param decel_at_end: Modyfikator do `constant_speed=True` — gdy True,
            używa `CruiseDecelProfile` (cruise + faza hamowania do v=0 na końcu)
            zamiast `ConstantSpeedProfile` (stała vel + nieciągły step v→0).
            Ramp hamowania eliminuje skok |a|(t) na rejoin pointcie (Bug #2,
            plan.md). Ignorowane gdy `constant_speed=False`.
        """
        self.waypoints = waypoints
        # Ekspozycja parametrów dla downstream (np. `WeightedSumFitness._curvature_cost`
        # liczy κ_admissible = max_accel / cruise²). Bez tych atrybutów `getattr`
        # downstream zwraca domyślne 1.0/2.0, co psuje skalowanie kosztu.
        self.cruise_speed = float(cruise_speed)
        self.max_accel = float(max_accel)

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

        if constant_speed and decel_at_end:
            # Kinematic safety (Bug #2 plan, Krok 5): klamruje cruise_speed do
            # `sqrt(0.5 · max_accel / κ_max)` żeby na cruise lateral_accel = v²·κ
            # ≤ 0.5·max_accel; dec rate dostosowany żeby total ≤ max_accel.
            safe_cruise, safe_decel = self._kinematic_safe_profile_params(
                self.tck, requested_cruise=cruise_speed, max_accel=max_accel
            )
            self.kinematic_clamp = {
                "requested_cruise": float(cruise_speed),
                "applied_cruise": float(safe_cruise),
                "applied_decel": float(safe_decel),
            }
            self.profile = CruiseDecelProfile(
                total_distance=self.arc_length,
                cruise_speed=safe_cruise,
                max_accel=safe_decel,
            )
        elif constant_speed:
            self.profile = ConstantSpeedProfile(
                total_distance=self.arc_length,
                speed=cruise_speed,
            )
        else:
            self.profile = TrapezoidalProfile(
                total_distance=self.arc_length,
                cruise_speed=cruise_speed,
                max_accel=max_accel,
            )

        # Całkowity czas lotu po tej trajektorii [s]
        self.total_duration = self.profile.total_duration

    @staticmethod
    def _kinematic_safe_profile_params(
        tck,
        requested_cruise: float,
        max_accel: float,
        curvature_safety_factor: float = 0.5,
        n_samples: int = 100,
    ) -> tuple[float, float]:
        """Zwraca (safe_cruise, effective_decel) zachowujące |a_total| ≤ max_accel.

        Krzywizna 3D: κ = ‖r' × r''‖ / ‖r'‖³ (niezależna od parametryzacji).
        Akceleracja na krzywej dla v(t):
            a_lateral = v² · κ (centripetal)
            a_longitudinal = v' (tangencjalna, fixed przez profil)
            |a_total| = sqrt(a_lat² + a_long²) ≤ max_accel

        Strategia: budżetujemy `S = curvature_safety_factor` na lateral.
            v_safe² · κ_max ≤ S · max_accel  ⇒  v_safe = sqrt(S · max_accel / κ_max)
            actual_lateral = final_cruise² · κ_max (≤ S · max_accel)
            effective_decel = sqrt(max_accel² - actual_lateral²)

        Gdy `requested_cruise` < `v_safe` (krzywa wystarczająco prosta), nie
        klamrujemy — w `effective_decel` używamy actual_lateral z requested_cruise.
        """
        if tck is None:
            return float(requested_cruise), float(max_accel)
        u = np.linspace(0.05, 0.95, n_samples)
        d1 = np.array(splev(u, tck, der=1)).T  # (n, 3)
        d2 = np.array(splev(u, tck, der=2)).T
        cross = np.cross(d1, d2)
        cross_norm = np.linalg.norm(cross, axis=1)
        d1_norm = np.linalg.norm(d1, axis=1)
        kappa = cross_norm / (d1_norm ** 3 + 1e-12)
        kappa_max = float(np.max(kappa))
        if kappa_max < 1e-9:
            # Krzywa w zasadzie prosta — pełen budżet na decel.
            return float(requested_cruise), float(max_accel)

        v_safe = float(np.sqrt(curvature_safety_factor * max_accel / kappa_max))
        final_cruise = min(float(requested_cruise), v_safe)

        actual_lateral = final_cruise * final_cruise * kappa_max
        effective_decel_sq = max_accel ** 2 - actual_lateral ** 2
        effective_decel = float(np.sqrt(max(1e-6, effective_decel_sq)))
        return final_cruise, effective_decel

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
    
    def get_state_at_distance(self, distance: float, target_speed: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Pobiera docelową pozycję i wektor prędkości na podstawie przebytej drogi (s).
        Umożliwia śledzenie ścieżki (Path Following) z pominięciem profilu czasu.
        """
        if self.arc_length <= 1e-6:
            u = 1.0
        else:
            u = np.clip(distance / self.arc_length, 0.0, 1.0)
        
        pos = np.array(splev(u, self.tck))
        
        # Jeśli jesteśmy na końcu krzywej, zatrzymajmy drona
        if u >= 1.0:
            return pos, np.zeros(3)
        
        # Pierwsza pochodna by pozyskać wektor styczny do ścieżki
        derivative = np.array(splev(u, self.tck, der=1))
        norm = np.linalg.norm(derivative)
        
        if norm < 1e-6:
            target_vel = np.zeros(3)
        else:
            unit_dir = derivative / norm
            # Wektor styczny pomnożony przez stałą zadaną prędkość!
            target_vel = unit_dir * target_speed 
            
        return pos, target_vel