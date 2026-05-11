import numpy as np
from scipy.interpolate import splprep, splev

from src.trajectory.TrapezoidalProfile import TrapezoidalProfile
from src.trajectory.ConstantSpeedProfile import ConstantSpeedProfile
from src.trajectory.CruiseDecelProfile import CruiseDecelProfile

class BSplineTrajectory:
    """B-Spline cubic interpolacja waypointów + profil prędkości (trapezoid / constant / cruise+decel)."""

    def __init__(
        self,
        waypoints: np.ndarray,
        cruise_speed: float,
        max_accel: float,
        constant_speed: bool = False,
        decel_at_end: bool = False,
    ):
        """Zbuduj `BSplineTrajectory` z waypointów i wybierz profil prędkości.

        Args:
            waypoints: `(N, 3)` punkty trasy.
            cruise_speed: Prędkość przelotowa [m/s].
            max_accel: Maks. dopuszczalne przyspieszenie [m/s²].
            constant_speed: `True` ⇒ profil bez fazy rampy startowej (online evasion).
            decel_at_end: Modyfikator do `constant_speed=True` — `True` ⇒
                `CruiseDecelProfile` (rampa hamowania na końcu); `False` ⇒
                `ConstantSpeedProfile` (skok `v → 0` na końcu).

        Efekty uboczne:
            Buduje `tck` (splprep), liczy `arc_length`, `total_duration`
            i ewentualnie `kinematic_clamp` z bezpiecznym profilem.
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
            # Kinematic safety: klamruje cruise_speed do
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
        """Zwróć `(safe_cruise, effective_decel)` zachowujące `|a_total| ≤ max_accel`.

        Budżetuje `curvature_safety_factor × max_accel` na lateral acc
        (`v² · κ_max`); resztę przeznacza na hamowanie longitudinalne.

        Args:
            tck: `tck` z `splprep`.
            requested_cruise: Żądana prędkość przelotowa [m/s].
            max_accel: Maks. dopuszczalna akceleracja [m/s²].
            curvature_safety_factor: Ułamek `max_accel` zarezerwowany
                na centripetal acc (`(0, 1]`).
            n_samples: Liczba próbek `u ∈ [0.05, 0.95]` do estymacji `κ_max`.

        Returns:
            `(safe_cruise, effective_decel)`; gdy krzywa jest praktycznie
            prosta (`κ_max → 0`), zwraca `(requested_cruise, max_accel)`.
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
        """Wylicz długość krzywej B-Spline jako sumę dystansów `num_samples` próbek."""
        u_samples = np.linspace(0, 1.0, num_samples)
        
        # splev dla tablicy 'u' zwraca listę [tablica_x, tablica_y, tablica_z]
        # Transpozycja (.T) zamienia to na wygodną macierz (num_samples, 3)
        points = np.array(splev(u_samples, self.tck)).T
        
        # Różnice między kolejnymi punktami (wektory przesunięć)
        diffs = np.diff(points, axis=0)
        
        # Suma długości wektorów (norm euklidesowych)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def get_state_at_time(self, t_flight: float) -> tuple[np.ndarray, np.ndarray]:
        """Zwróć `(target_pos, target_vel)` dla zadanego `t_flight` (po odjęciu hover).

        Args:
            t_flight: Czas lotu od startu trajektorii [s].

        Returns:
            Krotka `(pos (3,), vel (3,))`. Po osiągnięciu końca trasy
            `vel = 0`.
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
        """Zwróć `(target_pos, target_vel)` po przebytej drodze `distance` [m].

        Path-Following bez profilu czasowego — używa `target_speed` jako
        skalarnej prędkości. `vel = 0` po dotarciu na koniec trasy.

        Args:
            distance: Przebyta droga [m].
            target_speed: Zadana prędkość skalarne [m/s].

        Returns:
            Krotka `(pos (3,), vel (3,))`.
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