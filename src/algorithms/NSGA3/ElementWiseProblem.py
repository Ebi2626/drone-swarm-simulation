import numpy as np
from pymoo.core.problem import Problem  # ZMIANA 1: Dziedziczymy po Problem, nie ElementwiseProblem

class UAVSwarmPathPlanningProblem(Problem):
    def __init__(self, 
                 space_limits: list, 
                 n_drones: int, 
                 n_waypoints: int, 
                 start_positions: list, 
                 end_positions: list, 
                 obstacles: np.ndarray, 
                 **kwargs):
        
        self.n_drones = n_drones
        self.n_waypoints = n_waypoints
        
        # ZMIANA 2: Konwersja na float32 dla oszczędności pamięci (opcjonalne, ale zalecane)
        self.start_positions = np.array(start_positions, dtype=np.float64)
        self.end_positions = np.array(end_positions, dtype=np.float64)
        self.obstacles = obstacles
        
        # Pre-obliczenie l_max (jako wektor dla każdego drona)
        dist_start_end = np.linalg.norm(self.start_positions - self.end_positions, axis=1)
        self.l_max = 3.0 * dist_start_end  # Shape: (n_drones,)

        # Liczba zmiennych decyzyjnych
        n_var = n_drones * n_waypoints * 3 

        max_bounds = np.array(space_limits)
        min_bounds = np.array([0, 0, 0])
        
        # Upper i Lower bounds dla zmiennych
        self.xu = np.tile(max_bounds, n_drones * n_waypoints)
        self.xl = np.tile(min_bounds, n_drones * n_waypoints)

        # Inicjalizacja klasy bazowej Problem
        # WAŻNE: n_ieq_constr to suma ograniczeń. U Ciebie:
        # G1 (bateria): n_drones
        # G2 (wysokość): n_drones
        # G3 (koordynacja): n_drones
        # Razem: 3 * n_drones
        super().__init__(n_var=n_var, 
                         n_obj=3, 
                         n_ieq_constr=n_drones * 3, 
                         xl=self.xl, 
                         xu=self.xu,
                         **kwargs)

    # --------------------------------------------------------------------------
    # METODY WEKTOROWE (Działają na batchu populacji)
    # X ma kształt (N_pop, n_var) - to jest KLUCZOWA zmiana
    # --------------------------------------------------------------------------

    def _reshape_population(self, x):
        """
        Zmienia płaski wektor zmiennych w tensor trajektorii.
        Dodaje punkty Start i End do każdego osobnika.
        
        Input x: (N_pop, n_drones * n_waypoints * 3)
        Output trajs: (N_pop, n_drones, n_waypoints + 2, 3)
        """
        n_pop = x.shape[0]
        
        # 1. Reshape zmiennych decyzyjnych (punkty pośrednie)
        # (N_pop, n_drones, n_waypoints, 3)
        intermediate = x.reshape(n_pop, self.n_drones, self.n_waypoints, 3)
        
        # 2. Przygotowanie Start i End do broadcastingu
        # Start: (1, n_drones, 1, 3) -> broadcastuje się na N_pop
        starts = self.start_positions[None, :, None, :] 
        ends = self.end_positions[None, :, None, :]
        
        # Powielenie start/end dla każdego osobnika w populacji
        # To jest "tanie" w pamięci dzięki widokom, ale np.concatenate wymaga kopii.
        starts_pop = np.tile(starts, (n_pop, 1, 1, 1))
        ends_pop = np.tile(ends, (n_pop, 1, 1, 1))
        
        # 3. Sklejenie w pełne trajektorie
        # Wynik: (N_pop, n_drones, n_waypoints + 2, 3)
        full_trajectories = np.concatenate([starts_pop, intermediate, ends_pop], axis=2)
        
        return full_trajectories

    def _calc_segment_lengths(self, trajectories):
        """
        Oblicza długości segmentów dla całej populacji naraz.
        Input: (N_pop, n_drones, n_points, 3)
        Output: (N_pop, n_drones) - całkowita długość trasy per dron
        """
        # Różnice między punktami: (N_pop, n_drones, n_points-1, 3)
        diffs = np.diff(trajectories, axis=2)
        # Długości segmentów: (N_pop, n_drones, n_points-1)
        seg_lens = np.linalg.norm(diffs, axis=3)
        # Suma długości trasy: (N_pop, n_drones)
        return np.sum(seg_lens, axis=2)

    # --- FUNKCJE KOSZTU (OBJ) ---

    def _obj_trajectory_len(self, total_lengths):
        """
        F1: Całkowita długość tras roju (suma długości wszystkich dronów).
        Input: (N_pop, n_drones)
        Output: (N_pop,)
        """
        return np.sum(total_lengths, axis=1)

    def _obj_threat(self, trajectories):
        """
        F2: Kolizje z przeszkodami (Z interpolacją i MARGINESEM BEZPIECZEŃSTWA).
        Input: (N_pop, n_drones, n_points, 3)
        """
        n_pop = trajectories.shape[0]
        
        # --- KONFIGURACJA BEZPIECZEŃSTWA ---
        # Margines: promień drona + błąd GPS + zapas na wiatr
        # Np. 2.0 metry dodatkowego odstępu od każdego drzewa
        SAFETY_MARGIN = 2.0 
        
        # 1. ZAGĘSZCZANIE (Bez zmian)
        P0 = trajectories[:, :, :-1, :]
        P1 = trajectories[:, :, 1:, :]
        num_samples = 10 
        t_values = np.linspace(0, 1, num_samples)
        
        P0_exp = P0[:, :, :, None, :]
        P1_exp = P1[:, :, :, None, :]
        t_exp = t_values[None, None, None, :, None]
        
        dense_points = P0_exp + (P1_exp - P0_exp) * t_exp
        
        # 2. SPŁASZCZANIE (Bez zmian)
        points_flat = dense_points.reshape(-1, 3)
        points_xy = points_flat[:, :2]
        points_z = points_flat[:, 2]
        
        tree_xy = self.obstacles[:, :2]
        tree_r = self.obstacles[:, 2]
        tree_h = self.obstacles[:, 3]
        
        # 3. OBLICZENIA FIZYCZNE (Zmienione)
        dists = np.linalg.norm(points_xy[:, None, :] - tree_xy[None, :, :], axis=2)
        
        # TU JEST ZMIANA: Efektywny promień to r_drzewa + margines
        effective_radius = tree_r[None, :] + SAFETY_MARGIN
        
        # Wykrywanie kolizji z "nadmuchanym" drzewem
        col_xy = dists < effective_radius
        col_z = points_z[:, None] < tree_h[None, :]
        mask = col_xy & col_z
        
        pen = np.zeros_like(dists)
        radii_broadcasted = np.broadcast_to(effective_radius, dists.shape)
        
        if np.any(mask):
            # Kara jest liczona od granicy marginesu w głąb
            # Dodajemy też stałą karę (+1000) za sam fakt dotknięcia strefy buforowej.
            # To zniechęca algorytm do "ślizgania się" po granicy marginesu.
            penetration_depth = radii_broadcasted[mask] - dists[mask]
            pen[mask] = (penetration_depth * 10.0) + 1000.0
            
        # 4. AGREGACJA
        total_pen_per_point = np.sum(pen, axis=1)
        total_pen_per_ind = total_pen_per_point.reshape(n_pop, -1).sum(axis=1)
        
        return total_pen_per_ind

    def _obj_height(self, trajectories, h_pref=20.0):
        """
        F3: Odchylenie od wysokości.
        Input: (N_pop, n_drones, n_points, 3)
        Output: (N_pop,)
        """
        z_coords = trajectories[:, :, :, 2] # (N_pop, n_drones, n_points)
        
        # Składnik 1: Odchylenie kwadratowe
        cost_dev = np.sum((z_coords - h_pref)**2, axis=(1, 2))
        
        # Składnik 2: Pod ziemią (z < 0)
        violation = np.maximum(0, -z_coords)
        cost_ground = np.sum(violation, axis=(1, 2)) * 1000.0
        
        return cost_dev + cost_ground

    # --- OGRANICZENIA (CONSTR) ---

    def _constr_battery(self, lengths):
        """
        G1: (N_pop, n_drones). Wartość > 0 to błąd.
        """
        # lengths: (N_pop, n_drones)
        # l_max broadcastuje się z (n_drones,)
        return lengths - self.l_max[None, :]

    def _constr_height_safe(self, trajectories, h_min=5.0):
        """
        G2: (N_pop, n_drones).
        """
        z_coords = trajectories[:, :, :, 2] # (N_pop, n_drones, n_points)
        min_z_per_drone = np.min(z_coords, axis=2) # (N_pop, n_drones)
        return h_min - min_z_per_drone

    def _constr_coordination(self, lengths, tol=50.0):
        """
        G3: (N_pop, n_drones).
        """
        # Średnia roju dla każdego osobnika: (N_pop, 1)
        means = np.mean(lengths, axis=1, keepdims=True)
        devs = np.abs(lengths - means)
        return devs - tol

    # --------------------------------------------------------------------------
    # GLÓWNA METODA EWALUACJI (Pymoo entry point)
    # --------------------------------------------------------------------------
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Metoda wywoływana raz na generację dla całej populacji.
        x: (1000, n_var) - macierz NumPy (nie pojedynczy wektor!)
        """
        
        # 1. Rekonstrukcja trajektorii dla całej populacji
        # Wynik: Tensor (1000, Drony, Punkty, 3)
        # To jest jedyna duża alokacja pamięci w tym kroku.
        trajs = self._reshape_population(x)
        
        # 2. Obliczenia pomocnicze (wspólne dla funkcji celu i ograniczeń)
        lengths = self._calc_segment_lengths(trajs) # (1000, Drony)
        
        # 3. Obliczenie celów (F)
        f1 = self._obj_trajectory_len(lengths)
        f2 = self._obj_threat(trajs)
        f3 = self._obj_height(trajs)
        
        # Zapis do out["F"]. Kształt musi być (N_pop, 3)
        # column_stack łączy wektory 1D w macierz kolumnową
        out["F"] = np.column_stack([f1, f2, f3])
        
        # 4. Obliczenie ograniczeń (G)
        g1 = self._constr_battery(lengths)      # (1000, Drony)
        g2 = self._constr_height_safe(trajs)    # (1000, Drony)
        g3 = self._constr_coordination(lengths) # (1000, Drony)
        
        # Zapis do out["G"]. Kształt musi być (N_pop, n_constr)
        # n_constr = 3 * n_drones
        out["G"] = np.column_stack([g1, g2, g3])
        
        # Usunięcie debugowania typu print() w pętli produkcyjnej jest kluczowe dla wydajności!
        # Jeśli chcesz debugować, sprawdzaj tylko pierwszego osobnika:
        # if kwargs.get("algorithm") and kwargs["algorithm"].n_gen % 10 == 0:
        #     print(f"Gen {kwargs['algorithm'].n_gen}: Best F1 = {np.min(f1)}")
