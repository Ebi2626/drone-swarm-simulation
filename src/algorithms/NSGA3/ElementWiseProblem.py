import numpy as np
from pymoo.core.problem import ElementwiseProblem

def calculate_trajectory_cost(self, trajectories):
    """
    Oblicza całkowitą długość tras wszystkich dronów.
    
    Args:
        trajectories (np.ndarray): Macierz (n_drones, n_points, 3)
    
    Returns:
        float: Całkowita długość trasy roju.
    """
    # Oblicz różnice między punktami (P_i+1 - P_i)
    # diffs ma kształt (n_drones, n_points-1, 3)
    diffs = np.diff(trajectories, axis=1)
    
    # Oblicz długość każdego segmentu (norma euklidesowa wzdłuż osi współrzędnych)
    # segment_lengths ma kształt (n_drones, n_points-1)
    segment_lengths = np.linalg.norm(diffs, axis=2)
    
    # Zsumuj wszystko (dla wszystkich segmentów i wszystkich dronów)
    total_length = np.sum(segment_lengths)
    
    return total_length

def calculate_threat_cost(self, trajectories, obstacles):
    """
    Oblicza karę za kolizję z lasem (600 drzew) przy użyciu wektoryzacji.
    
    Args:
        trajectories: (n_drones, n_points, 3)
        obstacles: (n_obstacles, 4) -> [x, y, radius, height]
    """
    # 1. Przygotowanie danych (Spłaszczenie trajektorii)
    # Zmieniamy kształt z (Drones, Points, 3) na (Total_Points, 3)
    # Żeby sprawdzić każdy punkt niezależnie od tego, do którego drona należy
    points = trajectories.reshape(-1, 3) 
    
    # Rozdzielamy współrzędne punktów
    points_xy = points[:, :2]  # Kształt: (M, 2)
    points_z = points[:, 2]    # Kształt: (M,)
    
    # Rozdzielamy dane drzew
    tree_xy = obstacles[:, :2]     # Kształt: (T, 2)
    tree_radius = obstacles[:, 2]  # Kształt: (T,)
    tree_height = obstacles[:, 3]  # Kształt: (T,)
    
    # 2. BROADCASTING - Magia NumPy
    # Chcemy macierz odległości o wymiarach (Liczba_Punktów, Liczba_Drzew)
    # points_xy[:, None, :] -> (M, 1, 2)
    # tree_xy[None, :, :]   -> (1, T, 2)
    # Wynik różnicy: (M, T, 2)
    diff = points_xy[:, None, :] - tree_xy[None, :, :]
    
    # Obliczamy dystans Euklidesowy 2D (w płaszczyźnie XY)
    # Wynik dist_matrix: (M, T) - odległość każdego punktu do każdego drzewa
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    # 3. Weryfikacja Kolizji
    # Sprawdzamy warunek 1: Czy jesteśmy "w pniu" (w poziomie)?
    # (M, T) < (1, T) -> Broadcasting promienia na wszystkie punkty
    collision_xy = dist_matrix < tree_radius
    
    # Sprawdzamy warunek 2: Czy jesteśmy poniżej korony drzewa (w pionie)?
    # (M, 1) < (1, T) -> Broadcasting wysokości drona na wszystkie drzewa
    collision_z = points_z[:, None] < tree_height
    
    # Kolizja następuje tylko gdy OBA warunki są spełnione
    full_collision_mask = collision_xy & collision_z
    
    # 4. Obliczenie Kary
    # Obliczamy jak głęboko weszliśmy w drzewo (radius - dist)
    penetration = tree_radius - dist_matrix
    
    # Zerujemy tam, gdzie nie ma kolizji
    penetration[~full_collision_mask] = 0.0
    
    # Sumujemy wszystkie naruszenia
    total_penalty = np.sum(penetration)
    
    return total_penalty

def calculate_height_cost(self, trajectories, h_pref=20.0):
    """
    Oblicza odchylenie od preferowanej wysokości i karę za latanie zbyt nisko.
    """
    # Wyciągnij tylko współrzędne Z (wysokości)
    # z_coords ma kształt (n_drones, n_points)
    z_coords = trajectories[:, :, 2]
    
    # 1. Składnik: Odchylenie od H_pref (np. chcemy latać na 50m)
    # Używamy kwadratu różnicy, żeby karać duże odchyłki mocniej
    deviation_cost = np.sum((z_coords - h_pref) ** 2)
    
    # 2. Składnik (opcjonalny): Kara za latanie "pod ziemią" (z < 0)
    # W praktyce 'xl' (dolna granica) powinno to załatwić, ale warto mieć zabezpieczenie
    ground_violation = np.sum(np.maximum(0, -z_coords)) * 1000
    
    return deviation_cost + ground_violation

class UAVSwarmPathPlanning(ElementwiseProblem):
    def __init__(self, 
                 space_limits: list, # limitations of the space [max_x, max_y, max_z]
                 n_drones: int, # number of drones in the swarm
                 n_waypoints: int, # number of waypoints per drone
                 start_positions: list, # starting positions for each drone
                 end_positions: list, # ending positions for each drone
                 obstacles: np.ndarray, # array of obstacles [x, y, radius, height]
                 **kwargs):
        
        self.n_drones = n_drones
        self.n_waypoints = n_waypoints
        self.start_positions = np.array(start_positions)
        self.end_positions = np.array(end_positions)
        self.obstacles = obstacles
        n_var = n_drones * n_waypoints * 3 # Genotype length: 3 coordinates (x,y,z) per waypoint per drone

        max_bounds = np.array(space_limits)
        min_bounds = np.array([0, 0, 0]) # Zakładamy start od 0, można to sparametryzować
        
        self.xu = np.tile(max_bounds, n_drones * n_waypoints) # Upper bounds
        self.xl = np.tile(min_bounds, n_drones * n_waypoints) # Lower bounds

        super().__init__(n_var=n_var, 
                         n_obj=3,          # Three target functions (trajectory, threat, height)
                         n_ieq_constr=2,   # Two kinds of constrains (length, collision)
                         xl=self.xl, 
                         xu=self.xu,
                         **kwargs)
    def _get_path_lengths(self, trajectories):
        """
        Oblicza długość trasy dla każdego drona.
        
        Args:
            trajectories: (n_drones, n_points, 3)
        Returns:
            np.ndarray: Wektor o długości n_drones zawierający dystanse.
        """
        # Oblicz wektory przesunięć między punktami (segmenty)
        segments = np.diff(trajectories, axis=1)
        
        # Oblicz długość euklidesową każdego segmentu
        segment_lengths = np.linalg.norm(segments, axis=2)
        
        # Zsumuj segmenty dla każdego drona
        total_lengths = np.sum(segment_lengths, axis=1)
        
        return total_lengths
    
    def check_battery_constraint(self, trajectories, l_max):
        """
        Ograniczenie G1: Czy długość trasy mieści się w zasięgu baterii?
        
        Args:
            trajectories: (n_drones, n_points, 3)
            l_max: float lub np.array (maksymalny zasięg)
            
        Returns:
            np.ndarray: Wektor naruszeń (wartości > 0 to błąd).
        """
        current_lengths = self._get_path_lengths(trajectories)
        
        # Wzór: Aktualna_Długość - Max_Długość <= 0
        # Jeśli poleciał 1200m, a limit 1000m -> Wynik 200 (Naruszenie)
        g_values = current_lengths - l_max
        
        return g_values
    
    def check_min_height_constraint(self, trajectories, h_min_safe):
        """
        Ograniczenie G2: Czy dron nie leci niżej niż twardy limit bezpieczeństwa?
        
        Args:
            trajectories: (n_drones, n_points, 3)
            h_min_safe: float (np. 5.0 metrów)
            
        Returns:
            np.ndarray: Wektor naruszeń.
        """
        # Pobierz wszystkie współrzędne Z (wysokości)
        # Shape: (n_drones, n_points)
        z_coords = trajectories[:, :, 2]
        
        # Znajdź najniższy punkt w całej trasie dla każdego drona
        min_flight_heights = np.min(z_coords, axis=1)
        
        # Wzór: Limit - Aktualne_Min <= 0
        # Jeśli min. wysokość to 2m, a limit 5m -> Wynik 3 (Naruszenie)
        g_values = h_min_safe - min_flight_heights
        
        return g_values

    def check_coordination_constraint(self, trajectories, tolerance=50.0):
        """
        Ograniczenie G3: Czy różnice w długościach tras są akceptowalne?
        
        Args:
            trajectories: (n_drones, n_points, 3)
            tolerance: float (maksymalna dopuszczalna różnica w metrach od średniej)
            
        Returns:
            np.ndarray: Wektor naruszeń dla każdego drona.
        """
        lengths = self._get_path_lengths(trajectories)
        
        # Oblicz średnią długość trasy w roju
        mean_swarm_length = np.mean(lengths)
        
        # Oblicz odchylenie każdego drona od tej średniej
        deviations = np.abs(lengths - mean_swarm_length)
        
        # Wzór: Odchylenie - Tolerancja <= 0
        # Jeśli odchylenie to 100m, a tolerancja 50m -> Wynik 50 (Naruszenie)
        g_values = deviations - tolerance
        
        return g_values

    def _evaluate(self, x, out, *args, **kwargs):
        # x are only waypoints without start and end points
        intermediate_points = x.reshape((self.n_drones, self.n_waypoints_intermediate, 3))
        
        trajectories = []
        for i in range(self.n_drones):
            # Sklejanie: [Start] + [Punkty Pośrednie z Genotypu] + [Stop]
            start_pt = self.start_positions[i].reshape(1, 3)
            end_pt = self.end_positions[i].reshape(1, 3)
            current_traj = np.vstack([start_pt, intermediate_points[i], end_pt])
            trajectories.append(current_traj)

        trajectories = np.array(trajectories)

        # --- Tutaj następuje wywołanie Twoich funkcji kosztu ---
        f1 = calculate_trajectory_cost(trajectories)
        f2 = calculate_threat_cost(trajectories, self.obstacles)
        f3 = calculate_height_cost(trajectories)

        # Ograniczenia
        g1 = self.check_battery_constraint(trajectories, l_max=self.l_max)
        g2 = self.check_min_height_constraint(trajectories, h_min_safe=2.0)
        g3 = self.check_coordination_constraint(trajectories, tolerance=50.0)
        
        # Placeholder dla wyników
        out["F"] = [f1, f2, f3]
        out["G"] = np.concatenate([g1, g2, g3])      # Wartości przykładowe
