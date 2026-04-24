import numpy as np
from numpy.typing import NDArray
from pymoo.core.sampling import Sampling

class StraightLineNoiseSampling(Sampling):
    """
    Ustandaryzowana strategia inicjalizacji dla planowania ścieżek UAV.
    Zastępuje HeuristicSampling w celu zapewnienia rzetelnego porównania 
    metaheurystyk w badaniach naukowych.
    
    Generuje osobniki przez interpolację linii prostej między startem a celem,
    a następnie aplikuje szum Gaussa do wszystkich współrzędnych. Gwarantuje to
    utrzymanie różnorodności początkowej populacji niezbędnej dla algorytmów 
    ewolucyjnych i rojowych, zapobiegając przedwczesnej zbieżności.
    """
    def __init__(self,
                 start_pos: NDArray,
                 target_pos: NDArray,
                 n_inner_points: int,
                 n_drones: int,
                 noise_std: float = 2.0,
                 noise_std_z: float | None = None):
        """
        :param start_pos: Pozycje startowe dronów (Kształt: [n_drones, 3])
        :param target_pos: Pozycje docelowe dronów (Kształt: [n_drones, 3])
        :param n_inner_points: Liczba węzłów kontrolnych pomiędzy startem a celem
        :param n_drones: Liczba dronów w roju (wielkość wektora wielo-agentowego)
        :param noise_std: Odchylenie standardowe szumu Gaussa w XY.
        :param noise_std_z: Odchylenie standardowe szumu Gaussa w Z; jeśli None,
                            używamy ``noise_std`` (zachowanie izotropowe, wstecznie
                            kompatybilne). Dla planowania nisko-poziomowego (start
                            i cel w tym samym przedziale wysokości) warto dać
                            mniejszą wartość, by nie rozsypać całej populacji Z
                            o zakres większy niż realna geometria lotu.
        """
        super().__init__()
        self.start = start_pos
        self.target = target_pos
        self.n_inner_points = n_inner_points
        self.n_drones = n_drones
        self.noise_std = noise_std
        self.noise_std_z = float(noise_std_z) if noise_std_z is not None else float(noise_std)

    def _do(self, problem, n_samples: int, **kwargs) -> NDArray:
        # 1. Generowanie równomiernych kroków interpolacji dla punktów wewnętrznych
        # Np. dla 3 punktów wygeneruje: [0.25, 0.50, 0.75]
        t_vals = np.linspace(0, 1, self.n_inner_points + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner_points, 1)
        
        # Przekształcenie startu i celu do mechanizmu rozgłaszania (broadcasting)
        # Kształt: (1, n_drones, 1, 3)
        s = self.start[None, :, None, :]
        e = self.target[None, :, None, :]
        
        # 2. Baza: Wyliczenie idealnych węzłów na linii prostej
        points = s + t * (e - s)
        
        # Klonowanie wyliczonych punktów dla całej populacji
        # Kształt wejściowy populacji: (n_samples, n_drones, n_inner_points, 3)
        X = np.tile(points, (n_samples, 1, 1, 1))
        
        # 3. Aplikacja szumu Gaussa w celu dywersyfikacji populacji.
        # Szum jest anizotropowy: inny σ dla XY, inny dla Z — dla scenariuszy, w
        # których zakres wysokości startu i celu jest znacznie mniejszy niż zakres
        # poziomy, szum izotropowy spłaszczyłby populację Z na dolny limit.
        noise_scale = np.ones_like(X)
        noise_scale[..., 0] *= self.noise_std
        noise_scale[..., 1] *= self.noise_std
        noise_scale[..., 2] *= self.noise_std_z
        noise = np.random.normal(loc=0.0, scale=1.0, size=X.shape) * noise_scale
        X_noisy = X + noise
        
        # 4. Spłaszczenie tensora do postaci oczekiwanej przez Pymoo (n_samples, n_vars)
        X_flat = X_noisy.reshape(n_samples, -1)
        
        # 5. Twarde przycięcie (Clipping) do granic przestrzeni (world bounds)
        # Gwarantuje to, że zmutowane punkty nie opuszczą limitów fizycznych środowiska 
        # (np. nie znajdą się pod ziemią lub poza mapą).
        if problem.xl is not None and problem.xu is not None:
            X_flat = np.clip(X_flat, problem.xl, problem.xu)
            
        return X_flat