import numpy as np
from numpy.typing import NDArray
from pymoo.core.sampling import Sampling

class StraightLineNoiseSampling(Sampling):
    """Inicjalizacja populacji jako interpolacja linii prostej start↔cel +
    Gaussian noise. Współdzielona przez NSGA-III, OOA, SSA i MSFFOA dla
    zachowania ceteris paribus w porównaniu metaheurystyk.
    """
    def __init__(self,
                 start_pos: NDArray,
                 target_pos: NDArray,
                 n_inner_points: int,
                 n_drones: int,
                 noise_std: float = 2.0,
                 noise_std_z: float | None = None,
                 rng: np.random.Generator | int | None = None
                ):
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
        self.rng = np.random.default_rng(rng)
        self.start = start_pos
        self.target = target_pos
        self.n_inner_points = n_inner_points
        self.n_drones = n_drones
        self.noise_std = noise_std
        self.noise_std_z = float(noise_std_z) if noise_std_z is not None else float(noise_std)

    def _do(self, problem, n_samples: int, **kwargs) -> NDArray:
        t_vals = np.linspace(0, 1, self.n_inner_points + 2)[1:-1]
        t = t_vals.reshape(1, 1, self.n_inner_points, 1)

        s = self.start[None, :, None, :]
        e = self.target[None, :, None, :]

        points = s + t * (e - s)
        X = np.tile(points, (n_samples, 1, 1, 1))

        # Anizotropowy szum: σ_z może być mniejszy gdy zakres wysokości startu
        # i celu jest istotnie mniejszy niż zakres poziomy — szum izotropowy
        # spłaszczyłby populację Z na dolny limit po klipowaniu.
        noise_scale = np.ones_like(X)
        noise_scale[..., 0] *= self.noise_std
        noise_scale[..., 1] *= self.noise_std
        noise_scale[..., 2] *= self.noise_std_z
        noise = self.rng.normal(loc=0.0, scale=1.0, size=X.shape) * noise_scale
        X_noisy = X + noise

        X_flat = X_noisy.reshape(n_samples, -1)

        if problem.xl is not None and problem.xu is not None:
            X_flat = np.clip(X_flat, problem.xl, problem.xu)

        return X_flat