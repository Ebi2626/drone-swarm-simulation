from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gymnasium import spaces # Używamy gymnasium (nowy standard)
import numpy as np

class SwarmBaseWorld(BaseAviary):
    """
    Klasa pośrednia implementująca metody wymagane przez Gym,
    których nie używamy bezpośrednio w algorytmach ewolucyjnych.
    """
    def __init__(self, **kwargs):
        # Żadnego p.connect tutaj! 
        # BaseAviary zrobi to samo, ale użyje naszego spatchowanego p.connect
        super().__init__(**kwargs)
    
    def _actionSpace(self):
        """
        Definiuje przestrzeń akcji (RPM silników).
        Dla 4 silników na drona.
        """
        # Zakładamy 4 silniki na drona
        act_lower_bound = np.array([[0., 0., 0., 0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """
        Definiuje przestrzeń obserwacji (Stan drona).
        Zwracamy Box o kształcie stanu drona (Kinematics).
        """
        # Stan drona w PyBullet Drones to zazwyczaj 20 elementów (poz, quat, rpy, vel, ang_vel, last_clipped_action)
        # Dla algorytmów ewolucyjnych pobieramy stan bezpośrednio, więc to tylko formalność.
        # Używamy np.inf, bo nie ograniczamy pozycji.
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DRONES, 20), dtype=np.float32)

    def _computeObs(self):
        """
        Zwraca aktualny stan obserwacji.
        """
        return self._getDroneStateVector(0) # Zwraca stan dla wszystkich dronów
        
    def _preprocessAction(self, action):
        """
        Opcjonalnie: Przetwarzanie akcji przed wysłaniem do silników.
        BaseAviary tego wymaga w kroku step().
        """
        return action
    
    def _computeReward(self):
        """
        Wymagana przez BaseAviary.
        Dla algorytmów ewolucyjnych fitness liczymy poza środowiskiem,
        więc tutaj zwracamy 0 lub -1.
        """
        return -1.0

    def _computeTerminated(self):
        """
        Warunek zakończenia epizodu (sukces/porażka).
        Zwracamy False, bo sterujemy długością lotu z zewnątrz (np. w verify_env).
        """
        return False

    def _computeTruncated(self):
        """
        Warunek przerwania (np. limit czasu/kroków).
        Zwracamy False.
        """
        return False

    def _computeInfo(self):
        """
        Dodatkowe informacje (słownik).
        Wymagane przez niektóre wersje Gym, choć BaseAviary może mieć domyślną.
        Warto nadpisać dla pewności.
        """
        return {"answer": 42} # Placeholder
