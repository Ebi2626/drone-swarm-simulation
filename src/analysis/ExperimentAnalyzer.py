import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentAnalyzer:
    def __init__(self, metrics_file: str | Path, output_dir: str | Path):
        self.metrics_file = Path(metrics_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku: {self.metrics_file}")
            
        self.df = pd.read_parquet(self.metrics_file)
        
        # Ustawienia stylu akademickiego dla wykresów
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 12,
            'figure.dpi': 300
        })

    def plot_path_length(self):
        """Generuje wykres pudełkowy dla średniej długości ścieżki drona."""
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=self.df, 
            x='optimizer', 
            y='mean_path_length_per_drone', 
            hue='avoidance',
            palette='Set2'
        )
        plt.title('Wpływ algorytmu na średnią długość ścieżki w roju')
        plt.ylabel('Średnia długość trajektorii [m]')
        plt.xlabel('Algorytm optymalizacji bazowej')
        plt.legend(title='Algorytm uniku (Avoidance)')
        plt.tight_layout()
        
        out_path = self.output_dir / 'plot_path_length.png'
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Zapisano wykres: {out_path}")

    def plot_swarm_dispersion(self):
        """Generuje wykres rozproszenia roju (odległości między dronami)."""
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=self.df, 
            x='optimizer', 
            y='mean_inter_drone_distance', 
            hue='avoidance',
            palette='pastel'
        )
        plt.title('Spójność roju: Średni dystans między dronami w czasie lotu')
        plt.ylabel('Średni dystans między dronami [m]')
        plt.xlabel('Algorytm optymalizacji bazowej')
        plt.legend(title='Algorytm uniku')
        plt.tight_layout()
        
        out_path = self.output_dir / 'plot_swarm_dispersion.png'
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Zapisano wykres: {out_path}")

    def plot_planning_time(self):
        """Generuje wykres czasu planowania uniku dla różnych strategii."""
        # Filtrujemy eksperymenty, gdzie avoidance != "none"
        df_avoidance = self.df[self.df['avoidance'] != 'none'].copy()
        
        if df_avoidance.empty:
            logger.warning("Brak danych z aktywnym algorytmem uniku do narysowania czasów planowania.")
            return

        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(
            data=df_avoidance, 
            x='avoidance', 
            y='mean_planning_wall_time_s', 
            hue='optimizer',
            palette='muted'
        )
        plt.title('Czas obliczeń algorytmów uniku przeszkód')
        plt.ylabel('Średni czas planowania trajektorii uniku [s]')
        plt.xlabel('Algorytm uniku')
        plt.yscale('log') # Skala logarytmiczna jest często lepsza dla czasów obliczeniowych
        plt.legend(title='Optymalizator globalny')
        plt.tight_layout()
        
        out_path = self.output_dir / 'plot_planning_time.png'
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Zapisano wykres: {out_path}")

    def generate_all_plots(self):
        self.plot_path_length()
        self.plot_swarm_dispersion()
        self.plot_planning_time()

if __name__ == "__main__":
    # Przykład użycia:
    # analyzer = ExperimentAnalyzer(metrics_file="results/master_metrics.parquet", output_dir="results/plots")
    # analyzer.generate_all_plots()
    pass