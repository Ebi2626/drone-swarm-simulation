#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_cylinder(ax, x, y, z_start, height, radius, color='red', alpha=0.3):
    """Rysuje cylinder (przeszkodę) na osiach 3D."""
    z = np.linspace(z_start, z_start + height, 20)
    theta = np.linspace(0, 2 * np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_grid = radius * np.cos(theta_grid) + x
    y_grid = radius * np.sin(theta_grid) + y
    
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=True)

def set_axes_equal_3d(ax, limits, z_stretch=5.0):
    """
    Ustawia rzeczywiste limity danych ze świata i dopasowuje kształt "pudełka" 3D.
    Parametr z_stretch pozwala sztucznie podwyższyć wizualnie oś Z (np. 5-krotnie), 
    co jest kluczowe dla map bardzo długich (Y=600m), a płaskich (Z=11m).
    """
    x_limits, y_limits, z_limits = limits
    
    # 1. Narzucamy dokładne, twarde limity z pliku world.csv (bez ujemnych wartości!)
    ax.set_xlim3d(x_limits)
    ax.set_ylim3d(y_limits)
    ax.set_zlim3d(z_limits)
    
    # 2. Obliczamy fizyczne rozpiętości
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    
    # 3. Ustawiamy proporcje prostopadłościanu.
    # Używamy z_stretch, żeby wysokość (11m) nie zniknęła na tle długości (600m).
    ax.set_box_aspect((x_range, y_range, z_range * z_stretch))
    
def main():
    parser = argparse.ArgumentParser(description="Wizualizacja 3D trajektorii roju dronów.")
    parser.add_argument("directory", type=str, help="Ścieżka do katalogu z plikami CSV.")
    args = parser.parse_args()

    dir_path = args.directory
    
    # Definicja nazw plików
    traj_file = os.path.join(dir_path, "trajectories.csv")
    world_file = os.path.join(dir_path, "world_boundaries.csv")
    obs_file = os.path.join(dir_path, "generated_obstacles.csv")

    # Sprawdzenie czy pliki istnieją
    for f in [traj_file, world_file, obs_file]:
        if not os.path.isfile(f):
            print(f"Błąd: Nie znaleziono pliku {f}")
            sys.exit(1)

    # 1. Wczytanie wymiarów świata
    df_world = pd.read_csv(world_file)
    df_world.set_index('Axis', inplace=True)
    limits = (
        (df_world.loc['X', 'Min_Bound'], df_world.loc['X', 'Max_Bound']),
        (df_world.loc['Y', 'Min_Bound'], df_world.loc['Y', 'Max_Bound']),
        (df_world.loc['Z', 'Min_Bound'], df_world.loc['Z', 'Max_Bound'])
    )

    # 2. Wczytanie trajektorii
    df_traj = pd.read_csv(traj_file)
    drones = df_traj['drone_id'].unique()

    # 3. Wczytanie przeszkód
    df_obs = pd.read_csv(obs_file)

    # Konfiguracja wykresu
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Wizualizacja trajektorii lotu dronów", fontsize=16, color='white')

    # Rysowanie przeszkód
    for _, row in df_obs.iterrows():
        plot_cylinder(
            ax, 
            x=row['x'], 
            y=row['y'], 
            z_start=row['z'], 
            height=row['height'], 
            radius=row['radius'], 
            color='salmon', 
            alpha=0.4
        )

    # Kolory dla dronów (wspiera do 10 dronów, dla większej ilości powtarza)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Rysowanie trajektorii
    for idx, drone_id in enumerate(drones):
        drone_data = df_traj[df_traj['drone_id'] == drone_id].sort_values(by='time')
        x_vals = drone_data['x'].values
        y_vals = drone_data['y'].values
        z_vals = drone_data['z'].values
        
        color = colors[idx % len(colors)]
        
        # Rysowanie linii trasy
        ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=2, label=f"Dron {int(drone_id)}")
        
        # Zaznaczenie Startu i Celu (odpowiednio okrąg i krzyżyk)
        ax.scatter(x_vals[0], y_vals[0], z_vals[0], color=color, s=50, marker='o')
        ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color=color, s=100, marker='x')

    # Konfiguracja etykiet i wizuali
    ax.set_xlabel('Oś X [m]')
    ax.set_ylabel('Oś Y [m]')
    ax.set_zlabel('Oś Z [m]')
    
    # Wymuszenie fizycznych proporcji świata
    set_axes_equal_3d(ax, limits)
    
    # Dopasowanie tła (opcjonalnie dla ciemnego motywu)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Legenda (usuwamy duplikaty)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()