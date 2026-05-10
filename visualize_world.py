#!/usr/bin/env python3
"""Wizualizacja rzutu 2D (widok z gory) swiata symulacji z katalogu wynikow.

Skrypt czyta surowe pliki CSV/YAML z katalogu pojedynczego runu i generuje
schemat XY pokazujacy granice swiata oraz wszystkie przeszkody w odpowiednim
ksztalcie (cylinder dla forest, box dla urban). Opcjonalnie nakłada
zaplanowane trajektorie i pozycje startowe/docelowe.

Pliki wymagane (w katalogu run_dir):
  - world_boundaries.csv   - wymiary swiata X, Y, Z
  - generated_obstacles.csv - przeszkody (auto-detekcja cylinder vs box)

Pliki opcjonalne (uzywane przy odpowiednich flagach):
  - counted_trajectories.csv - zaplanowane waypointy per drone
  - .hydra/config.yaml       - pozycje startowe i docelowe dronow

Przyklad uzycia:
    python visualize_world.py results/exp_*/nsga-3_forest_seed1/
    python visualize_world.py results/.../ -o forest_view.pdf
    python visualize_world.py results/.../ --show-trajectories --show-start-end
    python visualize_world.py results/.../ -o view.png --safety-margin 1.0
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


@dataclass(frozen=True)
class WorldBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def lx(self) -> float:
        return self.x_max - self.x_min

    @property
    def ly(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class CylinderObstacle:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class BoxObstacle:
    x: float
    y: float
    length_x: float
    width_y: float


Obstacle = CylinderObstacle | BoxObstacle


def read_world_bounds(path: Path) -> WorldBounds:
    """Parsuje world_boundaries.csv (kolumny: Axis, Dimension, Min_Bound, Max_Bound, Center)."""
    bounds: dict[str, tuple[float, float]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            axis = row["Axis"]
            bounds[axis] = (float(row["Min_Bound"]), float(row["Max_Bound"]))
    missing = {"X", "Y", "Z"} - bounds.keys()
    if missing:
        raise ValueError(f"world_boundaries.csv: brak osi {sorted(missing)}")
    return WorldBounds(
        x_min=bounds["X"][0], x_max=bounds["X"][1],
        y_min=bounds["Y"][0], y_max=bounds["Y"][1],
        z_min=bounds["Z"][0], z_max=bounds["Z"][1],
    )


def read_obstacles(path: Path) -> list[Obstacle]:
    """Parsuje generated_obstacles.csv z auto-detekcja typu (cylinder vs box).

    Cylinder: kolumny x, y, z, radius, height [, unused_dim] (forest).
    Box:      kolumny x, y, z, length, width, height (urban).
    """
    obstacles: list[Obstacle] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        is_cylinder = "radius" in cols
        is_box = "length" in cols and "width" in cols
        if not (is_cylinder or is_box):
            raise ValueError(
                f"generated_obstacles.csv: nieobslugiwany format kolumn {sorted(cols)}"
            )
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            if is_cylinder:
                obstacles.append(CylinderObstacle(x, y, float(row["radius"])))
            else:
                obstacles.append(BoxObstacle(
                    x, y, float(row["length"]), float(row["width"])
                ))
    return obstacles


def read_trajectories(path: Path) -> dict[int, list[tuple[float, float]]]:
    """Parsuje counted_trajectories.csv. Zwraca {drone_id: [(x, y), ...]}."""
    per_drone: dict[int, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drone_id = int(row["drone_id"])
            per_drone.setdefault(drone_id, []).append(
                (float(row["x"]), float(row["y"]))
            )
    return per_drone


def read_start_end_from_config(run_dir: Path) -> tuple[list, list] | None:
    """Wyciaga initial_xyzs i end_xyzs z .hydra/config.yaml. None gdy brak yaml."""
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
    except ImportError:
        print("[WARN] PyYAML niedostepne — pomijam start/end markery.", file=sys.stderr)
        return None
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    starts = cfg.get("initial_xyzs") or cfg.get("environment", {}).get("initial_xyzs")
    ends = cfg.get("end_xyzs") or cfg.get("environment", {}).get("end_xyzs")
    if starts is None or ends is None:
        return None
    return starts, ends


def detect_environment(run_dir: Path) -> str:
    """Heurystyczna detekcja srodowiska z nazwy katalogu."""
    name = run_dir.name.lower()
    if "forest" in name:
        return "forest"
    if "urban" in name:
        return "urban"
    if "empty" in name:
        return "empty"
    return "unknown"


# Paleta dla 5 dronów (default) — swarm-friendly kolory wysokoprzeciwstawne.
_DRONE_COLORS = ["#1976d2", "#388e3c", "#f57c00", "#7b1fa2", "#00838f",
                 "#c62828", "#5d4037", "#455a64", "#e65100", "#33691e"]


def _compute_layout(
    bounds: WorldBounds,
    max_aspect_ratio: float = 5.0,
    target_width_in: float = 12.0,
) -> tuple[tuple[float, float], float]:
    """Liczy figsize i matplotlib aspect dla orientacji poziomej.

    Po obrocie 90 stopni: os misji (world Y) lezy poziomo, korytarz (world X)
    lezy pionowo. Naturalny stosunek danych to L_y/L_x (10:1 dla forest,
    3,33:1 dla urban). Aby plot miescil sie na stronie A4 i pozostal czytelny,
    ograniczamy display ratio do `max_aspect_ratio` (default 5:1) — przy
    wezszych korytarzach wprowadzamy lagodna kompresje osi pionowej (przeszkody
    cylindryczne staja sie nieznacznie owalne, ale pozostaja czytelne).

    Returns:
        ((width_in, height_in), mpl_aspect)
        - width_in, height_in: figsize w calach
        - mpl_aspect: argument dla ax.set_aspect()
            1.0 = equal (brak deformacji geometrii)
            <1.0 = os pionowa rozciagnieta (przeszkody wyzsze niz szersze)
    """
    L_horiz = bounds.ly  # data range na poziomej osi po obrocie
    L_vert = bounds.lx   # data range na pionowej osi po obrocie
    natural_ratio = L_horiz / L_vert

    if natural_ratio <= max_aspect_ratio:
        # Naturalna geometria miesci sie w limicie — bez deformacji
        display_ratio = natural_ratio
        mpl_aspect = 1.0
    else:
        # Wymuszamy max_aspect: 1 jednostka X (pionowa) bedzie zajmowala
        # wiecej pikseli niz 1 jednostka Y (pozioma). Deformacja = stretch_factor.
        display_ratio = max_aspect_ratio
        mpl_aspect = max_aspect_ratio / natural_ratio  # < 1.0

    fig_width = target_width_in
    fig_height = fig_width / display_ratio
    return (fig_width, fig_height), mpl_aspect


def render_world(
    bounds: WorldBounds,
    obstacles: list[Obstacle],
    trajectories: dict[int, list[tuple[float, float]]] | None,
    start_end: tuple[list, list] | None,
    title: str,
    safety_margin: float = 0.0,
    max_aspect_ratio: float = 5.0,
    output_path: Optional[Path] = None,
) -> None:
    """Generuje rzut XY w orientacji poziomej (os misji Y - pozioma).

    Konwencja po obrocie 90 stopni: world_y -> plot_x (mission direction),
    world_x -> plot_y (corridor width). Dron startuje po lewej, leci w prawo.
    """
    figsize, mpl_aspect = _compute_layout(bounds, max_aspect_ratio)
    fig, ax = plt.subplots(figsize=figsize)

    # 1. Granice swiata (przerywany prostokat)
    # Po obrocie: lewy-dolny rog plotu = (world_y_min, world_x_min)
    ax.add_patch(Rectangle(
        (bounds.y_min, bounds.x_min), bounds.ly, bounds.lx,
        fill=False, edgecolor="black", linewidth=1.2, linestyle="--",
        label="granice swiata",
    ))

    # 2. Przeszkody (z opcjonalna strefa marginesu bezpieczenstwa)
    margin_label_used = False
    obs_label_used = False
    for obs in obstacles:
        if isinstance(obs, CylinderObstacle):
            # Cylinder: po obrocie srodek na (world_y, world_x), promien zachowany
            if safety_margin > 0:
                ax.add_patch(Circle(
                    (obs.y, obs.x), obs.radius + safety_margin,
                    facecolor="#ffcdd2", edgecolor="none", alpha=0.4,
                    label=None if margin_label_used else f"margines $\\delta$ = {safety_margin} m",
                ))
                margin_label_used = True
            ax.add_patch(Circle(
                (obs.y, obs.x), obs.radius,
                facecolor="#c62828", edgecolor="#7f0000", linewidth=0.7, alpha=0.85,
                label=None if obs_label_used else "przeszkoda",
            ))
            obs_label_used = True
        else:  # BoxObstacle
            # Box: po obrocie szerokosc plot-pozioma = world_y_extent (width_y),
            #                  szerokosc plot-pionowa = world_x_extent (length_x)
            if safety_margin > 0:
                ax.add_patch(Rectangle(
                    (obs.y - obs.width_y / 2 - safety_margin,
                     obs.x - obs.length_x / 2 - safety_margin),
                    obs.width_y + 2 * safety_margin,
                    obs.length_x + 2 * safety_margin,
                    facecolor="#ffcdd2", edgecolor="none", alpha=0.4,
                    label=None if margin_label_used else f"margines $\\delta$ = {safety_margin} m",
                ))
                margin_label_used = True
            ax.add_patch(Rectangle(
                (obs.y - obs.width_y / 2, obs.x - obs.length_x / 2),
                obs.width_y, obs.length_x,
                facecolor="#c62828", edgecolor="#7f0000", linewidth=0.7, alpha=0.85,
                label=None if obs_label_used else "przeszkoda",
            ))
            obs_label_used = True

    # 3. Trajektorie zaplanowane per drone
    if trajectories:
        for drone_id, path in sorted(trajectories.items()):
            if not path:
                continue
            world_xs, world_ys = zip(*path)
            color = _DRONE_COLORS[drone_id % len(_DRONE_COLORS)]
            # Po obrocie: plot_x = world_y, plot_y = world_x
            ax.plot(world_ys, world_xs, color=color, linewidth=1.2, alpha=0.8,
                    label=f"drone {drone_id}")

    # 4. Markery start/end
    if start_end is not None:
        starts, ends = start_end
        # Po obrocie: plot_x = world_y, plot_y = world_x
        sx_plot = [p[1] for p in starts]
        sy_plot = [p[0] for p in starts]
        ex_plot = [p[1] for p in ends]
        ey_plot = [p[0] for p in ends]
        ax.scatter(sx_plot, sy_plot, marker="o", s=80, c="white", edgecolor="black",
                   linewidth=1.5, zorder=5, label="start")
        ax.scatter(ex_plot, ey_plot, marker="*", s=140, c="gold", edgecolor="black",
                   linewidth=1.0, zorder=5, label="cel")

    # 5. Osie, tytul, legenda
    margin_horiz = bounds.ly * 0.01
    margin_vert = bounds.lx * 0.05
    ax.set_xlim(bounds.y_min - margin_horiz, bounds.y_max + margin_horiz)
    ax.set_ylim(bounds.x_min - margin_vert, bounds.x_max + margin_vert)
    ax.set_aspect(mpl_aspect)
    ax.set_xlabel("Y [m]  (kierunek misji)")
    ax.set_ylabel("X [m]  (szerokosc korytarza)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle=":")

    # Adnotacja o przeskalowaniu (jesli zastosowano)
    if mpl_aspect < 1.0:
        stretch = 1.0 / mpl_aspect
        ax.text(
            0.99, 1.01,
            f"os X przeskalowana x{stretch:.1f} dla czytelnosci",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, style="italic", color="#555",
        )

    # Legenda — usuniecie duplikatow, na zewnatrz osi by nie zaslaniala przeszkod
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc="upper left", bbox_to_anchor=(1.005, 1.0),
              fontsize=8, framealpha=0.9, borderaxespad=0.0)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Zapisano: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rzut 2D (widok z gory) swiata symulacji.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Sciezka do katalogu pojedynczego runu",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Sciezka pliku wyjsciowego (PDF/PNG/SVG). Brak = okno interaktywne.",
    )
    parser.add_argument(
        "-t", "--show-trajectories", action="store_true",
        help="Naloz zaplanowane trajektorie z counted_trajectories.csv",
    )
    parser.add_argument(
        "-s", "--show-start-end", action="store_true",
        help="Naloz markery start/cel z .hydra/config.yaml",
    )
    parser.add_argument(
        "--safety-margin", type=float, default=0.0,
        help="Wizualizuj margines bezpieczenstwa (jasnoczerwona aureola wokol przeszkod) [m]",
    )
    parser.add_argument(
        "--max-aspect", type=float, default=5.0,
        help="Maksymalny stosunek display ratio plotu (poziomy:pionowy). "
             "Przy bardziej wydluzonych swiatach (forest = 10:1) os pionowa "
             "zostanie lagodnie rozciagnieta dla czytelnosci przeszkod. "
             "Default: 5.0. Uzyj 1000.0 by wymusic 'aspect=equal' bez deformacji.",
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Niestandardowy tytul wykresu",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"Katalog nie istnieje: {run_dir}")

    bounds_path = run_dir / "world_boundaries.csv"
    obs_path = run_dir / "generated_obstacles.csv"
    if not bounds_path.exists():
        parser.error(f"Brak pliku: {bounds_path}")
    if not obs_path.exists():
        parser.error(f"Brak pliku: {obs_path}")

    bounds = read_world_bounds(bounds_path)
    obstacles = read_obstacles(obs_path)

    trajectories: dict[int, list[tuple[float, float]]] | None = None
    if args.show_trajectories:
        traj_path = run_dir / "counted_trajectories.csv"
        if traj_path.exists():
            trajectories = read_trajectories(traj_path)
        else:
            print(f"[WARN] {traj_path} nie istnieje — pomijam trajektorie.",
                  file=sys.stderr)

    start_end = None
    if args.show_start_end:
        start_end = read_start_end_from_config(run_dir)
        if start_end is None:
            print("[WARN] Brak .hydra/config.yaml lub initial_xyzs/end_xyzs — "
                  "pomijam markery start/cel.", file=sys.stderr)

    env_name = detect_environment(run_dir)
    title = args.title or (
        f"Widok z gory ({env_name}) — {run_dir.name}\n"
        f"swiat {bounds.lx:.0f}x{bounds.ly:.0f} m, "
        f"przeszkody: {len(obstacles)}"
    )

    render_world(
        bounds=bounds,
        obstacles=obstacles,
        trajectories=trajectories,
        start_end=start_end,
        title=title,
        safety_margin=args.safety_margin,
        max_aspect_ratio=args.max_aspect,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
