"""E2E test: detekcja kolizji *dron–dron* w pełnym pipelinie

Scenariusz problemu (obserwacja z 2026-05-07):
Dwa drony zbliżają się do siebie. Avoidance algorithm wykonuje gwałtowny
manewr unikowy → drony tracą stabilność → spadają i uderzają w ziemię.
PyBullet rejestruje kontakt z `ground_body_id=0` (kolizja z ziemią), ale
NIE rejestruje wcześniejszego styku dron–dron — bo go fizycznie nie było
(unik się powiódł kosztem stabilności). Wynikowy `collisions.csv` raportuje
wyłącznie `other_body_id=0` zamiast właściwej kolizji wewnątrzrojowej.

Ten test sprawdza odwrotny scenariusz — gdy *fizycznie* dochodzi do styku
dron–dron — i weryfikuje, że pipeline poprawnie raportuje:
1. `SwarmBaseWorld.get_detailed_collisions()` zwraca parę (agent_idx,
   other_body_id), gdzie `other_body_id` ∈ `DRONE_IDS` (NIE
   `ground_body_id`).
2. `SwarmBaseWorld.get_agent_collisions()` raportuje (idx_a, idx_b) =
   pełna identyfikacja kolizji wewnątrzrojowej.
3. `SimulationLogger.log_collision` zapisuje `other_body_id` jako body
   drona — `collisions.csv` pokazuje kolizję dron–dron, a NIE z ziemią.
4. `SimulationLogger.crashed_drones` zawiera oba drony (każdy mógł być
   pierwszym wykrywającym kontakt).

Warunek "zielony": **w `collision_buffer` musi być >=1 event, w którym
`other_body_id` matchuje jakiekolwiek `DRONE_IDS`.** Jeśli zawartość bufora
to tylko `other_body_id=ground_body_id=0`, test FAIL — co dokładnie
odpowiada raportowanemu bug-owi.

Test używa **rzeczywistego PyBullet'a** (DIRECT, headless) — bez mocków —
żeby wyłapywać regresje na faktycznym pipeline'ie kolizji (URDF collision
shape, contact point detection, body-id mapping). Cykl symulacji jest
celowo minimalny (≤ 50 kroków) bo pozycje początkowe dronów już się
przekrywają (CF2X collision shape: cylinder radius=0.06m; spawn 0.05m od
siebie ⇒ contact w t=0).
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from src.environments.EmptyWorld import EmptyWorld
from src.environments.abstraction.generate_obstacles import ObstaclesData
from src.environments.abstraction.generate_world_boundaries import WorldData
from src.environments.obstacles.ObstacleShape import ObstacleShape
from src.utils.SimulationLogger import SimulationLogger


@pytest.fixture
def overlapping_drones_world():
    """Inicjalizuje EmptyWorld z 2 dronami fizycznie nakładającymi się.

    CF2X URDF collision shape: cylinder radius=0.06m, length=0.025m.
    Spawn na (5.0, 5.0, 1.0) i (5.05, 5.0, 1.0) ⇒ horyzontalny dystans
    0.05m < 2·0.06m = 0.12m ⇒ shapes się przekrywają od t=0.
    Wysokość 1.0m wystarczająca by uniknąć styku z ziemią (z=0) w
    pierwszych krokach symulacji.
    """
    wd = WorldData(
        dimensions=np.array([10.0, 10.0, 5.0]),
        min_bounds=np.array([0.0, 0.0, 0.0]),
        max_bounds=np.array([10.0, 10.0, 5.0]),
        bounds=np.zeros((3, 2)),
        center=np.array([5.0, 5.0, 2.5]),
    )
    obstacles = ObstaclesData(
        data=np.zeros((0, 6), dtype=np.float64),
        shape_type=ObstacleShape.CYLINDER,
    )
    initial_xyzs = np.array([[5.0, 5.0, 1.0], [5.05, 5.0, 1.0]])
    end_xyzs = initial_xyzs.copy()

    env = EmptyWorld(
        world_data=wd,
        obstacles_data=obstacles,
        num_drones=2,
        primary_num_drones=2,
        initial_xyzs=initial_xyzs,
        end_xyzs=end_xyzs,
        gui=False,
    )
    try:
        env.reset(seed=0)
        yield env
    finally:
        # Każdy test musi zamknąć client by uniknąć leak'u między testami
        # (PyBullet trzyma global state per-process). Skip jeśli już closed.
        try:
            p.disconnect(env.CLIENT)
        except Exception:
            pass


def _step_physics_until_contact(env: EmptyWorld, max_steps: int = 50) -> list:
    """Stepuje fizykę aż `get_detailed_collisions` zwróci niepusty wynik
    lub osiągnie `max_steps`. Zwraca finalną listę kolizji.
    """
    collisions: list = []
    for _ in range(max_steps):
        p.stepSimulation()
        collisions = env.get_detailed_collisions()
        if collisions:
            break
    return collisions


class TestInterDroneCollisionE2E:
    def test_get_detailed_collisions_reports_drone_body_id(
        self, overlapping_drones_world
    ) -> None:
        env = overlapping_drones_world
        drone_body_ids = set(int(b) for b in env.DRONE_IDS)
        ground_body_id = int(env.ground_body_id) if env.ground_body_id is not None else -1

        collisions = _step_physics_until_contact(env)

        assert collisions, (
            "get_detailed_collisions zwróciło pustą listę po 50 krokach "
            "fizyki — drony nakładające się o 0.05m powinny być w kontakcie."
        )

        drone_drone_events = [
            (a_idx, o_id)
            for (a_idx, o_id) in collisions
            if int(o_id) in drone_body_ids
        ]

        assert drone_drone_events, (
            f"Brak kolizji dron–dron. Otrzymano {collisions} (DRONE_IDS="
            f"{sorted(drone_body_ids)}, ground={ground_body_id}). "
            "Pipeline raportuje wyłącznie kolizje z ziemią/przeszkodami — bug."
        )

    def test_get_agent_collisions_reports_inter_drone_pair(
        self, overlapping_drones_world
    ) -> None:
        env = overlapping_drones_world
        _step_physics_until_contact(env)

        agent_pairs = env.get_agent_collisions()
        # Po deduplikacji para (0,1) lub (1,0) — albo obie. Sprawdzamy
        # że istnieje przynajmniej jedna z indeksami w {0,1}.
        symmetric_pair_present = any(
            {int(a), int(b)} == {0, 1} for (a, b) in agent_pairs
        )
        assert symmetric_pair_present, (
            f"get_agent_collisions powinno zawierać parę {{0,1}}, otrzymano "
            f"{agent_pairs}. Bug w mapping body_id→agent_idx lub w fizyce."
        )

    def test_logger_persists_drone_drone_collision_to_csv(
        self, overlapping_drones_world, tmp_path: Path
    ) -> None:
        """Wymusza pełen pipeline: env.get_detailed_collisions →
        logger.log_collision → save → CSV. Zielony tylko jeśli zapisany
        `collisions.csv` ma ≥1 wiersz z `other_body_id` dronowym.
        """
        env = overlapping_drones_world
        drone_body_ids = set(int(b) for b in env.DRONE_IDS)

        logger = SimulationLogger(
            output_dir=str(tmp_path), log_freq=240, ctrl_freq=240, num_drones=2,
        )

        collisions_seen: list[tuple[int, int]] = []
        sim_time = 1.5  # > 1.0 (próg log_collision)
        for step in range(50):
            p.stepSimulation()
            sim_time += 1.0 / 240.0
            collisions = env.get_detailed_collisions()
            for d_id, o_id in collisions:
                logger.log_collision(sim_time, int(d_id), int(o_id))
                collisions_seen.append((int(d_id), int(o_id)))
            if collisions and any(
                int(o) in drone_body_ids for (_, o) in collisions
            ):
                break

        # Wymusza save — log_collision tylko buforuje, save() pisze CSV.
        logger.save()
        csv_path = tmp_path / "collisions.csv"
        assert csv_path.exists(), "save_logs nie wytworzył collisions.csv"

        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows, (
            "collisions.csv jest pusty — log_collision nie zarejestrował "
            f"żadnego eventu mimo że env raportowało {collisions_seen}."
        )

        drone_drone_rows = [
            row for row in rows if int(row["other_body_id"]) in drone_body_ids
        ]
        assert drone_drone_rows, (
            "collisions.csv zawiera wyłącznie kolizje z ziemią/przeszkodami "
            f"(rows={rows}, DRONE_IDS={sorted(drone_body_ids)}). "
            "Dokładnie ten symptom raportowany przez użytkownika — drone-drone "
            "contact zostaje pominięty."
        )

    def test_logger_marks_drone_as_crashed_after_inter_drone_contact(
        self, overlapping_drones_world, tmp_path: Path
    ) -> None:
        """`crashed_drones` musi zawierać przynajmniej jeden agent_idx z
        kolizji dron–dron. Bez tego downstream'owe `_process_collisions`
        w main.py nie usunie drona z `active_drones`.
        """
        env = overlapping_drones_world
        drone_body_ids = set(int(b) for b in env.DRONE_IDS)

        logger = SimulationLogger(
            output_dir=str(tmp_path), log_freq=240, ctrl_freq=240, num_drones=2,
        )
        sim_time = 1.5
        for _ in range(50):
            p.stepSimulation()
            sim_time += 1.0 / 240.0
            for d_id, o_id in env.get_detailed_collisions():
                if int(o_id) in drone_body_ids:
                    logger.log_collision(sim_time, int(d_id), int(o_id))

        assert logger.crashed_drones, (
            "crashed_drones jest puste — drone-drone collision nie "
            "wywołało odpowiedniego wpisu, mimo że env detektował styk."
        )


# ============================================================================
# PROXIMITY-BASED detection (drives the actual bug-fix)
# ============================================================================


@pytest.fixture
def separated_drones_world():
    """2 drony oddalone 2.0m horyzontalnie — fizycznego styku BRAK
    (CF2X collision shape: cylinder r=0.06m ⇒ kontakt @ 0.12m), ale
    ≤ `inter_uav_safety_threshold_m=4.0` (Kamień 1 ETL_TABLES.md
    Tabela 11). Symuluje "near-miss" — dron–dron NIE styka się
    fizycznie, więc obserwowany bug (kolizje raportowane wyłącznie
    jako ground=0) wynika z braku detekcji *zbliżenia*, nie braku
    kontaktu.
    """
    wd = WorldData(
        dimensions=np.array([10.0, 10.0, 5.0]),
        min_bounds=np.array([0.0, 0.0, 0.0]),
        max_bounds=np.array([10.0, 10.0, 5.0]),
        bounds=np.zeros((3, 2)),
        center=np.array([5.0, 5.0, 2.5]),
    )
    obstacles = ObstaclesData(
        data=np.zeros((0, 6), dtype=np.float64),
        shape_type=ObstacleShape.CYLINDER,
    )
    initial_xyzs = np.array([[5.0, 5.0, 1.0], [7.0, 5.0, 1.0]])  # 2.0m apart
    end_xyzs = initial_xyzs.copy()

    env = EmptyWorld(
        world_data=wd,
        obstacles_data=obstacles,
        num_drones=2,
        primary_num_drones=2,
        initial_xyzs=initial_xyzs,
        end_xyzs=end_xyzs,
        gui=False,
    )
    try:
        env.reset(seed=0)
        yield env
    finally:
        try:
            p.disconnect(env.CLIENT)
        except Exception:
            pass


class TestProximityInterDroneCollision:
    """Scenariusz raportowany przez użytkownika (2026-05-07): 2 drony
    zbliżają się → PID się nasyca przez impulse z LCP solver'a → drony
    tracą stabilność → spadają i uderzają w ziemię. Pre-fix pipeline
    rejestrował WYŁĄCZNIE finalny kontakt z `ground_body_id=0`.

    Fix (Krok 2 plan.md): `SwarmBaseWorld.get_inter_drone_proximity_collisions`
    wykrywa pary primary-swarm dronów które są ≤ INTER_DRONE_COLLISION_THRESHOLD_M
    (default 0.15m, = 1.25× teoretyczny styk 2*r=0.12m) ZANIM PyBullet
    wygeneruje LCP impulse. main.py `_process_collisions` woła to obok
    fizycznych kontaktów; obie kolizje (proximity + contact) trafiają do
    `collisions.csv` z `other_body_id` matchującym body drona.

    Threshold 0.15m — empiryczny pomiar w
    `/tmp/measure_repulsion_threshold.py`: PyBullet binarnie generuje LCP
    impulse przy dist≤0.12m (brak miękkiej strefy). 1.25× margin daje
    ~30ms wyprzedzenia (przy v=5m/s) na disable silnik+LiDAR ZANIM PID
    się nasyci.
    """

    def test_proximity_collisions_detected_at_default_threshold(
        self, separated_drones_world
    ) -> None:
        env = separated_drones_world

        # 2.0m apart > 0.15m threshold ⇒ proximity NIE powinno raportować.
        proximity = env.get_inter_drone_proximity_collisions()
        assert not proximity, (
            f"Drony oddalone 2.0m nie powinny być w proximity threshold "
            f"(default=0.15m), a otrzymano {proximity}."
        )

        # Z explicite większym threshold proximity POWINNO zwrócić parę.
        proximity_wide = env.get_inter_drone_proximity_collisions(threshold_m=4.0)
        assert proximity_wide, (
            f"threshold=4.0m, dist=2.0m → powinno raportować parę {{0,1}}, "
            f"otrzymano {proximity_wide}."
        )
        idx_pairs = {frozenset((int(a), int(b))) for a, b, _d in proximity_wide}
        assert frozenset({0, 1}) in idx_pairs

    def test_proximity_collision_uses_default_threshold_for_close_drones(
        self, tmp_path: Path
    ) -> None:
        """Drony 0.13m apart (>0.12m teoretyczny styk, ale ≤0.15m default
        threshold) — proximity detection łapie je ZANIM PyBullet wyśle LCP."""
        wd = WorldData(
            dimensions=np.array([10.0, 10.0, 5.0]),
            min_bounds=np.array([0.0, 0.0, 0.0]),
            max_bounds=np.array([10.0, 10.0, 5.0]),
            bounds=np.zeros((3, 2)),
            center=np.array([5.0, 5.0, 2.5]),
        )
        obstacles = ObstaclesData(
            data=np.zeros((0, 6), dtype=np.float64),
            shape_type=ObstacleShape.CYLINDER,
        )
        # 0.13m apart — ≤ 0.15m default proximity threshold, ale > 0.12m
        # contact threshold. PyBullet LCP NIE generuje impulse, ale nasza
        # detekcja powinna złapać.
        xyzs = np.array([[5.0, 5.0, 1.0], [5.13, 5.0, 1.0]])
        env = EmptyWorld(
            world_data=wd, obstacles_data=obstacles, num_drones=2,
            primary_num_drones=2, initial_xyzs=xyzs, end_xyzs=xyzs.copy(),
            gui=False,
        )
        try:
            env.reset(seed=0)
            for _ in range(3):
                p.stepSimulation()

            # PyBullet contact = NIE (fizycznie się nie stykają)
            assert not env.get_detailed_collisions(), (
                "Setup error: drony 0.13m apart nie powinny stykać się fizycznie."
            )
            # Proximity = TAK (łapie zanim contact)
            prox = env.get_inter_drone_proximity_collisions()
            assert prox, "0.13m apart powinno trafić w proximity threshold 0.15m."
            assert frozenset((prox[0][0], prox[0][1])) == frozenset({0, 1})
        finally:
            try:
                p.disconnect(env.CLIENT)
            except Exception:
                pass

    def test_get_all_inter_drone_collisions_combines_contact_and_proximity(
        self, overlapping_drones_world
    ) -> None:
        """Drony 0.05m apart (overlap, fizyczny kontakt). `get_all_inter_drone_collisions`
        zwraca parę z source='contact'. Proximity-only metoda też zwraca,
        ale unified wrapper deduplikuje na rzecz contact."""
        env = overlapping_drones_world
        for _ in range(3):
            p.stepSimulation()

        all_coll = env.get_all_inter_drone_collisions()
        assert all_coll, "Drony overlap (0.05m) powinny generować kolizję."
        assert any(src == "contact" for *_, src in all_coll), (
            f"Oczekiwany source='contact' dla overlap, otrzymano {all_coll}."
        )
        # Para {0,1} pojawia się dokładnie raz (deduplikacja contact vs proximity)
        keys = [frozenset((a, b)) for a, b, _d, _s in all_coll]
        assert keys.count(frozenset({0, 1})) == 1


class TestDisableDroneAfterCollision:
    """Krok 2 plan.md (2026-05-07): po wykryciu kolizji
    `SwarmFlightController.disable_drone(idx)` wyłącza silniki + LiDAR
    żeby oszczędzać CPU i nie raportować "false" hitów z dronu który
    fizycznie spada.

    Bezpośrednio testujemy interfejs `disable_drone` + `compute_actions`
    + `_run_lidar_and_detect` (przez `latest_scans[idx]`).
    """

    def _make_minimal_controller(self, num_drones: int = 2):
        """Tworzy minimalnego controller'a bez avoidance — wystarczy dla
        testów disable_drone i zerowych RPM."""
        from src.algorithms.SwarmFlightController import SwarmFlightController

        # Minimalny dummy parent (logger=None wystarczy dla nas).
        class _Parent:
            logger = None
            num_drones = 2
            initial_positions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
            end_positions = np.array([[5.0, 0.0, 1.0], [5.0, 1.0, 1.0]])

        ctrl = SwarmFlightController(
            parent=_Parent(),
            num_drones=num_drones,
            is_obstacle=False,
            avoidance_algorithm=None,
            params={
                "ctrl_freq": 48, "hover_duration": 0.5, "finish_radius": 0.5,
                "cruise_speed": 8.0, "max_accel": 2.0, "collision_radius": 0.4,
                "enable_avoidance": False,
            },
        )
        return ctrl

    def test_disable_drone_zeros_rpm_in_compute_actions(self) -> None:
        ctrl = self._make_minimal_controller(num_drones=2)

        # Pre-fill `_base_trajectories` z dummy splajnami (omijamy
        # `_prepare_trajectories` które wymaga parent.drones_trajectories).
        class _DummyTraj:
            def get_state_at_time(self, t):
                return (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        ctrl._base_trajectories = [_DummyTraj(), _DummyTraj()]
        ctrl._lidars = None  # skip lidar

        # Sztuczny stan: dron 0 hover, dron 1 hover.
        state = np.zeros(20, dtype=np.float64)
        state[2] = 1.0  # z=1m
        state[3:7] = [0.0, 0.0, 0.0, 1.0]  # quaternion (w-last)
        states = [state.copy(), state.copy()]

        # Pre-disable: oba drony generują niezerowe RPM (hover thrust).
        actions_before = ctrl.compute_actions(states, current_time=1.0)
        assert actions_before.shape == (2, 4)
        assert np.any(actions_before > 0), "Pre-disable RPM powinno być >0."

        # Disable drone 1
        ctrl.disable_drone(1)

        actions_after = ctrl.compute_actions(states, current_time=1.1)
        # Drone 0 — wciąż niezerowe RPM
        assert np.any(actions_after[0] > 0), "Drone 0 nie powinien być wyłączony."
        # Drone 1 — wszystkie 4 RPM = 0
        assert np.allclose(actions_after[1], 0.0), (
            f"Drone 1 RPM po disable = {actions_after[1]} ≠ 0."
        )

    def test_disable_drone_clears_latest_scans_and_is_idempotent(self) -> None:
        ctrl = self._make_minimal_controller(num_drones=2)

        # Sztucznie wstaw "stary" hit do latest_scans by zweryfikować clear.
        ctrl.latest_scans[0] = ["fake_hit_object"]
        ctrl.latest_scans[1] = ["fake_hit_object"]

        ctrl.disable_drone(0)
        assert ctrl.latest_scans[0] == []
        assert ctrl.latest_scans[1] == ["fake_hit_object"]

        # Idempotentne — drugie wywołanie nie powinno rzucać błędu.
        ctrl.disable_drone(0)
        assert 0 in ctrl._disabled_drones

        # Out-of-range — silent no-op (nie rzuca).
        ctrl.disable_drone(99)
        ctrl.disable_drone(-1)
        assert ctrl._disabled_drones == {0}

    def test_near_miss_drones_must_log_inter_drone_event_not_ground(
        self, tmp_path: Path,
    ) -> None:
        """User-reported scenariusz (2026-05-07): "Te '0' to kolizje między
        dronami". 2 drony są blisko (0.5m), drone 0 spada do ziemi
        (commanded altitude spadek), drone 1 hover. Pre-fix: pipeline
        zalogował WYŁĄCZNIE ground hit (`other_body_id=0`) tracąc
        kontekst że to inter-drone interaction.

        Fix (2026-05-07 decyzja A): rozszerzenie INTER_DRONE_COLLISION_THRESHOLD_M
        z 0.15m do 0.5m. Drony 0.5m apart trafiają w threshold ⇒ proximity
        event logowany ZANIM drone 0 spadnie do ziemi. `collisions.csv`
        ma teraz inter-drone wpis z `other_body_id ∈ DRONE_IDS`.
        """
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.utils.enums import DroneModel

        wd = WorldData(
            dimensions=np.array([10.0, 10.0, 5.0]),
            min_bounds=np.array([0.0, 0.0, 0.0]),
            max_bounds=np.array([10.0, 10.0, 5.0]),
            bounds=np.zeros((3, 2)),
            center=np.array([5.0, 5.0, 2.5]),
        )
        obstacles = ObstaclesData(
            data=np.zeros((0, 6), dtype=np.float64),
            shape_type=ObstacleShape.CYLINDER,
        )
        # Drone 0 spada do ziemi (target z=0), drone 1 hover. 0.4m horyzontalnie.
        # 0.4m daje margin do INTER_DRONE_COLLISION_THRESHOLD_M=0.5m: gdy
        # drone 0 zaczyna spadać, dist rośnie pionowo (sqrt(0.4² + Δz²)),
        # ale przez ≥ 100 ctrl steps utrzymuje dist ≤ 0.5m → proximity event.
        # Pre-fix (threshold 0.15m) min_dist > threshold cały czas → nic
        # nie wykryto. Post-fix (0.5m) → jest co najmniej 1 wpis dron-dron.
        initial_xyzs = np.array([[5.0, 5.0, 1.5], [5.4, 5.0, 1.5]])
        targets = np.array([[5.0, 5.0, 0.0], [5.4, 5.0, 1.5]])

        env = EmptyWorld(
            world_data=wd, obstacles_data=obstacles,
            num_drones=2, primary_num_drones=2,
            initial_xyzs=initial_xyzs, end_xyzs=targets, gui=False,
        )
        try:
            env.reset(seed=0)
            ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
            drone_body_ids = set(int(b) for b in env.DRONE_IDS)
            ground_body_id = int(env.ground_body_id)

            logger = SimulationLogger(
                output_dir=str(tmp_path), log_freq=240, ctrl_freq=240, num_drones=2,
            )

            min_inter_drone_dist = float("inf")
            sim_time = 1.5
            for step in range(int(5.0 * env.CTRL_FREQ)):
                rpms = np.zeros((2, 4), dtype=np.float64)
                for i in range(2):
                    state = env._getDroneStateVector(i)
                    action, _, _ = ctrls[i].computeControlFromState(
                        control_timestep=1.0 / env.CTRL_FREQ,
                        state=state, target_pos=targets[i],
                        target_rpy=np.array([0.0, 0.0, 0.0]),
                    )
                    rpms[i] = action

                env.step(rpms)
                sim_time += 1.0 / env.CTRL_FREQ

                pos_0 = np.array(p.getBasePositionAndOrientation(env.DRONE_IDS[0])[0])
                pos_1 = np.array(p.getBasePositionAndOrientation(env.DRONE_IDS[1])[0])
                cur_dist = float(np.linalg.norm(pos_0 - pos_1))
                if cur_dist < min_inter_drone_dist:
                    min_inter_drone_dist = cur_dist

                # Pipeline logic — same as production
                for d_id, o_id in env.get_detailed_collisions():
                    if d_id < 2:
                        logger.log_collision(sim_time, int(d_id), int(o_id))
                for a_idx, b_idx, _d in env.get_inter_drone_proximity_collisions():
                    body_b = int(env.DRONE_IDS[int(b_idx)])
                    body_a = int(env.DRONE_IDS[int(a_idx)])
                    if int(a_idx) not in logger.crashed_drones:
                        logger.log_collision(sim_time, int(a_idx), body_b)
                    if int(b_idx) not in logger.crashed_drones:
                        logger.log_collision(sim_time, int(b_idx), body_a)

            logger.save()
            csv_path = tmp_path / "collisions.csv"
            assert csv_path.exists(), (
                f"Brak collisions.csv. min_dist={min_inter_drone_dist:.2f}m. "
                "Setup error — drone 0 powinien spaść."
            )

            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            # Pre-condition: scenariusz jest realistyczny — drony są BLISKO
            # ale fizycznie się NIE dotykają (min_dist > 0.12m czyli 2*r CF2X).
            # Test ważny tylko jeśli min_dist mieści się w "near-miss zone":
            # > pre-fix threshold (0.15m) ale ≤ post-fix threshold (0.5m).
            assert min_inter_drone_dist > 0.15, (
                f"min_dist={min_inter_drone_dist:.3f}m za małe — drony "
                "fizycznie się stykają, scenariusz nie jest 'near-miss'."
            )
            assert min_inter_drone_dist <= 0.5, (
                f"min_dist={min_inter_drone_dist:.3f}m za duże — drony nie "
                "weszły w zone post-fix threshold (0.5m), nic nie wykryjemy."
            )

            # GŁÓWNA ASERCJA (oczekiwana ZAWODZIĆ obecnie):
            # collisions.csv MUSI zawierać przynajmniej jeden wpis z
            # `other_body_id ∈ DRONE_IDS`, dokumentując że drone fall był
            # related do bliskości drugiego drona. Obecnie pipeline tego
            # NIE oznacza — wszystkie wpisy są ground hits.
            drone_drone_rows = [
                r for r in rows if int(r["other_body_id"]) in drone_body_ids
            ]
            assert drone_drone_rows, (
                f"❌ Pipeline traci inter-drone context. min_dist between drones "
                f"= {min_inter_drone_dist:.3f}m (między 0.15m a 1.0m, czyli "
                "'near-miss' zone), drone hit ground, ale collisions.csv NIE "
                f"oznacza tego jako inter-drone interaction. rows={rows}, "
                f"DRONE_IDS={sorted(drone_body_ids)}, ground={ground_body_id}. "
                "Required fix: szerszy proximity threshold (np. 4.0m z Kamień 1) "
                "LUB post-hoc classification ground hits jako drone-induced."
            )
        finally:
            try:
                p.disconnect(env.CLIENT)
            except Exception:
                pass

    def test_realistic_cross_trajectory_logs_inter_drone_collision(
        self, tmp_path: Path,
    ) -> None:
        """REALISTYCZNY test (2026-05-07): replikuje scenariusz produkcyjny
        bez sztucznego overlap'u na starcie.

        Setup:
        - 2 drony oddalone 1.0m (initial_xyzs[1] - initial_xyzs[0] = (0, 1, 0))
        - Cross-trajectory: target drone 0 = pozycja drone 1, target drone 1 =
          pozycja drone 0. Drony LECĄ na siebie.
        - Real DSLPIDControl (gym_pybullet_drones) liczy RPM
        - Real env.step() (PYB_STEPS_PER_CTRL physics steps per ctrl tick)
        - Real `_process_collisions`-like logic z main.py (port poniżej):
          najpierw `get_detailed_collisions`, potem `get_inter_drone_proximity_collisions`.
        - Real SimulationLogger.save() do `tmp_path/collisions.csv`

        Asercja (DOKŁADNIE warunek użytkownika):
        Jeśli min(inter-drone distance) podczas symulacji < proximity threshold
        (drony FIZYCZNIE się zbliżyły do strefy kolizji), MUSI być co najmniej
        jeden wpis w `collisions.csv` z `other_body_id ∈ DRONE_IDS`.
        Jeśli wszystkie wpisy mają `other_body_id == ground_body_id` (typowo 0),
        test FAIL — to dokładnie ten bug który użytkownik raportuje.

        Test NIE fabrykuje warunków (drony NIE są pre-overlapped) — przepuszcza
        je przez prawdziwy pipeline tak samo jak SSA/MSFFOA/etc. produkuje
        kolizje w realnym runie. Jeśli pipeline traci drone-drone events
        (bo proximity threshold jest za ciasny, bo physics step granularity
        miss'uje contact, lub jakikolwiek inny powód), test FAIL.
        """
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.utils.enums import DroneModel

        wd = WorldData(
            dimensions=np.array([10.0, 10.0, 5.0]),
            min_bounds=np.array([0.0, 0.0, 0.0]),
            max_bounds=np.array([10.0, 10.0, 5.0]),
            bounds=np.zeros((3, 2)),
            center=np.array([5.0, 5.0, 2.5]),
        )
        obstacles = ObstaclesData(
            data=np.zeros((0, 6), dtype=np.float64),
            shape_type=ObstacleShape.CYLINDER,
        )
        # Cross-trajectory: drone 0 leci do (5, 5.5), drone 1 leci do (5, 4.5)
        # Initial dist: 1.0m; trajektorie krzyżują się w środku.
        initial_xyzs = np.array([[5.0, 4.5, 1.0], [5.0, 5.5, 1.0]])
        targets = np.array([[5.0, 5.5, 1.0], [5.0, 4.5, 1.0]])

        env = EmptyWorld(
            world_data=wd, obstacles_data=obstacles,
            num_drones=2, primary_num_drones=2,
            initial_xyzs=initial_xyzs, end_xyzs=targets, gui=False,
        )
        try:
            env.reset(seed=0)
            ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(2)]
            drone_body_ids = set(int(b) for b in env.DRONE_IDS)
            ground_body_id = int(env.ground_body_id) if env.ground_body_id is not None else -1

            logger = SimulationLogger(
                output_dir=str(tmp_path), log_freq=240, ctrl_freq=240, num_drones=2,
            )

            # Symulujemy ~6s bezpośrednio przez PYB_FREQ steps + co physics
            # step jeden control update. Robimy jak main.py: env.step(action),
            # potem _process_collisions equivalent.
            min_inter_drone_dist = float("inf")
            sim_time = 1.5  # > 1.0 (próg log_collision)

            for step in range(int(6.0 * env.CTRL_FREQ)):
                # Compute control actions
                rpms = np.zeros((2, 4), dtype=np.float64)
                for i in range(2):
                    state = env._getDroneStateVector(i)
                    action, _, _ = ctrls[i].computeControlFromState(
                        control_timestep=1.0 / env.CTRL_FREQ,
                        state=state, target_pos=targets[i],
                        target_rpy=np.array([0.0, 0.0, 0.0]),
                    )
                    rpms[i] = action

                env.step(rpms)
                sim_time += 1.0 / env.CTRL_FREQ

                # Track min inter-drone distance
                pos_0 = np.array(p.getBasePositionAndOrientation(env.DRONE_IDS[0])[0])
                pos_1 = np.array(p.getBasePositionAndOrientation(env.DRONE_IDS[1])[0])
                cur_dist = float(np.linalg.norm(pos_0 - pos_1))
                if cur_dist < min_inter_drone_dist:
                    min_inter_drone_dist = cur_dist

                # Port `_process_collisions` z main.py — łapanie obu źródeł:
                # 1) fizyczne kontakty
                for d_id, o_id in env.get_detailed_collisions():
                    if d_id < 2:
                        logger.log_collision(sim_time, int(d_id), int(o_id))
                # 2) proximity-based inter-drone
                for a_idx, b_idx, _d in env.get_inter_drone_proximity_collisions():
                    body_b = int(env.DRONE_IDS[int(b_idx)])
                    body_a = int(env.DRONE_IDS[int(a_idx)])
                    if int(a_idx) not in logger.crashed_drones:
                        logger.log_collision(sim_time, int(a_idx), body_b)
                    if int(b_idx) not in logger.crashed_drones:
                        logger.log_collision(sim_time, int(b_idx), body_a)

            logger.save()
            csv_path = tmp_path / "collisions.csv"
            assert csv_path.exists(), (
                f"Brak collisions.csv (sim min_dist={min_inter_drone_dist:.3f}m). "
                "Symulacja nie wygenerowała żadnej kolizji."
            )

            with csv_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            # Pre-condition: drony MUSIAŁY się zbliżyć do strefy kolizji,
            # inaczej test nie aplikuje (skip). W praktyce dla cross-trajectory
            # 1m apart, min_dist schodzi do <0.15m (z naszego eksperymentu —
            # do 0.025m).
            assert min_inter_drone_dist < env.INTER_DRONE_COLLISION_THRESHOLD_M, (
                f"Drony nie zbliżyły się dostatecznie (min_dist={min_inter_drone_dist:.3f}m, "
                f"threshold={env.INTER_DRONE_COLLISION_THRESHOLD_M}m). Test nie aplikuje — "
                "albo PID je zatrzymał albo cross-trajectory geometry nie wystarcza."
            )

            # GŁÓWNA ASERCJA: collisions.csv MUSI zawierać przynajmniej jeden
            # wpis z `other_body_id ∈ DRONE_IDS`. Brak takiego wpisu = bug
            # raportowany przez użytkownika (kolizja dron-dron maskowana
            # jako ground hit).
            drone_drone_rows = [
                r for r in rows if int(r["other_body_id"]) in drone_body_ids
            ]
            ground_only_rows = [
                r for r in rows if int(r["other_body_id"]) == ground_body_id
            ]

            assert drone_drone_rows, (
                f"❌ BUG: collisions.csv NIE ma żadnej kolizji dron-dron. "
                f"min_inter_drone_dist={min_inter_drone_dist:.3f}m "
                f"(threshold={env.INTER_DRONE_COLLISION_THRESHOLD_M}m, "
                f"drony fizycznie się zbliżyły!). "
                f"Wszystkie wiersze: rows={rows}, "
                f"DRONE_IDS={sorted(drone_body_ids)}, ground={ground_body_id}, "
                f"ground_only={len(ground_only_rows)}. "
                "Pipeline traci inter-drone events — albo proximity threshold "
                "jest za ciasny, albo `_process_collisions` jest w złej kolejności, "
                "albo physics step granularity miss'uje contact."
            )
        finally:
            try:
                p.disconnect(env.CLIENT)
            except Exception:
                pass

    def test_disable_drone_does_not_freeze_body(
        self, overlapping_drones_world
    ) -> None:
        """Decyzja użytkownika 2026-05-08: `disable_drone` NIE freeze'uje
        fizyki PyBullet — drone ma się normalnie zderzyć i spaść na ziemię
        pod wpływem grawitacji. Wcześniejsze freeze (`mass=0` +
        `resetBaseVelocity(0)`) ukrywało faktyczną dynamikę post-collision.

        Test weryfikuje że po `disable_drone`:
        - mass body_id pozostaje > 0 (drone podlega grawitacji),
        - residual velocity nie jest zerowane przez nasz kod,
        - po kilku krokach fizyki drone PRZESUNĄŁ SIĘ (spadł).
        """
        from src.algorithms.SwarmFlightController import SwarmFlightController

        env = overlapping_drones_world
        body_id_0 = int(env.DRONE_IDS[0])

        # Pre-disable: drone ma mass > 0 (z URDF: 0.027 kg dla CF2X).
        info_before = p.getDynamicsInfo(body_id_0, -1)
        mass_before = info_before[0]
        assert mass_before > 0, f"Pre-disable mass={mass_before}, expected >0"

        # Lekki controller bez trajektorii (omijamy _prepare_trajectories).
        class _Parent:
            environment = env
            drones_trajectories = None
            logger = None
        ctrl = SwarmFlightController(
            parent=_Parent(),
            num_drones=int(env.primary_num_drones),
            is_obstacle=False,
            avoidance_algorithm=None,
            params={
                "ctrl_freq": 48, "hover_duration": 0.5, "finish_radius": 0.5,
                "cruise_speed": 8.0, "max_accel": 2.0, "collision_radius": 0.4,
                "enable_avoidance": False,
            },
        )

        ctrl.disable_drone(0)

        # Post-disable: mass UNCHANGED (drone wciąż podlega fizyce).
        info_after = p.getDynamicsInfo(body_id_0, -1)
        mass_after = info_after[0]
        assert mass_after == mass_before, (
            f"Post-disable mass={mass_after}, expected {mass_before} — "
            "decyzja 2026-05-08: disable NIE freeze'uje fizyki."
        )

        # Po kilku krokach fizyki drone spada (grawitacja działa).
        pos_before, _ = p.getBasePositionAndOrientation(body_id_0)
        for _ in range(50):
            p.stepSimulation()
        pos_after, _ = p.getBasePositionAndOrientation(body_id_0)
        # Z-component spadł (grawitacja).
        assert pos_after[2] < pos_before[2], (
            f"Drone nie spadł: z_before={pos_before[2]}, z_after={pos_after[2]}. "
            "Po disable_drone fizyka powinna być nienaruszona — drone spada balistycznie."
        )
