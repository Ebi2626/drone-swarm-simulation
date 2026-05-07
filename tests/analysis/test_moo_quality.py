"""Unit tests dla `populate_moo_quality` — spread/spacing/r2/gd/igd+
na trywialnych, ręcznie policzalnych przykładach.

Reference: Riquelme et al. (2015), Ishibuchi et al. (2015), Schott (1995).
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from src.analysis.db.populate_moo_quality import (
    _generational_distance,
    _generate_simplex_weights,
    _hypervolume,
    _igd_plus,
    _non_dominated,
    _r2_indicator,
    _spacing_metric,
    _spread_metric,
    populate_moo_quality,
)


class TestNonDominated:
    def test_keeps_only_non_dominated(self) -> None:
        F = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.5, 1.5]])
        front = _non_dominated(F)
        # (3,3) jest zdominowane przez (1.5,1.5) i (1,2)/(2,1).
        coords = {tuple(r) for r in front}
        assert (1.0, 2.0) in coords
        assert (2.0, 1.0) in coords
        assert (1.5, 1.5) in coords
        assert (3.0, 3.0) not in coords

    def test_empty_input(self) -> None:
        F = np.zeros((0, 3))
        assert _non_dominated(F).shape == (0, 3)


class TestSpacing:
    def test_uniform_spacing_yields_zero(self) -> None:
        # Punkty równomiernie rozłożone na linii prostej w R²: distances
        # do najbliższego sąsiada są stałe ⇒ S = 0.
        front = np.array([[0.0, 5.0], [1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
        s = _spacing_metric(front)
        assert s == pytest.approx(0.0, abs=1e-9)

    def test_two_points_well_defined(self) -> None:
        front = np.array([[0.0, 0.0], [3.0, 4.0]])
        # d_min dla obu = 3+4 = 7 (Manhattan); std = 0.
        assert _spacing_metric(front) == pytest.approx(0.0, abs=1e-9)


class TestSpread:
    def test_two_points_returns_value(self) -> None:
        front = np.array([[0.0, 1.0], [1.0, 0.0]])
        spread = _spread_metric(front)
        assert spread is not None
        # Przy 2 punktach diff = jedna odległość, |d_i − d̄| = 0 ⇒ Δ ~ 0.
        assert spread == pytest.approx(0.0, abs=1e-9)

    def test_single_point_returns_none(self) -> None:
        assert _spread_metric(np.array([[1.0, 2.0]])) is None


class TestR2:
    def test_perfect_match_to_ideal(self) -> None:
        # Front zawiera ideal point (0,0); r2 powinien być małe.
        front = np.array([[0.0, 0.0], [1.0, 1.0]])
        weights = _generate_simplex_weights(2, n_partitions=4)
        r2 = _r2_indicator(front, weights)
        assert r2 == pytest.approx(0.0, abs=1e-9)

    def test_spread_far_from_ideal_larger(self) -> None:
        # R2 mierzy odległość fronta od własnego ideal'a (component-wise min).
        # Dla fronta {(1,2),(2,1)} z_star=(1,1), R2 ∝ skala fronta;
        # 4× rozszerzenie skali → 4× R2 (Tchebycheff scales linearly).
        weights = _generate_simplex_weights(2, n_partitions=4)
        r2_close = _r2_indicator(np.array([[1.0, 2.0], [2.0, 1.0]]), weights)
        r2_far = _r2_indicator(np.array([[1.0, 5.0], [5.0, 1.0]]), weights)
        assert r2_far > r2_close
        assert r2_close > 0.0


class TestGenerationalDistance:
    def test_zero_when_front_in_reference(self) -> None:
        front = np.array([[1.0, 2.0], [2.0, 1.0]])
        ref = front.copy()
        gd = _generational_distance(front, ref)
        assert gd == pytest.approx(0.0, abs=1e-9)

    def test_constant_offset(self) -> None:
        front = np.array([[1.0, 1.0]])
        ref = np.array([[0.0, 0.0]])
        # Distance = sqrt(2). p=2 ⇒ GD = (mean(d^2))^(1/2) = sqrt(2).
        gd = _generational_distance(front, ref)
        assert gd == pytest.approx(np.sqrt(2.0), abs=1e-9)


class TestHypervolume:
    def test_single_point_box_volume(self) -> None:
        # F={(0,0)}, r*=(1,1) → HV = (1-0)·(1-0) = 1.
        front = np.array([[0.0, 0.0]])
        r = np.array([1.0, 1.0])
        hv = _hypervolume(front, r)
        assert hv == pytest.approx(1.0, abs=1e-9)

    def test_dominated_point_yields_zero(self) -> None:
        # F={(2,2)} jest zdominowane przez r*=(1,1) (NIE: f<r component-wise).
        # Funkcja zwraca 0.0 (brak wkładu).
        front = np.array([[2.0, 2.0]])
        r = np.array([1.0, 1.0])
        assert _hypervolume(front, r) == 0.0

    def test_two_point_2d_known_value(self) -> None:
        # F={(0,1),(1,0)}, r*=(2,2). HV = (2·2) - 2·(1·1) - 1·1 ?
        # Spróbujmy ręcznie: po sortowaniu po x: (0,1),(1,0).
        # HV = (2-0)·(2-1) + (2-1)·(2-0) - (2-1)·(2-1)
        #    = 2·1 + 1·2 - 1·1 = 2 + 2 - 1 = 3.
        front = np.array([[0.0, 1.0], [1.0, 0.0]])
        r = np.array([2.0, 2.0])
        hv = _hypervolume(front, r)
        assert hv == pytest.approx(3.0, abs=1e-6)

    def test_higher_is_better_after_improvement(self) -> None:
        # Front lepszy (bliżej origin) → większe HV przy ustalonym r*.
        r = np.array([5.0, 5.0])
        hv_worse = _hypervolume(np.array([[3.0, 3.0]]), r)
        hv_better = _hypervolume(np.array([[1.0, 1.0]]), r)
        assert hv_better > hv_worse


class TestIGDPlus:
    def test_zero_when_front_dominates_reference(self) -> None:
        # Front lepszy lub równy reference ⇒ d^+ = 0 dla każdego ref.
        front = np.array([[0.0, 0.0]])
        ref = np.array([[1.0, 1.0]])
        # f − r = (-1,-1) → max(0, *) = 0 → d_plus = 0.
        igd = _igd_plus(front, ref)
        assert igd == pytest.approx(0.0, abs=1e-9)

    def test_penalty_when_front_worse(self) -> None:
        # Front gorszy: f = (2,2), ref = (1,1) → max(0, 1) per komponent
        # → d_plus = sqrt(2).
        front = np.array([[2.0, 2.0]])
        ref = np.array([[1.0, 1.0]])
        igd = _igd_plus(front, ref)
        assert igd == pytest.approx(np.sqrt(2.0), abs=1e-9)


@pytest.fixture
def synthetic_db_with_h5(tmp_path: Path):
    """Tworzy minimalny `analysis.db` + h5 dla 1 runu i populate'uje."""
    from src.analysis.db.initialize_database import initialize_database

    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    run_dir = exp_dir / "msffoa_forest_msffoa_seed1"
    run_dir.mkdir(parents=True)
    h5_dir = run_dir / "optimization_history"
    h5_dir.mkdir()

    # 3 generacje × 5 individuals × 3 obj. Pierwsza gen rozproszona;
    # ostatnia "skupiona" na froncie dla testu spread.
    rng = np.random.default_rng(42)
    obj = rng.uniform(low=0.0, high=10.0, size=(3, 5, 3))
    obj[2, 0] = [0.0, 1.0, 1.0]
    obj[2, 1] = [1.0, 0.0, 1.0]
    obj[2, 2] = [1.0, 1.0, 0.0]
    obj[2, 3] = [0.5, 0.5, 0.5]
    obj[2, 4] = [10.0, 10.0, 10.0]  # zdominowany
    h5_path = h5_dir / "optimization_history.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("objectives_matrix", data=obj)

    db_path = initialize_database(exp_dir)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO runs (
            run_id, run_dir_name, source_path, optimizer_algo, avoidance_algo,
            environment, seed, algorithm_pair
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("run1", "msffoa_forest_msffoa_seed1", str(run_dir),
         "MSFOA", "MSFOA", "forest", 1, "MSFOA_MSFOA"),
    )
    conn.commit()
    return conn, h5_path


class TestPopulateMooQuality:
    def test_inserts_spread_spacing_r2(self, synthetic_db_with_h5) -> None:
        conn, h5_path = synthetic_db_with_h5
        populate_moo_quality(conn, "run1", h5_path, reference_set=None)
        conn.commit()
        rows = conn.execute(
            """
            SELECT generation, metric_name, metric_value
            FROM optimization_generation_stats
            WHERE source_name = 'moo_quality'
            ORDER BY generation, metric_name
            """
        ).fetchall()
        names_per_gen: dict[int, set[str]] = {}
        for g, n, _v in rows:
            names_per_gen.setdefault(g, set()).add(n)
        for g, names in names_per_gen.items():
            assert "spread" in names or "spacing" in names or "r2_indicator" in names

    def test_gd_igd_inserted_when_reference_provided(self, synthetic_db_with_h5) -> None:
        conn, h5_path = synthetic_db_with_h5
        ref = np.array([[0.0, 0.0, 0.0]])
        populate_moo_quality(conn, "run1", h5_path, reference_set=ref)
        conn.commit()
        rows = conn.execute(
            """
            SELECT metric_name FROM optimization_generation_stats
            WHERE source_name = 'moo_quality'
            """
        ).fetchall()
        names = {r[0] for r in rows}
        assert "gd" in names
        assert "igd_plus" in names

    def test_idempotent_replace(self, synthetic_db_with_h5) -> None:
        conn, h5_path = synthetic_db_with_h5
        populate_moo_quality(conn, "run1", h5_path, reference_set=None)
        populate_moo_quality(conn, "run1", h5_path, reference_set=None)
        conn.commit()
        # Brak duplikatów dzięki PK + INSERT OR REPLACE.
        cnt = conn.execute(
            """
            SELECT COUNT(*) FROM optimization_generation_stats
            WHERE source_name = 'moo_quality'
              AND metric_name = 'spread'
              AND generation = 0
            """
        ).fetchone()[0]
        assert cnt <= 1

    def test_emits_front_size_per_gen(self, synthetic_db_with_h5) -> None:
        # Kamień 2: front_size powinno być emitowane dla każdej generacji
        # zawsze (nawet bez reference_set).
        conn, h5_path = synthetic_db_with_h5
        populate_moo_quality(conn, "run1", h5_path, reference_set=None)
        conn.commit()
        rows = conn.execute(
            """
            SELECT generation, metric_value
            FROM optimization_generation_stats
            WHERE source_name = 'moo_quality' AND metric_name = 'front_size'
            ORDER BY generation
            """
        ).fetchall()
        # 3 generacje w synthetic h5 → 3 wpisy front_size.
        assert len(rows) == 3
        # Last gen — manualnie zaprojektowane: 4 niezdominowane (5-ty
        # to (10,10,10) → zdominowany przez (1,1,0) etc.). Jest też
        # (0.5,0.5,0.5) który dominuje (1,1,0), (1,0,1), (0,1,1)? Sprawdź:
        # (0.5,0.5,0.5) vs (1,1,0): leq(<=)?  (0.5≤1, 0.5≤1, 0.5≤0)? NIE.
        # (0.5,0.5,0.5) vs (0,1,1): (0.5≤0)? NIE.
        # Czyli wszystkie 4 niezdominowane (1,1,0), (1,0,1), (0,1,1),
        # (0.5,0.5,0.5). Front_size last gen = 4.
        last_gen_size = rows[-1][1]
        assert last_gen_size == 4.0

    def test_hv_normalized_emitted_with_ideal_point(self, synthetic_db_with_h5) -> None:
        # Kamień 2: HV_norm = HV / Π(r* − ideal). Wymaga ideal_point.
        conn, h5_path = synthetic_db_with_h5
        ref_point = np.array([2.0, 2.0, 2.0])
        ideal = np.array([0.0, 0.0, 0.0])
        populate_moo_quality(
            conn, "run1", h5_path,
            reference_point=ref_point, ideal_point=ideal,
        )
        conn.commit()
        rows = conn.execute(
            """
            SELECT metric_value
            FROM optimization_generation_stats
            WHERE source_name = 'moo_quality'
              AND metric_name = 'hypervolume_normalized'
            """
        ).fetchall()
        assert len(rows) >= 1
        for (v,) in rows:
            assert 0.0 <= v <= 1.0
