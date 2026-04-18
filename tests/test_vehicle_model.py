import numpy as np

from mlpa.track_model import TrackGeometry, build_track_from_position
from mlpa.vehicle_model import (
    VehicleParams,
    backward_pass,
    forward_pass,
    max_corner_speed,
    solve_envelope,
)
from mlpa.envelope import compare_lap_to_envelope


def _straight_track(length_m: float = 1000.0, step_m: float = 5.0) -> TrackGeometry:
    d = np.arange(0.0, length_m + step_m, step_m)
    return TrackGeometry(
        distance=d, x=d.copy(), y=np.zeros_like(d),
        heading=np.zeros_like(d), curvature=np.zeros_like(d),
    )


def _circle_track(radius_m: float = 100.0, step_m: float = 2.0) -> TrackGeometry:
    circumference = 2 * np.pi * radius_m
    d = np.arange(0.0, circumference + step_m, step_m)
    theta = d / radius_m
    x = radius_m * np.sin(theta)
    y = radius_m * (1.0 - np.cos(theta))
    return build_track_from_position(d, x, y)


def _dragstrip_with_hairpin(
    straight_m: float = 600.0, hairpin_r: float = 40.0, step_m: float = 2.0,
) -> TrackGeometry:
    d_s = np.arange(0.0, straight_m, step_m)
    d_c = np.arange(0.0, np.pi * hairpin_r, step_m)
    x, y = [], []
    x.extend(d_s);  y.extend(np.zeros_like(d_s))
    theta = d_c / hairpin_r
    x.extend(straight_m + hairpin_r * np.sin(theta))
    y.extend(-hairpin_r * (1 - np.cos(theta)))
    for s in d_s:
        x.append(straight_m - s); y.append(-2 * hairpin_r)
    theta = d_c / hairpin_r
    x.extend(-hairpin_r * np.sin(theta))
    y.extend(-2 * hairpin_r + hairpin_r * (1 - np.cos(theta)))
    x = np.array(x); y = np.array(y)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return build_track_from_position(d, x, y)


def _monza_style_chicane_track(
    straight_before_m: float = 500.0,
    straight_after_m: float = 500.0,
    chicane_radius_m: float = 30.0,
    native_sample_m: float = 3.0,
) -> tuple[TrackGeometry, float]:
    """Synthetic Monza-style right-left chicane plus flanking straights.

    Sampled at ~3m spacing to match FastF1's actual native position rate.
    Returns (track, expected_apex_kappa).
    """
    R = chicane_radius_m
    d_s1 = np.arange(0.0, straight_before_m, native_sample_m)
    arc_len = np.pi / 2 * R
    d_arc = np.arange(0.0, arc_len, native_sample_m)
    d_s2 = np.arange(0.0, straight_after_m, native_sample_m)

    x, y = [], []
    # Straight 1
    x.extend(d_s1);  y.extend(np.zeros_like(d_s1))
    # Right-hand quarter circle
    theta = d_arc / R
    x_end = straight_before_m
    x.extend(x_end + R * np.sin(theta))
    y.extend(-R * (1 - np.cos(theta)))
    # Left-hand quarter circle to restore heading
    x_start = x[-1]; y_start = y[-1]
    theta = d_arc / R
    x.extend(x_start + R * (1 - np.cos(theta)))
    y.extend(y_start - R * np.sin(theta))
    # Straight 2
    dx_straight = x[-1] - x[-2]
    dy_straight = y[-1] - y[-2]
    heading = np.arctan2(dy_straight, dx_straight)
    for s in d_s2:
        x.append(x[-1] + native_sample_m * np.cos(heading))
        y.append(y[-1] + native_sample_m * np.sin(heading))

    x = np.array(x); y = np.array(y)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    track = build_track_from_position(d, x, y)
    return track, 1.0 / R


# ---- unit tests ----

def test_straight_has_no_cornering_limit():
    track = _straight_track()
    params = VehicleParams()
    v = max_corner_speed(track.curvature, params)
    assert np.all(v > 100.0 / 3.6)


def test_forward_pass_accelerates_on_a_straight():
    track = _straight_track()
    params = VehicleParams()
    v_corner = max_corner_speed(track.curvature, params)
    v = forward_pass(track.distance, track.curvature, v_corner, params, v_start_ms=50.0 / 3.6)
    assert v[-1] > v[0] and v[-1] < 400.0 / 3.6


def test_backward_pass_respects_corner_cap():
    track = _circle_track(radius_m=80.0)
    params = VehicleParams()
    v_corner = max_corner_speed(track.curvature, params)
    v_back = backward_pass(track.distance, track.curvature, v_corner, params, v_end_ms=40.0 / 3.6)
    assert np.all(v_back <= v_corner + 1e-6)


def test_envelope_monotone_in_passes():
    track = _circle_track(radius_m=120.0)
    params = VehicleParams()
    sol = solve_envelope(track, params)
    assert np.all(sol.v_model <= sol.v_corner + 1e-6)
    assert np.all(sol.v_model <= sol.v_forward + 1e-6)
    assert np.all(sol.v_model <= sol.v_backward + 1e-6)
    assert sol.lap_time_s > 0.0


def test_downforce_raises_corner_speed():
    kappa = np.array([1.0 / 80.0])
    low_df = VehicleParams(cla_m2=0.5)
    high_df = VehicleParams(cla_m2=4.0)
    assert max_corner_speed(kappa, high_df)[0] > max_corner_speed(kappa, low_df)[0]


def test_radius_ordering_matches_physics():
    params = VehicleParams()
    v_tight = max_corner_speed(np.array([1.0 / 40.0]), params)[0]
    v_open = max_corner_speed(np.array([1.0 / 200.0]), params)[0]
    assert v_open > v_tight


def test_solve_envelope_is_periodic_on_closed_loop():
    track = _dragstrip_with_hairpin(600.0, 40.0, 2.0)
    sol = solve_envelope(track, VehicleParams())
    assert sol.periodic_residual_ms < 0.5, (
        f"residual={sol.periodic_residual_ms:.3f} m/s"
    )


def test_solve_envelope_reaches_top_speed_on_long_straight():
    track = _dragstrip_with_hairpin(1200.0, 40.0, 2.0)
    sol = solve_envelope(track, VehicleParams())
    assert sol.v_model.max() * 3.6 > 250.0


def test_chicane_apex_curvature_is_resolved():
    """Would have caught the earlier peak-wipeout bug.

    A 30 m radius chicane should return apex kappa within 30% of 1/30.
    """
    track, expected_kappa = _monza_style_chicane_track(chicane_radius_m=30.0)
    peak_kappa = np.max(np.abs(track.curvature))
    assert peak_kappa > 0.7 * expected_kappa, (
        f"apex resolution lost: peak={peak_kappa:.4f}, expected~{expected_kappa:.4f}"
    )
    assert peak_kappa < 1.3 * expected_kappa, (
        f"apex over-sharp: peak={peak_kappa:.4f}, expected~{expected_kappa:.4f}"
    )


def test_envelope_comparison_at_98pct_is_plausible():
    """A driver at 98% of the envelope should lose ~2% of lap time, positive."""
    track = _dragstrip_with_hairpin(800.0, 50.0, 2.0)
    params = VehicleParams()
    sol = solve_envelope(track, params)
    driver_speed = sol.v_model * 0.98
    cmp = compare_lap_to_envelope(track, sol, driver_speed, params)
    assert cmp.unused_time_s > 0.0, f"unused_time={cmp.unused_time_s:.3f}"
    assert cmp.unused_time_s < 0.05 * cmp.lap_time_model_s, (
        f"unused_time={cmp.unused_time_s:.3f} vs lap_time={cmp.lap_time_model_s:.3f}"
    )
    assert cmp.grip_utilisation.max() < 1.15, (
        f"peak_util={cmp.grip_utilisation.max():.2f}"
    )
