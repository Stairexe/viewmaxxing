import numpy as np
import pytest

from vf.trajectories import PRESETS, make_trajectory, valid_frame_count

K = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])


def centres_and_forward(w2c):
    c2w = np.linalg.inv(w2c)
    return c2w[:, :3, 3], c2w[:, :3, 2]


def test_frame_counts_snap_to_lyra_chunks():
    assert valid_frame_count(81) == 81
    assert valid_frame_count(100) == 161
    assert valid_frame_count(1) == 81


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_is_valid(preset):
    tr = make_trajectory(preset, 161, K, 1280, 720)
    w2c = tr["w2c"]
    assert w2c.shape == (161, 4, 4) and tr["intrinsics"].shape == (161, 3, 3)
    assert np.allclose(w2c[0], np.eye(4), atol=1e-6)  # starts at the anchor camera
    R = w2c[:, :3, :3]
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-5)  # proper rotations
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-5)
    C, _ = centres_and_forward(w2c)
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    assert steps.max() < 0.08  # smooth: no jumps between frames
    assert np.all(np.abs(C[:, 1]) < 1e-6)  # stays at eye height


def test_walk_forward_moves_forward():
    C, _ = centres_and_forward(make_trajectory("walk_forward", 161, K, 1280, 720, distance=1.5)["w2c"])
    assert C[-1, 2] == pytest.approx(1.5, abs=1e-3)


def test_turn_around_full_circle_back_home():
    C, F = centres_and_forward(make_trajectory("turn_around", 241, K, 1280, 720, distance=0.5)["w2c"])
    assert np.allclose(C[-1], 0, atol=1e-6)
    assert F[len(F) // 2, 2] == pytest.approx(-1.0, abs=0.05)  # halfway: looking backwards


def test_orbit_keeps_looking_at_centre():
    depth = 2.0
    C, F = centres_and_forward(make_trajectory("orbit", 161, K, 1280, 720, degrees=60, orbit_depth=depth)["w2c"])
    to_centre = np.array([0, 0, depth]) - C
    to_centre /= np.linalg.norm(to_centre, axis=1, keepdims=True)
    assert np.allclose(np.sum(to_centre * F, axis=1), 1.0, atol=1e-6)
