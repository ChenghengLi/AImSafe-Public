"""
Tests for engine/fatigue_tracker.py — fatigue scoring with PERCLOS, yawning, and smoothed transitions.
"""

from engine.fatigue_tracker import FatigueTracker


class TestFatigueTracker:
    def test_no_fatigue_when_normal(self):
        ft = FatigueTracker()
        score = ft.update(blink_rate=18, microsleep=False, head_pitch=5, centroid=(0.5, 0.5))
        assert score == 0.0

    def test_perclos_adds_score(self):
        ft = FatigueTracker()
        # perclos=0.30 → excess=0.10 → ~5.0 raw → smoothed upward
        score = ft.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                          perclos=0.30, drowsy=True)
        assert score > 0

    def test_perclos_below_threshold_no_contribution(self):
        """PERCLOS < 0.20 should not contribute (calibrated for 30s window)."""
        ft = FatigueTracker()
        score = ft.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                          perclos=0.15, drowsy=False)
        assert score == 0.0

    def test_perclos_contribution_increases(self):
        ft1 = FatigueTracker()
        s1 = ft1.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                        perclos=0.25, drowsy=True)
        ft2 = FatigueTracker()
        s2 = ft2.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                        perclos=0.45, drowsy=True)
        assert s2 > s1

    def test_drowsy_accumulates_over_time(self):
        ft = FatigueTracker()
        for _ in range(10):
            score = ft.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                              perclos=0.25, drowsy=True)
        assert score > 3

    def test_drowsy_decays_when_alert(self):
        ft = FatigueTracker()
        for _ in range(20):
            ft.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                      perclos=0.30, drowsy=True)
        high = ft.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                         perclos=0.30, drowsy=True)
        for _ in range(50):
            ft.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                      perclos=0.05, drowsy=False)
        low = ft.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5),
                        perclos=0.05, drowsy=False)
        assert low < high

    def test_microsleep_builds_gradually(self):
        ft = FatigueTracker()
        s1 = ft.update(blink_rate=18, microsleep=True, head_pitch=0, centroid=(0.5, 0.5))
        assert s1 > 0
        for _ in range(10):
            s1 = ft.update(blink_rate=18, microsleep=True, head_pitch=0, centroid=(0.5, 0.5))
        assert s1 > 5

    def test_yawn_adds_fatigue(self):
        """Yawning (bostezos) should contribute to fatigue score."""
        ft = FatigueTracker()
        score_no_yawn = ft.update(blink_rate=18, microsleep=False, head_pitch=0,
                                  centroid=(0.5, 0.5), yawn_detected=False)
        ft2 = FatigueTracker()
        score_yawn = ft2.update(blink_rate=18, microsleep=False, head_pitch=0,
                                centroid=(0.5, 0.5), yawn_detected=True)
        assert score_yawn > score_no_yawn

    def test_yawn_accumulates(self):
        """Frequent yawning should build up fatigue."""
        ft = FatigueTracker()
        for _ in range(5):
            score = ft.update(blink_rate=18, microsleep=False, head_pitch=0,
                              centroid=(0.5, 0.5), yawn_detected=True)
        assert score > 2

    def test_stress_adds_small_contribution(self):
        ft = FatigueTracker()
        score = ft.update(blink_rate=18, microsleep=False, head_pitch=0,
                          centroid=(0.5, 0.5), stress_level=0.8)
        assert score > 0

    def test_head_nod_adds_score(self):
        ft = FatigueTracker()
        score = ft.update(blink_rate=18, microsleep=False, head_pitch=35, centroid=(0.5, 0.5))
        assert score > 0

    def test_combined_signals_add_up(self):
        ft = FatigueTracker()
        for _ in range(10):
            score = ft.update(blink_rate=18, microsleep=True, head_pitch=35, centroid=(0.5, 0.5),
                              perclos=0.35, drowsy=True, yawn_detected=True, stress_level=0.5)
        assert score > 10

    def test_score_never_exceeds_100(self):
        ft = FatigueTracker()
        for _ in range(100):
            score = ft.update(blink_rate=100, microsleep=True, head_pitch=90, centroid=(0.5, 0.5),
                              perclos=0.60, drowsy=True, yawn_detected=True, stress_level=1.0)
        assert score <= 100

    def test_backward_compatible_without_new_params(self):
        """Fatigue tracker still works when new params not provided."""
        ft = FatigueTracker()
        score = ft.update(blink_rate=18, microsleep=False, head_pitch=0, centroid=(0.5, 0.5))
        assert score == 0.0
