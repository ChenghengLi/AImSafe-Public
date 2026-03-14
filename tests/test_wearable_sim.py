"""
Tests for integration/wearable_sim.py — simulated heart rate and HRV data.
"""

from integration.wearable_sim import WearableSim


class TestWearableSim:
    def test_initial_values_near_baseline(self):
        ws = WearableSim(base_hr=72, base_hrv=50)
        hr, hrv = ws.update()
        assert 50 <= hr <= 100
        assert 20 <= hrv <= 80

    def test_returns_tuple(self):
        ws = WearableSim()
        result = ws.update()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hr_within_bounds(self):
        ws = WearableSim()
        for _ in range(100):
            hr, hrv = ws.update()
            assert 50 <= hr <= 180
            assert 10 <= hrv <= 100

    def test_stress_increases_hr(self):
        ws = WearableSim(base_hr=72)
        # Collect calm HR
        calm_hrs = []
        for _ in range(50):
            hr, _ = ws.update()
            calm_hrs.append(hr)

        # Set max stress and collect stressed HR
        ws.set_stress(1.0)
        stressed_hrs = []
        for _ in range(50):
            hr, _ = ws.update()
            stressed_hrs.append(hr)

        # Average stressed HR should be higher
        assert sum(stressed_hrs) / len(stressed_hrs) > sum(calm_hrs) / len(calm_hrs)

    def test_stress_decreases_hrv(self):
        ws = WearableSim(base_hrv=50)
        # Collect calm HRV
        calm_hrvs = []
        for _ in range(50):
            _, hrv = ws.update()
            calm_hrvs.append(hrv)

        # Set max stress
        ws.set_stress(1.0)
        stressed_hrvs = []
        for _ in range(50):
            _, hrv = ws.update()
            stressed_hrvs.append(hrv)

        assert sum(stressed_hrvs) / len(stressed_hrvs) < sum(calm_hrvs) / len(calm_hrvs)

    def test_stress_clamped(self):
        ws = WearableSim()
        ws.set_stress(-5.0)
        assert ws._stress_factor == 0.0
        ws.set_stress(10.0)
        assert ws._stress_factor == 1.0

    def test_values_are_rounded(self):
        ws = WearableSim()
        hr, hrv = ws.update()
        # round(x, 1) should give at most 1 decimal place
        assert hr == round(hr, 1)
        assert hrv == round(hrv, 1)
