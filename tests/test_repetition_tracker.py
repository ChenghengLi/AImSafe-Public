"""
Tests for engine/repetition_tracker.py — cyclical motion counting.
"""

from engine.repetition_tracker import RepetitionTracker


class TestRepetitionTracker:
    def test_no_cycles_at_start(self):
        rt = RepetitionTracker()
        count = rt.update(arm_angle=90, back_angle=10)
        assert count == 0

    def test_arm_cycle_counted(self):
        rt = RepetitionTracker()
        # Extend arm past 120°
        rt.update(arm_angle=130, back_angle=10)
        # Return below 90° → completes 1 cycle
        count = rt.update(arm_angle=80, back_angle=10)
        assert count == 1

    def test_back_cycle_counted(self):
        rt = RepetitionTracker()
        # Bend past 30°
        rt.update(arm_angle=90, back_angle=35)
        # Return below 15° → completes 1 cycle
        count = rt.update(arm_angle=90, back_angle=10)
        assert count == 1

    def test_multiple_arm_cycles(self):
        rt = RepetitionTracker()
        for _ in range(5):
            rt.update(arm_angle=130, back_angle=10)
            rt.update(arm_angle=80, back_angle=10)
        count = rt.update(arm_angle=90, back_angle=10)
        assert count == 5

    def test_combined_arm_and_back_cycles(self):
        rt = RepetitionTracker()
        # 3 arm cycles
        for _ in range(3):
            rt.update(arm_angle=130, back_angle=10)
            rt.update(arm_angle=80, back_angle=10)
        # 2 back cycles
        for _ in range(2):
            rt.update(arm_angle=90, back_angle=35)
            rt.update(arm_angle=90, back_angle=10)
        count = rt.update(arm_angle=90, back_angle=10)
        assert count == 5

    def test_no_cycle_without_full_extension(self):
        rt = RepetitionTracker()
        # Arm at 110° (below 120° threshold) — should not trigger
        rt.update(arm_angle=110, back_angle=10)
        count = rt.update(arm_angle=80, back_angle=10)
        assert count == 0

    def test_no_cycle_without_full_return(self):
        rt = RepetitionTracker()
        # Extend arm
        rt.update(arm_angle=130, back_angle=10)
        # Return to 95° (above 90° threshold) — cycle not completed
        count = rt.update(arm_angle=95, back_angle=10)
        assert count == 0
