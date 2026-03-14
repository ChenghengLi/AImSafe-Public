"""Tests for engine/ergonomic_scorer.py"""
from engine.ergonomic_scorer import ErgonomicScorer


class TestErgonomicScorer:
    def test_upright_posture_low_score(self):
        scorer = ErgonomicScorer()
        angles = {"back": 5, "shoulder_tilt": 2, "left_knee": 175, "right_knee": 175,
                  "left_arm_extension": 30, "right_arm_extension": 30}
        score = scorer.score(angles, head_pitch=5)
        assert 1 <= score <= 4  # negligible to low

    def test_bent_posture_high_score(self):
        scorer = ErgonomicScorer()
        angles = {"back": 55, "shoulder_tilt": 10, "left_knee": 100, "right_knee": 100,
                  "left_arm_extension": 150, "right_arm_extension": 150}
        score = scorer.score(angles, head_pitch=30)
        assert score >= 8  # medium to high

    def test_score_clamped_to_15(self):
        scorer = ErgonomicScorer()
        angles = {"back": 90, "shoulder_tilt": 20, "left_knee": 60, "right_knee": 60,
                  "left_arm_extension": 180, "right_arm_extension": 180}
        score = scorer.score(angles, head_pitch=45)
        assert score <= 15

    def test_score_minimum_1(self):
        scorer = ErgonomicScorer()
        angles = {"back": 0, "shoulder_tilt": 0, "left_knee": 180, "right_knee": 180,
                  "left_arm_extension": 0, "right_arm_extension": 0}
        score = scorer.score(angles, head_pitch=0)
        assert score >= 1

    def test_missing_angles_uses_defaults(self):
        scorer = ErgonomicScorer()
        score = scorer.score({}, head_pitch=0)
        assert 1 <= score <= 15
