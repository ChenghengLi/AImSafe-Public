"""
Simplified REBA (Rapid Entire Body Assessment) ergonomic scorer.
Takes pose angles and head pitch, returns a score from 1-15 with risk level.
"""


class ErgonomicScorer:
    """Computes a simplified REBA ergonomic risk score."""

    RISK_LEVELS = {
        (1, 3): "Negligible",
        (4, 7): "Low",
        (8, 10): "Medium",
        (11, 15): "High",
    }

    @staticmethod
    def _trunk_score(back_angle: float) -> int:
        """Trunk score from back angle (degrees from vertical)."""
        a = abs(back_angle)
        if a <= 10:
            return 1
        elif a <= 20:
            return 2
        elif a <= 40:
            return 3
        elif a <= 60:
            return 4
        else:
            return 5

    @staticmethod
    def _neck_score(head_pitch: float) -> int:
        """Neck score from head pitch angle."""
        a = abs(head_pitch)
        if a <= 10:
            return 1
        elif a <= 20:
            return 2
        else:
            return 3

    @staticmethod
    def _legs_score(left_knee: float, right_knee: float) -> int:
        """Legs score from average knee angle."""
        avg = (left_knee + right_knee) / 2
        if avg > 160:
            return 1  # standing
        elif avg >= 130:
            return 2  # slight bend
        else:
            return 3  # deep bend

    @staticmethod
    def _upper_arm_score(left_arm: float, right_arm: float) -> int:
        """Upper arm score from max arm extension angle."""
        max_arm = max(left_arm, right_arm)
        if max_arm < 60:
            return 1
        elif max_arm <= 100:
            return 2
        elif max_arm <= 140:
            return 3
        else:
            return 4

    def score(self, angles: dict[str, float], head_pitch: float) -> int:
        """
        Compute simplified REBA score (1-15).

        Parameters
        ----------
        angles : dict with keys: back, shoulder_tilt, left_knee, right_knee,
                 left_arm_extension, right_arm_extension
        head_pitch : head pitch angle in degrees

        Returns
        -------
        int : REBA score clamped to 1-15
        """
        back = abs(angles.get("back", 0))
        left_knee = angles.get("left_knee", 180)
        right_knee = angles.get("right_knee", 180)
        left_arm = angles.get("left_arm_extension", 0)
        right_arm = angles.get("right_arm_extension", 0)

        trunk = self._trunk_score(back)
        neck = self._neck_score(head_pitch)
        legs = self._legs_score(left_knee, right_knee)
        upper_arm = self._upper_arm_score(left_arm, right_arm)

        # Table A: trunk + neck + legs → score_a (sum/3 scaled to 1-8)
        raw_a = trunk + neck + legs  # range 3-11
        score_a = max(1, min(8, round((raw_a - 3) / (11 - 3) * 7 + 1)))

        # Table B: upper arm → score_b (1-4)
        score_b = upper_arm

        # Final REBA = score_a + score_b, clamped 1-15
        reba = max(1, min(15, score_a + score_b))
        return reba

    @classmethod
    def risk_level(cls, score: int) -> str:
        """Return risk level string for a given REBA score."""
        for (lo, hi), level in cls.RISK_LEVELS.items():
            if lo <= score <= hi:
                return level
        return "High"
