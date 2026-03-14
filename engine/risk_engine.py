"""
Central risk engine — evaluates all safety rules and produces a fused WorkerState.
This is the brain of the system: vision data in → safety assessment out.
"""

import time
import config
from vision.pose_analyzer import PoseResult
from vision.face_analyzer import FaceResult
from engine.worker_state import WorkerState, RiskLevel, ZoneStatus, Alert
from engine.zone_manager import ZoneManager
from engine.fatigue_tracker import FatigueTracker
from engine.repetition_tracker import RepetitionTracker
from engine.ergonomic_scorer import ErgonomicScorer


class RiskEngine:
    """Evaluates safety rules and fuses all signals into a WorkerState."""

    def __init__(self, zone_manager: ZoneManager):
        self.zone_manager = zone_manager
        self.fatigue_tracker = FatigueTracker()
        self.repetition_tracker = RepetitionTracker()
        self.ergonomic_scorer = ErgonomicScorer()
        self._alert_timestamps: dict[str, float] = {}

    def evaluate(
        self,
        pose: PoseResult,
        face: FaceResult,
        heart_rate: float | None = None,
        hrv: float | None = None,
    ) -> WorkerState:
        """
        Run all rules against current frame data and return a complete WorkerState.
        """
        state = WorkerState(timestamp=time.time())

        # ── Pose risk ──
        pose_score = 0.0
        if pose.visible:
            state.angles = pose.angles
            state.pose_flags = pose.flags
            state.body_centroid = pose.body_centroid

            # Graduated back angle risk
            back_angle = pose.angles.get("back", 0)
            if back_angle > 30:
                # 30°→0, 40°→20, 55°→50, 70°→80 (scaled)
                pose_score = max(pose_score, min((back_angle - 30) * 2, 80))

            if pose.flags.get("unsafe_lift"):
                pose_score = max(pose_score, 80)
                self._maybe_alert(state, "unsafe_lift", "WARNING",
                                  f"Unsafe lifting posture — back angle {back_angle:.0f}°")

        state.pose_risk = pose_score

        # REBA ergonomic scoring
        if pose.visible:
            head_p = face.head_pitch if face.visible else 0.0
            state.reba_score = self.ergonomic_scorer.score(pose.angles, head_p)
            if state.reba_score >= 11:
                self._maybe_alert(state, "ergonomic", "WARNING",
                                  f"Very high ergonomic risk (REBA: {state.reba_score})")
            elif state.reba_score >= 8:
                self._maybe_alert(state, "ergonomic", "INFO",
                                  f"Medium ergonomic risk (REBA: {state.reba_score})")

        # ── Zone proximity ──
        proximity_score = 0.0
        if pose.visible:
            zone_status, zone_name = self.zone_manager.check(*pose.body_centroid)
            state.zone_status = zone_status
            state.current_zone_name = zone_name

            if zone_status == ZoneStatus.RESTRICTED:
                proximity_score = 40
                self._maybe_alert(state, "proximity", "INFO",
                                  f"Worker entered restricted zone: {zone_name}")
            elif zone_status == ZoneStatus.MACHINE_PROXIMITY:
                proximity_score = 20
                self._maybe_alert(state, "proximity", "INFO",
                                  f"Worker near machinery: {zone_name}")

        # ── Fatigue ──
        fatigue_score = 0.0
        if face.visible:
            state.blink_rate = face.blink_rate
            state.microsleep = face.microsleep
            state.head_pitch = face.head_pitch
            state.perclos = getattr(face, 'perclos', 0.0)
            state.drowsy = getattr(face, 'drowsy', False)

            fatigue_score = self.fatigue_tracker.update(
                blink_rate=face.blink_rate,
                microsleep=face.microsleep,
                head_pitch=face.head_pitch,
                centroid=pose.body_centroid if pose.visible else (0.5, 0.5),
                perclos=state.perclos,
                drowsy=state.drowsy,
                yawn_detected=getattr(face, 'yawn_detected', False),
                stress_level=getattr(face, 'stress_level', 0.0),
            )

            # PERCLOS > 50% = severe incident
            if state.perclos > 0.50:
                self._maybe_alert(state, "fatigue", "CRITICAL",
                                  f"SEVERE DROWSINESS — PERCLOS {state.perclos:.0%}! Eyes closed most of the time!")
            elif face.microsleep:
                self._maybe_alert(state, "fatigue", "CRITICAL",
                                  "Microsleep detected — eyes closed too long!")
            elif state.drowsy:
                self._maybe_alert(state, "fatigue", "WARNING",
                                  f"Drowsiness detected — PERCLOS {state.perclos:.0%}")
            elif fatigue_score > 65:
                self._maybe_alert(state, "fatigue", "WARNING",
                                  f"High fatigue detected (score: {fatigue_score:.0f})")

        if face.visible:
            state.stress_level = getattr(face, 'stress_level', 0.0)
            state.yawn_detected = getattr(face, 'yawn_detected', False)
            state.gaze_focus = getattr(face, 'gaze_focus', 1.0)

        state.fatigue_score = fatigue_score

        # ── Repetition ──
        rep_score = 0.0
        if pose.visible:
            arm_angle = max(
                pose.angles.get("left_arm_extension", 0),
                pose.angles.get("right_arm_extension", 0),
            )
            back_angle = pose.angles.get("back", 0)
            rep_count = self.repetition_tracker.update(arm_angle, back_angle)
            state.repetition_count = rep_count

            if rep_count > config.REPETITION_LIMIT:
                rep_score = 50
                self._maybe_alert(state, "repetition", "INFO",
                                  f"High repetition count: {rep_count} motions in {config.REPETITION_WINDOW // 60} min")

        # ── Wearable ──
        state.heart_rate = heart_rate
        state.hrv = hrv

        # ── Composite score ──
        weighted = (
            pose_score * config.RISK_WEIGHTS["pose"]
            + fatigue_score * config.RISK_WEIGHTS["fatigue"]
            + proximity_score * config.RISK_WEIGHTS["proximity"]
            + rep_score * config.RISK_WEIGHTS["repetition"]
        )
        # Safety score = inverse of risk (100 = perfectly safe)
        state.overall_safety_score = max(0, 100 - weighted)
        state.risk_level = self._score_to_level(state.overall_safety_score)

        return state

    def _maybe_alert(self, state: WorkerState, rule: str, severity: str, message: str):
        """Add alert if cooldown has elapsed."""
        now = time.time()
        cooldown = config.ALERT_COOLDOWN.get(rule, 10)
        last = self._alert_timestamps.get(rule, 0)
        if now - last >= cooldown:
            state.active_alerts.append(Alert(rule_name=rule, severity=severity, message=message))
            self._alert_timestamps[rule] = now

    @staticmethod
    def _score_to_level(safety_score: float) -> RiskLevel:
        if safety_score >= 70:
            return RiskLevel.SAFE
        elif safety_score >= 40:
            return RiskLevel.CAUTION
        elif safety_score >= 20:
            return RiskLevel.WARNING
        return RiskLevel.DANGER
