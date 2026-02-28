"""
EnhancedTracking.py - All-in-one tracking improvements for CourtSense

Improvements implemented:
1. Kalman filter for ball tracking (smooth, handle occlusion)
2. Advanced shot classification (from ef.py trajectory analysis)
3. Ball speed in real-world units (km/h)
4. Rally detection & point scoring
5. Ball bounce detection
6. Swing phase detection (preparation, swing, follow-through)
7. Multi-scale ball detection for robustness
8. Court line detection (Hough transforms)
9. Improved player re-identification (deep features + spatial)
"""

import math
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 1. KALMAN FILTER FOR BALL TRACKING
# ============================================================

class BallKalmanFilter:
    """Kalman filter for 2D ball tracking with constant velocity model.
    
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        # Process noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[2, 2] = 5e-2
        self.kf.processNoiseCov[3, 3] = 5e-2
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        # Initial state covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False
        self.frames_since_detection = 0
        self.max_predict_frames = 15  # predict up to 15 frames without detection

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update filter with a new detection."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
            self.frames_since_detection = 0
            return x, y

        self.kf.predict()
        corrected = self.kf.correct(measurement)
        self.frames_since_detection = 0
        return float(np.asarray(corrected[0]).flat[0]), float(np.asarray(corrected[1]).flat[0])

    def predict(self) -> Optional[Tuple[float, float]]:
        """Predict next position without a measurement (for occlusion)."""
        if not self.initialized:
            return None
        self.frames_since_detection += 1
        if self.frames_since_detection > self.max_predict_frames:
            return None  # lost track
        predicted = self.kf.predict()
        return float(np.asarray(predicted[0]).flat[0]), float(np.asarray(predicted[1]).flat[0])

    def get_velocity(self) -> Tuple[float, float]:
        """Return current estimated velocity (vx, vy) in pixels/frame."""
        if not self.initialized:
            return 0.0, 0.0
        state = self.kf.statePost
        return float(state[2]), float(state[3])

    @property
    def is_tracking(self) -> bool:
        return self.initialized and self.frames_since_detection <= self.max_predict_frames


# ============================================================
# 2. ADVANCED SHOT CLASSIFIER (integrated from ef.py)
# ============================================================

def classify_shot_advanced(
    past_ball_pos: List[List],
    court_width: int = 640,
    court_height: int = 360,
    previous_shot: Optional[List] = None,
) -> Dict[str, Any]:
    """Advanced shot classification using trajectory analysis.
    
    Returns dict with: direction, height, style, confidence, wall_bounces, speed_kmh
    """
    result = {
        "direction": "unknown",
        "height": "unknown",
        "style": "unknown",
        "confidence": 0.0,
        "wall_bounces": 0,
        "display": "unknown",
    }

    if len(past_ball_pos) < 4:
        return result

    trajectory_length = min(20, len(past_ball_pos))
    trajectory = past_ball_pos[-trajectory_length:]

    try:
        metrics = _analyze_trajectory(trajectory, court_width, court_height)

        h_disp = metrics["horizontal_displacement"]
        v_disp = metrics["vertical_displacement"]
        dir_changes = metrics["direction_changes"]
        vel = metrics["velocity_profile"]
        coverage = metrics["court_coverage"]
        wall_bounces = metrics["wall_bounces"]

        # Direction
        direction = _classify_direction(h_disp, coverage, dir_changes)
        # Height
        height = _classify_height(v_disp, vel, trajectory, court_height)
        # Style
        style = _classify_style(vel, wall_bounces, dir_changes, trajectory)
        # Confidence
        confidence = _calc_confidence(metrics, previous_shot)

        result.update({
            "direction": direction,
            "height": height,
            "style": style,
            "confidence": confidence,
            "wall_bounces": wall_bounces,
            "display": f"{direction} {height}",
            "metrics": metrics,
        })
    except Exception:
        pass

    return result


def _analyze_trajectory(trajectory, court_width, court_height):
    """Full trajectory analysis with velocity, direction changes, coverage, bounces."""
    metrics = {}
    if len(trajectory) < 2:
        return {k: 0 for k in [
            "horizontal_displacement", "vertical_displacement",
            "direction_changes", "velocity_profile", "court_coverage", "wall_bounces"
        ]}

    start_x, start_y = trajectory[0][0], trajectory[0][1]
    end_x, end_y = trajectory[-1][0], trajectory[-1][1]
    metrics["horizontal_displacement"] = (end_x - start_x) / max(1, court_width)
    metrics["vertical_displacement"] = (end_y - start_y) / max(1, court_height)

    # Velocity analysis
    velocities = []
    for i in range(1, len(trajectory)):
        x1, y1 = trajectory[i - 1][0], trajectory[i - 1][1]
        x2, y2 = trajectory[i][0], trajectory[i][1]
        t1 = trajectory[i - 1][2] if len(trajectory[i - 1]) > 2 else i - 1
        t2 = trajectory[i][2] if len(trajectory[i]) > 2 else i
        dt = max(1, t2 - t1)
        v = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / dt
        velocities.append(v)

    if velocities:
        metrics["velocity_profile"] = {
            "avg_velocity": sum(velocities) / len(velocities),
            "max_velocity": max(velocities),
            "velocity_variance": float(np.var(velocities)),
            "velocity_trend": velocities[-1] - velocities[0] if len(velocities) > 1 else 0,
        }
    else:
        metrics["velocity_profile"] = {"avg_velocity": 0, "max_velocity": 0, "velocity_variance": 0, "velocity_trend": 0}

    # Direction changes
    dir_changes = 0
    last_dx = None
    last_dy = None
    for i in range(1, len(trajectory)):
        dx = 1 if trajectory[i][0] > trajectory[i - 1][0] else (-1 if trajectory[i][0] < trajectory[i - 1][0] else 0)
        dy = 1 if trajectory[i][1] > trajectory[i - 1][1] else (-1 if trajectory[i][1] < trajectory[i - 1][1] else 0)
        if last_dx is not None and dx != 0 and dx != last_dx:
            dir_changes += 1
        if last_dy is not None and dy != 0 and dy != last_dy:
            dir_changes += 1
        if dx != 0:
            last_dx = dx
        if dy != 0:
            last_dy = dy
    metrics["direction_changes"] = dir_changes

    # Court coverage
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    metrics["court_coverage"] = {
        "x_coverage": x_range / max(1, court_width),
        "y_coverage": y_range / max(1, court_height),
        "total_coverage": (x_range * y_range) / max(1, court_width * court_height),
    }

    # Wall bounce detection
    metrics["wall_bounces"] = _detect_wall_bounces(trajectory, court_width, court_height)

    return metrics


def _detect_wall_bounces(trajectory, court_width, court_height):
    """Detect wall bounces via direction reversal near walls."""
    bounces = 0
    wall_margin = 25

    for i in range(2, len(trajectory) - 1):
        x_prev, y_prev = trajectory[i - 1][0], trajectory[i - 1][1]
        x_curr, y_curr = trajectory[i][0], trajectory[i][1]
        x_next, y_next = trajectory[i + 1][0], trajectory[i + 1][1]

        near_wall = (
            x_curr < wall_margin
            or x_curr > court_width - wall_margin
            or y_curr < wall_margin
            or y_curr > court_height - wall_margin
        )
        if not near_wall:
            continue

        # Check direction reversal
        dx_before = x_curr - x_prev
        dx_after = x_next - x_curr
        dy_before = y_curr - y_prev
        dy_after = y_next - y_curr

        speed_before = math.sqrt(dx_before ** 2 + dy_before ** 2)
        if speed_before < 2.0:
            continue

        x_reversed = (dx_before * dx_after < 0) and abs(dx_before) > 2
        y_reversed = (dy_before * dy_after < 0) and abs(dy_before) > 2

        if x_reversed or y_reversed:
            bounces += 1

    return bounces


def _classify_direction(h_disp, coverage, dir_changes):
    abs_d = abs(h_disp)
    x_cov = coverage.get("x_coverage", 0)
    if abs_d > 0.4 and x_cov > 0.3:
        return "wide_crosscourt" if abs_d > 0.5 else "crosscourt"
    elif abs_d > 0.25 and dir_changes > 2:
        return "angled_crosscourt"
    elif abs_d > 0.15 and x_cov > 0.2:
        return "slight_crosscourt"
    elif abs_d < 0.08 and x_cov < 0.15:
        return "tight_straight"
    return "straight"


def _classify_height(v_disp, vel, trajectory, court_height):
    if len(trajectory) < 3:
        return "drive"
    max_h = min(p[1] for p in trajectory)
    traj_h = court_height - max_h
    avg_v = vel.get("avg_velocity", 0)
    v_var = vel.get("velocity_variance", 0)
    if traj_h > court_height * 0.4 and avg_v < 15:
        return "lob"
    elif traj_h < court_height * 0.15 and avg_v > 25:
        return "drive"
    elif v_var > 100:
        return "drop"
    return "drive"


def _classify_style(vel, wall_bounces, dir_changes, trajectory):
    avg_v = vel.get("avg_velocity", 0)
    v_var = vel.get("velocity_variance", 0)
    if wall_bounces > 2:
        return "boast"
    elif dir_changes > 4 and v_var > 50:
        return "nick"
    elif avg_v > 30:
        return "hard"
    elif avg_v < 10 and len(trajectory) > 10:
        return "soft"
    return "medium"


def _calc_confidence(metrics, previous_shot):
    confidence = 0.5
    vel = metrics.get("velocity_profile", {})
    v_var = vel.get("velocity_variance", 0)
    if v_var < 50:
        confidence += 0.2
    abs_d = abs(metrics.get("horizontal_displacement", 0))
    if abs_d > 0.3 or abs_d < 0.1:
        confidence += 0.2
    if previous_shot and metrics.get("direction_changes", 0) < 3:
        confidence += 0.1
    return min(1.0, max(0.1, confidence))


# ============================================================
# 3. BALL SPEED IN REAL-WORLD UNITS
# ============================================================

def calculate_ball_speed(
    past_ball_pos: List[List],
    homography: Any,
    reference_points_3d: List,
    video_fps: float = 30.0,
    pixel_to_3d_fn=None,
) -> Dict[str, float]:
    """Calculate ball speed in m/s and km/h using court coordinates.
    
    Uses the last few ball positions and homography to convert to real-world distance.
    """
    result = {"speed_ms": 0.0, "speed_kmh": 0.0, "speed_pixels": 0.0}

    if len(past_ball_pos) < 2 or pixel_to_3d_fn is None:
        return result

    try:
        # Use last 3 positions for smoothed speed
        n = min(5, len(past_ball_pos))
        recent = past_ball_pos[-n:]

        total_dist = 0.0
        total_frames = 0
        for i in range(1, len(recent)):
            p1 = pixel_to_3d_fn([recent[i - 1][0], recent[i - 1][1]], homography, reference_points_3d)
            p2 = pixel_to_3d_fn([recent[i][0], recent[i][1]], homography, reference_points_3d)
            # 3D distance in meters
            dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
            total_dist += dist
            # Frame difference
            f1 = recent[i - 1][2] if len(recent[i - 1]) > 2 else 0
            f2 = recent[i][2] if len(recent[i]) > 2 else 0
            total_frames += max(1, f2 - f1)

        # Pixel speed
        px_dist = math.sqrt(
            (recent[-1][0] - recent[0][0]) ** 2 + (recent[-1][1] - recent[0][1]) ** 2
        )
        result["speed_pixels"] = px_dist

        time_seconds = total_frames / max(1, video_fps)
        if time_seconds > 0:
            speed_ms = total_dist / time_seconds
            result["speed_ms"] = round(speed_ms, 1)
            result["speed_kmh"] = round(speed_ms * 3.6, 1)
    except Exception:
        pass

    return result


# ============================================================
# 4. RALLY DETECTION & POINT SCORING
# ============================================================

class RallyTracker:
    """Track rallies, detect starts/ends, count points."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.in_rally = False
        self.rally_start_frame = 0
        self.rally_shots = 0
        self.current_rally_shots: List[Dict] = []
        self.rallies: List[Dict] = []
        self.p1_points = 0
        self.p2_points = 0
        self.frames_no_ball = 0
        self.last_hitter = 0
        self.ball_near_front_wall_frames = 0
        self._rally_cooldown = 0  # prevent rapid start/stop

    def update(
        self,
        ball_detected: bool,
        ball_pos: Optional[Tuple[float, float]],
        players: Dict[int, Any],
        frame_num: int,
        court_height: int = 360,
    ) -> Dict[str, Any]:
        """Call each frame. Returns rally state info."""
        info = {
            "in_rally": self.in_rally,
            "rally_shots": self.rally_shots,
            "rally_duration_s": 0.0,
            "event": None,  # "rally_start", "rally_end", "point_p1", "point_p2"
            "p1_points": self.p1_points,
            "p2_points": self.p2_points,
            "total_rallies": len(self.rallies),
        }

        if self._rally_cooldown > 0:
            self._rally_cooldown -= 1

        if ball_detected and ball_pos is not None:
            self.frames_no_ball = 0

            # Detect if ball is near front wall (top of frame)
            if ball_pos[1] < court_height * 0.15:
                self.ball_near_front_wall_frames += 1
            else:
                self.ball_near_front_wall_frames = 0

            if not self.in_rally and self._rally_cooldown == 0:
                # Start rally when ball is active
                self.in_rally = True
                self.rally_start_frame = frame_num
                self.rally_shots = 0
                self.current_rally_shots = []
                info["event"] = "rally_start"

        else:
            self.frames_no_ball += 1
            self.ball_near_front_wall_frames = 0

            # End rally if ball lost for 1+ seconds
            if self.in_rally and self.frames_no_ball > self.fps * 1.0:
                self._end_rally(frame_num, info)

        if self.in_rally:
            duration = (frame_num - self.rally_start_frame) / max(1, self.fps)
            info["rally_duration_s"] = round(duration, 1)
            info["rally_shots"] = self.rally_shots
            info["in_rally"] = True

            # Safety: end extremely long rallies (> 2 min)
            if duration > 120:
                self._end_rally(frame_num, info)

        info["p1_points"] = self.p1_points
        info["p2_points"] = self.p2_points
        info["total_rallies"] = len(self.rallies)
        return info

    def register_shot(self, player_id: int, shot_info: Dict, frame_num: int):
        """Register that a player hit a shot during the current rally."""
        if self.in_rally:
            self.rally_shots += 1
            self.last_hitter = player_id
            self.current_rally_shots.append({
                "player": player_id,
                "frame": frame_num,
                "shot": shot_info,
            })

    def _end_rally(self, frame_num: int, info: Dict):
        """End the current rally and assign point."""
        self.in_rally = False
        self._rally_cooldown = int(self.fps * 0.5)  # half-second cooldown

        rally_data = {
            "start_frame": self.rally_start_frame,
            "end_frame": frame_num,
            "duration_s": round((frame_num - self.rally_start_frame) / max(1, self.fps), 1),
            "total_shots": self.rally_shots,
            "shots": self.current_rally_shots,
            "last_hitter": self.last_hitter,
        }
        self.rallies.append(rally_data)
        info["event"] = "rally_end"

        # Heuristic: point goes to the player who hit last
        # (the other player failed to return)
        if self.last_hitter == 1:
            self.p1_points += 1
            info["event"] = "point_p1"
        elif self.last_hitter == 2:
            self.p2_points += 1
            info["event"] = "point_p2"

    def get_summary(self) -> Dict:
        """Return full rally summary."""
        durations = [r["duration_s"] for r in self.rallies]
        shots_per_rally = [r["total_shots"] for r in self.rallies]
        return {
            "total_rallies": len(self.rallies),
            "p1_points": self.p1_points,
            "p2_points": self.p2_points,
            "avg_rally_duration": round(sum(durations) / max(1, len(durations)), 1),
            "avg_shots_per_rally": round(sum(shots_per_rally) / max(1, len(shots_per_rally)), 1),
            "longest_rally_s": max(durations) if durations else 0,
            "rallies": self.rallies,
        }


# ============================================================
# 5. BALL BOUNCE DETECTION
# ============================================================

def detect_bounces(
    past_ball_pos: List[List],
    court_width: int = 640,
    court_height: int = 360,
) -> List[Dict]:
    """Detect ball bounces from trajectory using direction reversal + wall proximity."""
    bounces = []
    if len(past_ball_pos) < 5:
        return bounces

    wall_margin = 25
    floor_zone = court_height * 0.85  # bottom 15% is floor area

    for i in range(2, len(past_ball_pos) - 2):
        x_prev, y_prev = past_ball_pos[i - 1][0], past_ball_pos[i - 1][1]
        x_curr, y_curr = past_ball_pos[i][0], past_ball_pos[i][1]
        x_next, y_next = past_ball_pos[i + 1][0], past_ball_pos[i + 1][1]

        dy_before = y_curr - y_prev
        dy_after = y_next - y_curr
        dx_before = x_curr - x_prev
        dx_after = x_next - x_curr

        speed = math.sqrt(dx_before ** 2 + dy_before ** 2)
        if speed < 2.0:
            continue

        frame_num = past_ball_pos[i][2] if len(past_ball_pos[i]) > 2 else i

        # Floor bounce: ball was going down, now going up
        if dy_before > 2 and dy_after < -1 and y_curr > floor_zone:
            bounces.append({"type": "floor", "x": x_curr, "y": y_curr, "frame": frame_num})

        # Wall bounce: horizontal direction reversal near wall
        elif (x_curr < wall_margin or x_curr > court_width - wall_margin):
            if dx_before * dx_after < 0 and abs(dx_before) > 2:
                wall = "left" if x_curr < wall_margin else "right"
                bounces.append({"type": f"wall_{wall}", "x": x_curr, "y": y_curr, "frame": frame_num})

        # Front wall bounce: near top
        elif y_curr < wall_margin:
            if dy_before < -2 and dy_after > 1:
                bounces.append({"type": "front_wall", "x": x_curr, "y": y_curr, "frame": frame_num})

    return bounces


# ============================================================
# 6. SWING PHASE DETECTION
# ============================================================

class SwingPhaseDetector:
    """Detect swing phases: preparation, swing, follow-through.
    
    Uses wrist velocity relative to shoulder over a sliding window.
    """

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        # Per-player history: list of (wrist_x, wrist_y, shoulder_x, shoulder_y)
        self.history: Dict[int, List[Tuple[float, float, float, float]]] = {1: [], 2: []}

    def update(self, player_id: int, keypoints: np.ndarray, frame_width: int, frame_height: int) -> str:
        """Update with latest keypoints and return phase.
        
        keypoints: (17, 2) normalized array.
        Returns: 'preparation', 'swing', 'follow_through', or 'idle'
        """
        if keypoints is None or len(keypoints) < 17:
            return "idle"

        try:
            # Use dominant wrist (whichever has more movement)
            lw = keypoints[9]  # left wrist
            rw = keypoints[10]  # right wrist
            ls = keypoints[5]  # left shoulder
            rs = keypoints[6]  # right shoulder

            # Pick the wrist that's higher (more likely swinging)
            if lw[1] < rw[1] and lw[0] > 0 and lw[1] > 0:
                wx, wy = float(lw[0]) * frame_width, float(lw[1]) * frame_height
                sx, sy = float(ls[0]) * frame_width, float(ls[1]) * frame_height
            else:
                wx, wy = float(rw[0]) * frame_width, float(rw[1]) * frame_height
                sx, sy = float(rs[0]) * frame_width, float(rs[1]) * frame_height

            if wx == 0 and wy == 0:
                return "idle"

            hist = self.history.get(player_id, [])
            hist.append((wx, wy, sx, sy))
            if len(hist) > self.window_size * 2:
                hist = hist[-(self.window_size * 2):]
            self.history[player_id] = hist

            if len(hist) < self.window_size:
                return "idle"

            # Calculate wrist velocity (relative to shoulder) over window
            recent = hist[-self.window_size:]
            velocities = []
            for i in range(1, len(recent)):
                dwx = recent[i][0] - recent[i - 1][0]
                dwy = recent[i][1] - recent[i - 1][1]
                dsx = recent[i][2] - recent[i - 1][2]
                dsy = recent[i][3] - recent[i - 1][3]
                # Relative wrist velocity (subtract body movement)
                rel_vx = dwx - dsx
                rel_vy = dwy - dsy
                velocities.append(math.sqrt(rel_vx ** 2 + rel_vy ** 2))

            avg_vel = sum(velocities) / max(1, len(velocities))
            max_vel = max(velocities) if velocities else 0

            # Wrist above shoulder = arm raised
            wrist_above_shoulder = wy < sy

            # Classify phase
            if max_vel > 15 and avg_vel > 8:
                return "swing"
            elif wrist_above_shoulder and avg_vel > 3:
                return "preparation"
            elif not wrist_above_shoulder and avg_vel > 3 and max_vel < 15:
                return "follow_through"
            else:
                return "idle"

        except Exception:
            return "idle"


# ============================================================
# 7. MULTI-SCALE BALL DETECTION
# ============================================================

def detect_ball_multiscale(
    frame: np.ndarray,
    ballmodel: Any,
    conf_threshold: float = 0.2,
    scales: Tuple[float, ...] = (1.0, 1.5, 0.75),
) -> Optional[Tuple[int, int, float]]:
    """Run ball detection at multiple scales for better small ball detection.
    
    Returns (cx, cy, confidence) of best detection, or None.
    """
    best = None
    best_conf = 0.0
    h, w = frame.shape[:2]

    for scale in scales:
        if scale == 1.0:
            scaled = frame
        else:
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled = cv2.resize(frame, (new_w, new_h))

        try:
            results = list(ballmodel(scaled, verbose=False, conf=conf_threshold, stream=True))
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                b = results[0].boxes
                conf = b.conf.cpu().numpy()
                xyxy = b.xyxy.cpu().numpy()
                idx = int(np.argmax(conf))
                c = float(conf[idx])

                if c > best_conf:
                    x1, y1, x2, y2 = xyxy[idx]
                    # Scale back to original coordinates
                    cx = int(((x1 + x2) / 2) / scale)
                    cy = int(((y1 + y2) / 2) / scale)
                    cx = max(0, min(w - 1, cx))
                    cy = max(0, min(h - 1, cy))
                    best = (cx, cy, c)
                    best_conf = c
        except Exception:
            continue

    return best


# ============================================================
# 8. COURT LINE DETECTION
# ============================================================

def detect_court_lines(
    frame: np.ndarray,
    min_line_length: int = 50,
    max_line_gap: int = 15,
) -> List[Tuple[int, int, int, int]]:
    """Detect court lines using Hough transform.
    
    Returns list of (x1, y1, x2, y2) line segments.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Edge detection
    edges = cv2.Canny(enhanced, 50, 150)
    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    result = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter: keep mostly horizontal or vertical lines (court lines)
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            if angle < 20 or angle > 70:  # roughly horizontal or vertical
                result.append((x1, y1, x2, y2))

    return result


def draw_court_lines(frame: np.ndarray, lines: List[Tuple[int, int, int, int]], color=(0, 200, 200), thickness=1):
    """Draw detected court lines on frame."""
    for x1, y1, x2, y2 in lines:
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


# ============================================================
# 9. IMPROVED PLAYER RE-IDENTIFICATION
# ============================================================

class EnhancedPlayerReID:
    """Improved player re-identification using multiple feature types.
    
    Combines:
    - Color histogram (HSV torso)
    - Spatial position with momentum
    - Height ratio estimation
    """

    def __init__(self):
        self.appearance: Dict[int, np.ndarray] = {}
        self.position_history: Dict[int, List[Tuple[float, float]]] = {1: [], 2: []}
        self.height_estimates: Dict[int, List[float]] = {1: [], 2: []}
        self.ema_alpha = 0.1  # slower EMA for more stable appearance

    def compute_features(
        self,
        keypoints: np.ndarray,
        frame: np.ndarray,
        frame_width: int,
        frame_height: int,
        bbox: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extract multiple features for a detected person."""
        features: Dict[str, Any] = {}

        # 1. Color histogram from torso
        try:
            pts = []
            for idx in [5, 6, 11, 12]:  # shoulders + hips
                x = float(keypoints[idx][0]) * frame_width
                y = float(keypoints[idx][1]) * frame_height
                if x > 0 and y > 0:
                    pts.append((x, y))
            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1 = int(max(0, min(xs) - 10))
                x2 = int(min(frame_width - 1, max(xs) + 10))
                y1 = int(max(0, min(ys) - 10))
                y2 = int(min(frame_height - 1, max(ys) + 10))
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    features["color_hist"] = hist.astype(np.float32)
        except Exception:
            pass
        if "color_hist" not in features:
            features["color_hist"] = np.zeros((30, 32), dtype=np.float32)

        # 2. Position (ankle center)
        try:
            la = keypoints[16]
            ra = keypoints[15]
            if la[0] > 0 and ra[0] > 0:
                cx = float((la[0] + ra[0]) / 2) * frame_width
                cy = float((la[1] + ra[1]) / 2) * frame_height
            elif bbox is not None:
                cx = float((bbox[0] + bbox[2]) / 2)
                cy = float((bbox[1] + bbox[3]) / 2)
            else:
                cx, cy = 0.0, 0.0
            features["center"] = (cx, cy)
        except Exception:
            features["center"] = (0.0, 0.0)

        # 3. Estimated height (head to ankle distance)
        try:
            head = keypoints[0]
            ankle_avg_y = (float(keypoints[15][1]) + float(keypoints[16][1])) / 2
            head_y = float(head[1])
            if head_y > 0 and ankle_avg_y > 0:
                features["height_ratio"] = abs(ankle_avg_y - head_y)
            else:
                features["height_ratio"] = 0.0
        except Exception:
            features["height_ratio"] = 0.0

        return features

    def match_cost(self, slot: int, features: Dict[str, Any], frame_diag: float) -> float:
        """Compute assignment cost for a detection to a player slot."""
        cost = 0.0

        # Spatial cost with momentum prediction
        cx, cy = features["center"]
        hist = self.position_history.get(slot, [])
        if len(hist) >= 2:
            # Predict position using velocity
            vx = hist[-1][0] - hist[-2][0]
            vy = hist[-1][1] - hist[-2][1]
            pred_x = hist[-1][0] + vx
            pred_y = hist[-1][1] + vy
            spatial_dist = math.sqrt((cx - pred_x) ** 2 + (cy - pred_y) ** 2)
        elif len(hist) == 1:
            spatial_dist = math.sqrt((cx - hist[-1][0]) ** 2 + (cy - hist[-1][1]) ** 2)
        else:
            spatial_dist = frame_diag * 0.5  # unknown

        spatial_cost = spatial_dist / max(1.0, frame_diag)

        # Color histogram cost
        ref_hist = self.appearance.get(slot)
        if ref_hist is not None:
            corr = cv2.compareHist(ref_hist, features["color_hist"], cv2.HISTCMP_CORREL)
            color_cost = (1.0 - float(corr)) * 0.5
            color_cost = max(0.0, min(1.0, color_cost))
        else:
            color_cost = 0.5

        # Height consistency cost
        height_estimates = self.height_estimates.get(slot, [])
        if height_estimates and features["height_ratio"] > 0:
            avg_h = sum(height_estimates) / len(height_estimates)
            height_cost = abs(features["height_ratio"] - avg_h) / max(0.01, avg_h)
            height_cost = min(1.0, height_cost)
        else:
            height_cost = 0.0

        # Weighted combination (spatial most important when far, color when close)
        if spatial_dist < 100:
            cost = 0.3 * spatial_cost + 0.5 * color_cost + 0.2 * height_cost
        else:
            cost = 0.6 * spatial_cost + 0.3 * color_cost + 0.1 * height_cost

        return cost

    def update_slot(self, slot: int, features: Dict[str, Any]):
        """Update stored features for a player slot."""
        # Position
        cx, cy = features["center"]
        if cx > 0 or cy > 0:
            hist = self.position_history.get(slot, [])
            hist.append((cx, cy))
            if len(hist) > 60:
                hist = hist[-60:]
            self.position_history[slot] = hist

        # Color histogram EMA
        cur = features["color_hist"]
        ref = self.appearance.get(slot)
        if ref is None or ref.shape != cur.shape:
            self.appearance[slot] = cur.copy()
        else:
            self.appearance[slot] = (1 - self.ema_alpha) * ref + self.ema_alpha * cur

        # Height
        if features["height_ratio"] > 0:
            h_list = self.height_estimates.get(slot, [])
            h_list.append(features["height_ratio"])
            if len(h_list) > 30:
                h_list = h_list[-30:]
            self.height_estimates[slot] = h_list


# ============================================================
# 10. HUD OVERLAY - Display all enhanced stats on frame
# ============================================================

def draw_enhanced_hud(
    frame: np.ndarray,
    shot_info: Dict,
    ball_speed: Dict,
    rally_info: Dict,
    swing_phases: Dict[int, str],
    bounces_count: int,
    frame_width: int,
    frame_height: int,
):
    """Draw a compact HUD with all enhanced tracking info."""
    y_offset = 15
    line_h = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    def put(text, y, col=color):
        cv2.putText(frame, text, (frame_width - 250, y), font, scale, bg_color, 2)
        cv2.putText(frame, text, (frame_width - 250, y), font, scale, col, 1)

    # Shot type
    shot_display = shot_info.get("display", "unknown")
    conf = shot_info.get("confidence", 0)
    put(f"Shot: {shot_display} ({conf:.0%})", y_offset, (0, 200, 255))
    y_offset += line_h

    # Ball speed
    speed = ball_speed.get("speed_kmh", 0)
    if speed > 0:
        put(f"Ball Speed: {speed:.0f} km/h", y_offset, (0, 255, 200))
        y_offset += line_h

    # Rally info
    if rally_info.get("in_rally"):
        put(f"Rally: {rally_info['rally_shots']} shots | {rally_info['rally_duration_s']}s", y_offset, (255, 255, 0))
    else:
        put("Between rallies", y_offset, (150, 150, 150))
    y_offset += line_h

    # Score
    put(f"Score: P1 {rally_info.get('p1_points', 0)} - P2 {rally_info.get('p2_points', 0)}", y_offset, (255, 200, 0))
    y_offset += line_h

    # Swing phases
    for pid in [1, 2]:
        phase = swing_phases.get(pid, "idle")
        if phase != "idle":
            put(f"P{pid}: {phase}", y_offset, (200, 150, 255))
            y_offset += line_h

    # Bounces
    if bounces_count > 0:
        put(f"Bounces detected: {bounces_count}", y_offset, (0, 200, 200))
