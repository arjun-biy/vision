import os
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

# Minimal helpers to unblock the pipeline. These are intentionally lightweight
# and focus on matching the call signatures used in get_data.py.


def cleanwrite() -> None:
    """Prepare output directories/files if needed.

    Currently ensures the `output` directory exists. No destructive actions.
    """
    os.makedirs("output", exist_ok=True)


def generate_homography(
    reference_points_2d: Sequence[Sequence[float]],
    reference_points_3d: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate homography matrices for XY and Z from 2D→3D references.

    Returns a tuple (H_xy, H_z). Pass this as `homography` to `pixel_to_3d`.
    """
    pts2d = np.asarray(reference_points_2d, dtype=np.float32)
    pts3d = np.asarray(reference_points_3d, dtype=np.float32)

    if len(pts2d) < 4 or len(pts3d) < 4:
        raise ValueError("At least 4 reference points are required for homography")

    # Map image (x, y) → court (X, Y)
    H_xy, _ = cv2.findHomography(pts2d, pts3d[:, :2])

    # Map image (x, y) → (x, Z) by pairing 2D x with 3D Z in a 2D plane
    z_coords = np.column_stack((pts2d[:, 0], pts3d[:, 2]))
    H_z, _ = cv2.findHomography(pts2d, z_coords)
    return H_xy, H_z


def pixel_to_3d(
    pixel_xy: Sequence[float],
    homography: Tuple[np.ndarray, np.ndarray],
    reference_points_3d: Sequence[Sequence[float]],  # unused but kept for API parity
) -> List[float]:
    """Project a single image-space point (x, y) to court-space (X, Y, Z).

    `homography` must be the (H_xy, H_z) tuple from `generate_homography`.
    """
    H_xy, H_z = homography
    pt = np.asarray(pixel_xy, dtype=np.float32).reshape(1, 1, 2)

    # Apply planar transforms
    xy_warp = cv2.perspectiveTransform(pt, H_xy).reshape(2)
    xz_warp = cv2.perspectiveTransform(pt, H_z).reshape(2)

    X, Y = float(xy_warp[0]), float(xy_warp[1])
    Z = float(xz_warp[1])  # second component encodes Z from the synthetic (x, Z) plane
    return [X, Y, Z]


def validate_reference_points(
    reference_points_2d: Sequence[Sequence[float]],
    reference_points_3d: Sequence[Sequence[float]],
) -> None:
    """Basic validation for reference point lists."""
    if len(reference_points_2d) < 4:
        raise ValueError("Need at least 4 2D reference points")
    if len(reference_points_3d) < 4:
        raise ValueError("Need at least 4 3D reference points")


def sum_pixels_in_bbox(frame: np.ndarray, bbox: Sequence[int]) -> int:
    """Sum grayscale pixel intensities in the given bbox [x, y, w, h]."""
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    roi = frame[max(0, y) : max(0, y2), max(0, x) : max(0, x2)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return int(gray.sum())


def is_camera_angle_switched(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    threshold: float = 0.5,
) -> bool:
    """Heuristic: compare summed intensity; large relative change → switched.

    `threshold` is the relative change fraction (0–1) to consider a switch.
    """
    h, w = frame.shape[:2]
    sum_current = sum_pixels_in_bbox(frame, [0, 0, w, h])
    sum_ref = sum_pixels_in_bbox(reference_frame, [0, 0, w, h])
    if sum_ref == 0:
        return False
    change = abs(sum_current - sum_ref) / sum_ref
    return change > threshold


def ballplayer_detections(
    *,
    frame: np.ndarray,
    frame_height: int,
    frame_width: int,
    frame_count: int,
    annotated_frame: np.ndarray,
    ballmodel: Any,
    pose_model: Any,
    mainball: Any,
    ball: Any,
    ballmap: Any,
    past_ball_pos: List[Any],
    ball_false_pos: List[Any],
    running_frame: int,
    other_track_ids: List[List[int]],
    updated: List[List[Any]],
    references1: List[Any],
    references2: List[Any],
    pixdiffs: List[Any],
    players: Dict[int, Any],
    player_last_positions: Dict[int, Any],
    occluded: bool,
    importantdata: List[Any],
) -> Tuple[Any, ...]:
    """Basic YOLO-based detection for players (pose) and ball.

    - Updates `players` dict with latest pose objects holding `.xyn`
    - Updates `mainball` position and `past_ball_pos`
    - Draws simple overlays on `annotated_frame`
    """

    class PoseWrapper:
        def __init__(self, xyn_arr: np.ndarray):
            # Ensure shape (1, 17, 2) so downstream indexing `.xyn[0][16][0]` works
            if xyn_arr.ndim == 2:  # (17,2)
                self.xyn = np.expand_dims(xyn_arr, axis=0)
            else:
                self.xyn = xyn_arr

    next_frame_count = frame_count + 1
    next_running = running_frame + 1

    # Helper: compute HSV color histogram for a torso ROI
    def compute_torso_hist(kp_arr: np.ndarray, fallback_box: np.ndarray | None) -> np.ndarray:
        try:
            # Use shoulders (5,6) and hips (11,12) to define torso box
            pts = []
            for idx in [5, 6, 11, 12]:
                x = float(kp_arr[idx][0]) * frame_width
                y = float(kp_arr[idx][1]) * frame_height
                pts.append((x, y))
            xs = [p[0] for p in pts if p[0] > 0]
            ys = [p[1] for p in pts if p[1] > 0]
            if len(xs) >= 2 and len(ys) >= 2:
                x1, x2 = int(max(0, min(xs) - 15)), int(min(frame_width - 1, max(xs) + 15))
                y1, y2 = int(max(0, min(ys) - 20)), int(min(frame_height - 1, max(ys) + 20))
            elif fallback_box is not None:
                fx1, fy1, fx2, fy2 = [int(v) for v in fallback_box]
                pad = 10
                x1, y1 = max(0, fx1 - pad), max(0, fy1 - pad)
                x2, y2 = min(frame_width - 1, fx2 + pad), min(frame_height - 1, fy2 + pad)
            else:
                return np.zeros((30, 32), dtype=np.float32)
            if x2 <= x1 or y2 <= y1:
                return np.zeros((30, 32), dtype=np.float32)
            roi = annotated_frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return hist.astype(np.float32)
        except Exception:
            return np.zeros((30, 32), dtype=np.float32)

    # Appearance store for re-ID across frames (attached to player_last_positions)
    appearance_store: Dict[int, np.ndarray] = player_last_positions.get("__appearance__", {}) or {}

    # --- Player pose detection ---
    try:
        pose_results = pose_model(frame, verbose=False, conf=0.25, stream=True)
        pose_results = list(pose_results)  # materialize generator from stream=True
        if pose_results and len(pose_results) > 0 and pose_results[0].keypoints is not None:
            kps = pose_results[0].keypoints  # Ultralytics Keypoints
            kps_xyn = np.asarray(kps.xyn)  # (n, 17, 2)
            n_people = kps_xyn.shape[0] if kps_xyn is not None else 0

            # Prefer people with largest bbox area (if boxes available)
            indices = list(range(n_people))
            try:
                boxes = pose_results[0].boxes.xyxy.cpu().numpy() if pose_results[0].boxes is not None else None
                scores = pose_results[0].boxes.conf.cpu().numpy() if pose_results[0].boxes is not None else None
            except Exception:
                boxes, scores = None, None

            # Compute detection centers and color histograms
            det_centers = []
            det_hists: List[np.ndarray] = []
            for i in range(n_people):
                try:
                    kp_i = kps_xyn[i]
                    la = kp_i[16]; ra = kp_i[15]
                    if la[0] > 0 and la[1] > 0 and ra[0] > 0 and ra[1] > 0:
                        cx = float((la[0] + ra[0]) / 2) * frame_width
                        cy = float((la[1] + ra[1]) / 2) * frame_height
                    elif boxes is not None and i < len(boxes):
                        x1, y1, x2, y2 = boxes[i]
                        cx = float((x1 + x2) / 2)
                        cy = float((y1 + y2) / 2)
                    else:
                        nose = kp_i[0]
                        cx = float(nose[0]) * frame_width
                        cy = float(nose[1]) * frame_height
                    # Hist
                    fb = boxes[i] if boxes is not None and i < len(boxes) else None
                    det_hists.append(compute_torso_hist(kp_i, fb))
                except Exception:
                    cx, cy = 0.0, 0.0
                    det_hists.append(np.zeros((30, 32), dtype=np.float32))
                det_centers.append((cx, cy))

            # Get previous centers for slots
            prev1 = player_last_positions.get(1, [])
            prev2 = player_last_positions.get(2, [])
            prev1c = prev1[-1] if len(prev1) > 0 else None
            prev2c = prev2[-1] if len(prev2) > 0 else None

            # Determine stable assignment (slot -> detection index)
            slot_to_idx = {}
            if n_people >= 1 and prev1c is not None and prev2c is not None:
                # Cost matrix rows=slots (2), cols=detections with spatial+color terms
                diag = float(np.hypot(frame_width, frame_height))
                # When players are close, emphasize color more
                close = np.hypot(prev1c[0] - prev2c[0], prev1c[1] - prev2c[1]) < 150
                alpha, beta = (0.5, 0.5) if close else (0.7, 0.3)
                costs = []
                for slot, (px, py) in zip([1, 2], [prev1c, prev2c]):
                    ref = appearance_store.get(slot)
                    row = []
                    for j, (cx, cy) in enumerate(det_centers):
                        d_spatial = np.hypot(cx - px, cy - py) / max(1.0, diag)
                        if ref is not None:
                            corr = cv2.compareHist(ref, det_hists[j], cv2.HISTCMP_CORREL)
                            color_cost = (1.0 - float(corr)) * 0.5  # map [-1,1] -> [1,0]; halve
                            color_cost = max(0.0, min(1.0, color_cost))
                        else:
                            color_cost = 0.5
                        row.append(alpha * d_spatial + beta * color_cost)
                    costs.append(row)
                costs = np.asarray(costs, dtype=np.float32)
                r_ind, c_ind = linear_sum_assignment(costs)
                for r, c in zip(r_ind, c_ind):
                    slot_to_idx[1 if r == 0 else 2] = int(c)
            elif n_people >= 2:
                # Initial assignment by x order: left -> P1, right -> P2
                order = np.argsort([c[0] for c in det_centers]).tolist()
                slot_to_idx[1] = int(order[0])
                slot_to_idx[2] = int(order[-1])
            elif n_people == 1:
                # Assign to closest slot
                cx, cy = det_centers[0]
                if prev1c is None and prev2c is None:
                    slot_to_idx[1] = 0
                else:
                    d1 = np.hypot(cx - prev1c[0], cy - prev1c[1]) if prev1c is not None else np.inf
                    d2 = np.hypot(cx - prev2c[0], cy - prev2c[1]) if prev2c is not None else np.inf
                    slot_to_idx[1 if d1 <= d2 else 2] = 0


            # Ensure players dict has Player instances
            from .Player import Player  # local import to avoid cycles
            if 1 not in players:
                players[1] = Player(1)
            if 2 not in players:
                players[2] = Player(2)

            # Assign up to two players
            # Decide which detection index each slot should use
            decided = []
            for slot in [1, 2]:
                if slot in slot_to_idx:
                    decided.append((slot, slot_to_idx[slot]))
            # Fallback if not decided (e.g., 0 or mismatch)
            used = {idx for _, idx in decided}
            for slot in [1, 2]:
                if len(decided) >= min(2, n_people):
                    break
                if slot not in slot_to_idx and len(used) < n_people:
                    # pick a remaining detection
                    for cand in range(n_people):
                        if cand not in used:
                            decided.append((slot, cand))
                            used.add(cand)
                            break

            for player_slot, idx in decided:
                pose_obj = PoseWrapper(kps_xyn[idx])
                players[player_slot].add_pose(pose_obj)

                # Draw skeleton and ankles if available
                try:
                    kp = players[player_slot].get_latest_pose().xyn[0]  # (17,2)
                    color = (255, 0, 0) if player_slot == 1 else (0, 0, 255)

                    # Draw all keypoints
                    for j in range(min(17, kp.shape[0])):
                        px = int(kp[j][0] * frame_width)
                        py = int(kp[j][1] * frame_height)
                        if px > 0 and py > 0:
                            cv2.circle(annotated_frame, (px, py), 3, color, -1)

                    # Basic COCO skeleton pairs
                    pairs = [
                        (5, 7), (7, 9),  # left arm
                        (6, 8), (8, 10), # right arm
                        (11, 13), (13, 15), # left leg
                        (12, 14), (14, 16), # right leg
                        (5, 6), (11, 12), (5, 11), (6, 12) # torso
                    ]
                    for a, b in pairs:
                        if a < kp.shape[0] and b < kp.shape[0]:
                            axp, ayp = int(kp[a][0] * frame_width), int(kp[a][1] * frame_height)
                            bxp, byp = int(kp[b][0] * frame_width), int(kp[b][1] * frame_height)
                            if axp > 0 and ayp > 0 and bxp > 0 and byp > 0:
                                cv2.line(annotated_frame, (axp, ayp), (bxp, byp), color, 2)

                    # Label
                    head_x, head_y = int(kp[0][0] * frame_width), int(kp[0][1] * frame_height)
                    if head_x > 0 and head_y > 0:
                        cv2.putText(annotated_frame, f"P{player_slot}", (head_x, max(15, head_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Player center (ankles avg) + trail
                    try:
                        la = kp[16]; ra = kp[15]
                        cx = int(((la[0] + ra[0]) / 2) * frame_width)
                        cy = int(((la[1] + ra[1]) / 2) * frame_height)
                        trail = player_last_positions.get(player_slot, [])
                        trail.append((cx, cy))
                        if len(trail) > 100:
                            trail = trail[-100:]
                        player_last_positions[player_slot] = trail

                        # Draw trail as fading circles
                        for t_idx, (tx, ty) in enumerate(trail[-30:]):
                            radius = max(2, 6 - (len(trail[-30:]) - t_idx) // 6)
                            cv2.circle(annotated_frame, (tx, ty), radius, color, -1)
                    except Exception:
                        pass

                    # Draw bounding box if present for this person
                    try:
                        if boxes is not None and idx < len(boxes):
                            x1, y1, x2, y2 = boxes[idx].astype(int)
                            conf = float(scores[idx]) if scores is not None and idx < len(scores) else 0.0
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, f"P{player_slot} {conf:.2f}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception:
                        pass
                except Exception:
                    pass

                # Update appearance model (EMA)
                try:
                    ref = appearance_store.get(player_slot)
                    cur = det_hists[idx]
                    if ref is None or ref.shape != cur.shape:
                        appearance_store[player_slot] = cur.copy()
                    else:
                        ema = 0.15
                        appearance_store[player_slot] = (1 - ema) * ref + ema * cur
                except Exception:
                    pass
    except Exception:
        pass

    # --- Ball detection ---
    try:
        ball_results = ballmodel(frame, verbose=False, conf=0.25, stream=True)
        ball_results = list(ball_results)  # materialize generator from stream=True
        if ball_results and len(ball_results) > 0 and ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
            b = ball_results[0].boxes
            # Choose highest confidence detection
            conf = b.conf.cpu().numpy()
            xyxy = b.xyxy.cpu().numpy()
            best = int(np.argmax(conf))
            x1, y1, x2, y2 = xyxy[best]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Update mainball (keep last pos in pastx/pasty)
            try:
                prev = mainball.getloc()
                mainball.pastx, mainball.pasty = prev[0], prev[1]
            except Exception:
                pass
            mainball.update(cx, cy, 0)

            # Keep a short trajectory in past_ball_pos
            past_ball_pos.append([cx, cy, next_frame_count])
            if len(past_ball_pos) > 2000:
                past_ball_pos[:] = past_ball_pos[-2000:]

            # Draw bounding box (square-style highlight) around the ball
            bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
            # Optional: enforce a square by expanding the shorter side
            bw, bh = bx2 - bx1, by2 - by1
            side = max(bw, bh)
            cx_sq = (bx1 + bx2) // 2
            cy_sq = (by1 + by2) // 2
            bx1 = max(0, cx_sq - side // 2)
            by1 = max(0, cy_sq - side // 2)
            bx2 = min(frame_width - 1, bx1 + side)
            by2 = min(frame_height - 1, by1 + side)

            # Draw a bold outer rectangle and a thin inner rectangle for visibility
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
            cv2.rectangle(annotated_frame, (bx1 + 2, by1 + 2), (bx2 - 2, by2 - 2), (0, 0, 0), 1)
            cv2.putText(annotated_frame, "Ball", (bx1, max(15, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Center marker (small square) for precision
            cv2.rectangle(annotated_frame, (cx - 2, cy - 2), (cx + 2, cy + 2), (0, 255, 0), -1)

            # Remove trajectory lines for a cleaner GIF-like look
    except Exception:
        pass

    idata = None
    # Persist appearance store back into state
    player_last_positions["__appearance__"] = appearance_store
    return (
        frame,  # 0
        next_frame_count,  # 1
        annotated_frame,  # 2
        mainball,  # 3
        ball,  # 4
        ballmap,  # 5
        past_ball_pos,  # 6
        ball_false_pos,  # 7
        next_running,  # 8
        other_track_ids,  # 9
        updated,  # 10
        references1,  # 11
        references2,  # 12
        pixdiffs,  # 13
        players,  # 14
        player_last_positions,  # 15
        None,  # 16 (unused)
        idata,  # 17
    )


def is_match_in_play(players: Dict[int, Any], mainball: Any) -> Any:
    """Placeholder: return False (no active rally)."""
    return False


def classify_shot(past_ball_pos: List[Any]) -> Any:
    """Placeholder shot classifier. Returns None (unknown)."""
    return None


def reorganize_shots(alldata: List[Any]) -> None:
    """Placeholder no-op to match expected API."""
    return None


def find_last(target: int, pairs: Sequence[Sequence[int]]) -> int:
    """Return index of last pair whose first element equals `target`."""
    for i in range(len(pairs) - 1, -1, -1):
        try:
            if pairs[i][0] == target:
                return i
        except Exception:
            continue
    return 0


def plot_coords(coords: Sequence[Sequence[float]]) -> None:
    """Placeholder: do nothing (optional visualization)."""
    return None


