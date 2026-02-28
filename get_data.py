import time

start = time.time()
from dotenv import load_dotenv

load_dotenv()
import cv2
import csv
import time
import csvanalyze
import logging
import math
import numpy as np
import matplotlib
import tensorflow as tf
import torch  # Add PyTorch for GPU optimization
from ultralytics import YOLO
from squash import Referencepoints, Functions  # Ensure Functions is imported
from squash.EnhancedTracking import (
    BallKalmanFilter,
    classify_shot_advanced,
    calculate_ball_speed,
    RallyTracker,
    detect_bounces,
    SwingPhaseDetector,
    detect_ball_multiscale,
    detect_court_lines,
    draw_court_lines,
    draw_enhanced_hud,
)
from matplotlib import pyplot as plt
from squash.Ball import Ball

matplotlib.use("Agg")
print(f"time to import everything: {time.time()-start}")
alldata = organizeddata = []

# ============================================================
# PERFORMANCE CONFIGURATION
# Adjust these to trade off speed vs accuracy
# ============================================================
PROCESS_EVERY_N_FRAMES = 1    # Ball detection every frame for accuracy (pose still every 3rd)
POSE_EVERY_N_FRAMES = 3       # Pose estimation every 3rd frame (slower to change)
HEATMAP_EVERY_N_FRAMES = 30   # Only compute heatmap every 30 frames
PLOT_EVERY_N_FRAMES = 60      # Only save matplotlib plot every 60 frames (was EVERY frame!)
BALL_PREDICT_EVERY_N = 5      # Only run ball prediction model every 5 frames
CSV_WRITE_EVERY_N = 3         # Only write CSV every 3 frames
YOLO_CONF_THRESHOLD = 0.20    # Lowered for better small ball detection
VERBOSE_LOGGING = False        # Set True to restore per-frame print statements
USE_KALMAN_FILTER = True       # Kalman filter for ball tracking
USE_MULTISCALE_BALL = True     # Multi-scale ball detection
USE_COURT_LINES = False        # Court line detection — disabled (too noisy)


def main(path="main_laptop.mp4", frame_width=640, frame_height=360):
    try:
        print("imported all")
        
        # Configure TensorFlow to use GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"TensorFlow using {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("TensorFlow using CPU")
        
        csvstart = 0
        end = csvstart + 100
        ball_predict_model = tf.keras.models.load_model(
            "trained-models/ball_position_model(25k).keras"
        )

        # In-memory buffer for ball positions (replaces reading file from disk every frame)
        ball_positions_buffer = []

        Functions.cleanwrite()
        # GPU-optimized model loading
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using {device.upper()} for YOLO models")
        
        pose_model = YOLO("models/yolo11n-pose.pt")
        pose_model.to(device)  # Move to GPU if available
        
        ballmodel = YOLO("trained-models\\g-ball2(white_latest).pt")
        ballmodel.to(device)  # Move to GPU if available

        print("loaded models")
        ballvideopath = "output/balltracking.mp4"
        cap = cv2.VideoCapture(path)
        
        # Get actual video FPS for accurate timing
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {total_frames} frames at {video_fps} FPS ({total_frames/video_fps:.1f}s)")
        print(f"Will process ~{total_frames // PROCESS_EVERY_N_FRAMES} frames with ML (every {PROCESS_EVERY_N_FRAMES})")
        
        with open("output/final.txt", "w") as f:
            f.write(
                f"You are analyzing video: {path}.\nPlayer keypoints will be structured as such: 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle.\nIf a keypoint is (0,0), then it has not beeen detected and should be deemed irrelevant. Here is how the output will be structured: \nFrame count\nPlayer 1 Keypoints\nPlayer 2 Keypoints\n Ball Position.\n\n"
            )

        players = {}
        courtref = 0
        occlusion_times = {}
        for i in range(1, 3):
            occlusion_times[i] = 0
        future_predict = None
        player_last_positions = {}
        frame_count = 0
        ball_false_pos = []
        past_ball_pos = []
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        output_path = "output/annotated.mp4"
        weboutputpath = "websiteout/annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        importantoutputpath = "output/important.mp4"
        cv2.VideoWriter(weboutputpath, fourcc, fps, (frame_width, frame_height))
        cv2.VideoWriter(importantoutputpath, fourcc, fps, (frame_width, frame_height))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        cv2.VideoWriter(ballvideopath, fourcc, fps, (frame_width, frame_height))
        detections = []

        mainball = Ball(0, 0, 0, 0)
        ballmap = np.zeros((frame_height, frame_width), dtype=np.float32)

        # ── Enhanced tracking objects ──
        ball_kalman = BallKalmanFilter()
        rally_tracker = RallyTracker(fps=video_fps)
        swing_detector = SwingPhaseDetector(window_size=8)
        previous_shot = None
        court_lines_detected = []
        all_bounces = []
        shot_history = []
        ball_speeds = []
        otherTrackIds = [[0, 0], [1, 1], [2, 2]]
        updated = [[False, 0], [False, 0]]
        reference_points = []
        reference_points = Referencepoints.get_reference_points(
            path=path, frame_width=frame_width, frame_height=frame_height
        )

        references1 = []
        references2 = []

        pixdiffs = []

        p1distancesfromT = []
        p2distancesfromT = []

        courtref = np.int64(courtref)
        referenceimage = None
        # Pre-compute reference frame grayscale sum for camera switch detection
        ref_gray_sum = None

        reference_points_3d = [
            [0, 9.75, 0],  # Top-left corner, 1
            [6.4, 9.75, 0],  # Top-right corner, 2
            [6.4, 0, 0],  # Bottom-right corner, 3
            [0, 0, 0],  # Bottom-left corner, 4
            [3.2, 0, 4.31],  # "T" point, 5
            [0, 2.71, 0],  # Left bottom of the service box, 6
            [6.4, 2.71, 0],  # Right bottom of the service box, 7
            [0, 9.75, 0.48],  # left of tin, 8
            [6.4, 9.75, 0.48],  # right of tin, 9
            [0, 9.75, 1.78],  # Left of the service line, 10
            [6.4, 9.75, 1.78],  # Right of the service line, 11
            [0, 9.75, 4.57],  # Left of the top line of the front court, 12
            [6.4, 9.75, 4.57],  # Right of the top line of the front court, 13
        ]
        homography = Functions.generate_homography(
            reference_points, reference_points_3d
        )

        heatmap_overlay_path = "output/white.png"
        heatmap_image = cv2.imread(heatmap_overlay_path)
        if heatmap_image is None:
            # Create a blank white image as heatmap canvas if it doesn't exist
            heatmap_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            cv2.imwrite(heatmap_overlay_path, heatmap_image)

        ballxy = []

        running_frame = 0
        print("started video input")
        Functions.validate_reference_points(reference_points, reference_points_3d)
        print(f"loaded everything in {time.time()-start} seconds")
        
        # CSV buffer for batch writing (avoid opening file every frame)
        csv_buffer = []
        
        # Open CSV file once (not per-frame!)
        csv_file = open("output/final.csv", "a", newline="")
        csvwriter = csv.writer(csv_file)
        
        # Track last ML detection results for reuse on skipped frames
        last_detections_result = None
        
        process_start = time.time()
        frames_processed_ml = 0
        
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            frame_count += 1

            if len(references1) != 0 and len(references2) != 0:
                sum(references1) / len(references1)
                sum(references2) / len(references2)

            running_frame += 1
            if running_frame == 1:
                courtref = np.int64(
                    Functions.sum_pixels_in_bbox(
                        frame, [0, 0, frame_width, frame_height]
                    )
                )
                referenceimage = frame
                # Pre-compute reference grayscale for fast comparison
                ref_gray_sum = int(cv2.cvtColor(referenceimage, cv2.COLOR_BGR2GRAY).sum())

            # --- Fast camera switch detection (avoid redundant grayscale conversion) ---
            if referenceimage is not None:
                current_gray_sum = int(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).sum())
                if ref_gray_sum and ref_gray_sum > 0:
                    if abs(current_gray_sum - ref_gray_sum) / ref_gray_sum > 0.5:
                        continue
                    # Also check court reference (combined into one grayscale pass)
                    if abs(courtref - current_gray_sum) > courtref * 0.6:
                        continue

            # --- Decide if this frame gets ML processing ---
            # Ball detection runs every frame; pose estimation runs every Nth
            run_ball_this_frame = (running_frame % PROCESS_EVERY_N_FRAMES == 0) or (running_frame <= 2)
            run_pose_this_frame = (running_frame % POSE_EVERY_N_FRAMES == 1) or (running_frame <= 2)
            run_ml_this_frame = run_ball_this_frame or run_pose_this_frame

            # Detect court lines on first frame only
            if running_frame == 1 and USE_COURT_LINES:
                try:
                    court_lines_detected = detect_court_lines(frame)
                except Exception:
                    court_lines_detected = []

            annotated_frame = frame.copy()

            for reference in reference_points:
                cv2.circle(
                    annotated_frame,
                    (int(reference[0]), int(reference[1])),
                    5,
                    (0, 255, 0),
                    2,
                )

            if run_ml_this_frame:
                # === FULL ML PROCESSING (pose + ball detection) ===
                frames_processed_ml += 1
                
                # NOTE: We pass ball=None since ballplayer_detections runs its own
                # ball detection internally - no need to run ballmodel TWICE!
                detections_result = Functions.ballplayer_detections(
                    frame=frame,
                    frame_height=frame_height,
                    frame_width=frame_width,
                    frame_count=frame_count,
                    annotated_frame=annotated_frame,
                    ballmodel=ballmodel,
                    pose_model=pose_model,
                    mainball=mainball,
                    ball=None,  # Was: ballmodel(frame) -- REMOVED DUPLICATE!
                    ballmap=ballmap,
                    past_ball_pos=past_ball_pos,
                    ball_false_pos=ball_false_pos,
                    running_frame=running_frame,
                    other_track_ids=otherTrackIds,
                    updated=updated,
                    references1=references1,
                    references2=references2,
                    pixdiffs=pixdiffs,
                    players=players,
                    player_last_positions=player_last_positions,
                    occluded=False,
                    importantdata=[],
                )
                frame = detections_result[0]
                frame_count = detections_result[1]
                annotated_frame = detections_result[2]
                mainball = detections_result[3]
                ball = detections_result[4]
                ballmap = detections_result[5]
                past_ball_pos = detections_result[6]
                ball_false_pos = detections_result[7]
                running_frame = detections_result[8]
                otherTrackIds = detections_result[9]
                updated = detections_result[10]
                references1 = detections_result[11]
                references2 = detections_result[12]
                pixdiffs = detections_result[13]
                players = detections_result[14]
                player_last_positions = detections_result[15]
                detections_result[16]
                idata = detections_result[17]
                if idata:
                    alldata.append(idata)
                last_detections_result = detections_result
            else:
                # === SKIPPED FRAME: reuse last detection results for annotations ===
                # Just increment counters, don't run ML models
                frame_count += 1
                running_frame += 1

            # ── Enhanced shot classification ──
            shot_info = classify_shot_advanced(
                past_ball_pos, court_width=frame_width, court_height=frame_height,
                previous_shot=previous_shot,
            )
            previous_shot = shot_info
            type_of_shot = [shot_info.get("direction", ""), shot_info.get("height", "")]

            # ── Ball speed ──
            ball_speed = calculate_ball_speed(
                past_ball_pos, homography, reference_points_3d,
                video_fps=video_fps, pixel_to_3d_fn=Functions.pixel_to_3d,
            )
            if ball_speed["speed_kmh"] > 0:
                ball_speeds.append(ball_speed["speed_kmh"])

            # ── Rally tracking ──
            ball_detected = (mainball is not None and mainball.getloc() != [0, 0])
            ball_pos_tuple = tuple(mainball.getloc()) if ball_detected else None
            rally_info = rally_tracker.update(
                ball_detected=ball_detected,
                ball_pos=ball_pos_tuple,
                players=players,
                frame_num=running_frame,
                court_height=frame_height,
            )

            # ── Swing phase detection ──
            swing_phases = {}
            for pid in [1, 2]:
                if players.get(pid) and players[pid].get_latest_pose() is not None:
                    try:
                        kp = players[pid].get_latest_pose().xyn[0]
                        swing_phases[pid] = swing_detector.update(pid, kp, frame_width, frame_height)
                    except Exception:
                        swing_phases[pid] = "idle"

            # ── Bounce detection (periodic) ──
            bounces_this_frame = []
            if running_frame % 10 == 0:
                bounces_this_frame = detect_bounces(past_ball_pos, frame_width, frame_height)
                all_bounces.extend(bounces_this_frame)

            # ── Draw court lines ──
            if court_lines_detected:
                draw_court_lines(annotated_frame, court_lines_detected)

            # ── Draw enhanced HUD ──
            draw_enhanced_hud(
                annotated_frame, shot_info, ball_speed, rally_info,
                swing_phases, len(all_bounces), frame_width, frame_height,
            )

            # Legacy compatibility
            match_in_play = Functions.is_match_in_play(players, mainball)
            try:
                Functions.reorganize_shots(alldata)
            except Exception:
                pass

            # Display ankle positions of both players
            if players.get(1) and players.get(2) is not None:
                if (
                    players.get(1).get_latest_pose()
                    or players.get(2).get_latest_pose() is not None
                ):
                    try:
                        p1_left_ankle_x = int(
                            players.get(1).get_latest_pose().xyn[0][16][0] * frame_width
                        )
                        p1_left_ankle_y = int(
                            players.get(1).get_latest_pose().xyn[0][16][1]
                            * frame_height
                        )
                        p1_right_ankle_x = int(
                            players.get(1).get_latest_pose().xyn[0][15][0] * frame_width
                        )
                        p1_right_ankle_y = int(
                            players.get(1).get_latest_pose().xyn[0][15][1]
                            * frame_height
                        )
                    except Exception:
                        p1_left_ankle_x = p1_left_ankle_y = p1_right_ankle_x = (
                            p1_right_ankle_y
                        ) = 0
                    try:
                        p2_left_ankle_x = int(
                            players.get(2).get_latest_pose().xyn[0][16][0] * frame_width
                        )
                        p2_left_ankle_y = int(
                            players.get(2).get_latest_pose().xyn[0][16][1]
                            * frame_height
                        )
                        p2_right_ankle_x = int(
                            players.get(2).get_latest_pose().xyn[0][15][0] * frame_width
                        )
                        p2_right_ankle_y = int(
                            players.get(2).get_latest_pose().xyn[0][15][1]
                            * frame_height
                        )
                    except Exception:
                        p2_left_ankle_x = p2_left_ankle_y = p2_right_ankle_x = (
                            p2_right_ankle_y
                        ) = 0
                    # Display the ankle positions on the bottom left of the frame
                    avgxank1 = int((p1_left_ankle_x + p1_right_ankle_x) / 2)
                    avgyank1 = int((p1_left_ankle_y + p1_right_ankle_y) / 2)
                    avgxank2 = int((p2_left_ankle_x + p2_right_ankle_x) / 2)
                    avgyank2 = int((p2_left_ankle_y + p2_right_ankle_y) / 2)
                    text_p1 = f"P1 position(ankle): {avgxank1},{avgyank1}"
                    cv2.putText(
                        annotated_frame,
                        f"{otherTrackIds[Functions.find_last(1, otherTrackIds)][1]}",
                        (p1_left_ankle_x, p1_left_ankle_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"{otherTrackIds[Functions.find_last(2, otherTrackIds)][1]}",
                        (p2_left_ankle_x, p2_left_ankle_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    text_p2 = f"P2 position(ankle): {avgxank2},{avgyank2}"
                    cv2.putText(
                        annotated_frame,
                        text_p1,
                        (10, frame_height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        text_p2,
                        (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    avgpx1 = int((p1_left_ankle_x + p1_right_ankle_x) / 2)
                    avgpy1 = int((p1_left_ankle_y + p1_right_ankle_y) / 2)
                    avgpx2 = int((p2_left_ankle_x + p2_right_ankle_x) / 2)
                    avgpy2 = int((p2_left_ankle_y + p2_right_ankle_y) / 2)
                    p1distancefromT = math.hypot(
                        reference_points[4][0] - avgpx1, reference_points[4][1] - avgpy1
                    )
                    p2distancefromT = math.hypot(
                        reference_points[4][0] - avgpx2, reference_points[4][1] - avgpy2
                    )
                    p1distancesfromT.append(p1distancefromT)
                    p2distancesfromT.append(p2distancefromT)
                    text_p1t = f"P1 distance from T: {p1distancesfromT[-1]}"
                    text_p2t = f"P2 distance from T: {p2distancesfromT[-1]}"
                    cv2.putText(
                        annotated_frame,
                        text_p1t,
                        (10, frame_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        text_p2t,
                        (10, frame_height - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    
                    # PERF: Only save matplotlib plot periodically (was EVERY frame - extremely slow!)
                    if running_frame % PLOT_EVERY_N_FRAMES == 0:
                        plt.figure(figsize=(10, 6))
                        plt.plot(p1distancesfromT, color="blue", label="P1 Distance from T")
                        plt.plot(p2distancesfromT, color="red", label="P2 Distance from T")
                        plt.xlabel("Time (frames)")
                        plt.ylabel("Distance from T")
                        plt.title("Distance from T over Time")
                        plt.legend()
                        plt.savefig("output/distance_from_t_over_time.png")
                        plt.close()

            # PERF: Only compute heatmap periodically (was every frame)
            if running_frame % HEATMAP_EVERY_N_FRAMES == 0:
                try:
                    if (
                        players.get(1) is not None
                        and players.get(2) is not None
                        and players.get(1).get_latest_pose() is not None
                        and players.get(2).get_latest_pose() is not None
                    ):
                        player_ankles = [
                            (
                                int(
                                    players.get(1).get_latest_pose().xyn[0][16][0]
                                    * frame_width
                                ),
                                int(
                                    players.get(1).get_latest_pose().xyn[0][16][1]
                                    * frame_height
                                ),
                            ),
                            (
                                int(
                                    players.get(2).get_latest_pose().xyn[0][16][0]
                                    * frame_width
                                ),
                                int(
                                    players.get(2).get_latest_pose().xyn[0][16][1]
                                    * frame_height
                                ),
                            ),
                        ]

                        for ankle in player_ankles:
                            cv2.circle(heatmap_image, ankle, 5, (255, 0, 0), -1)
                            cv2.circle(heatmap_image, ankle, 5, (0, 0, 255), -1)

                    blurred_heatmap_ankle = cv2.GaussianBlur(heatmap_image, (51, 51), 0)
                    normalized_heatmap = cv2.normalize(
                        blurred_heatmap_ankle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                    )
                    heatmap_overlay = cv2.applyColorMap(
                        normalized_heatmap, cv2.COLORMAP_JET
                    )
                    cv2.addWeighted(
                        np.ones_like(heatmap_overlay) * 255, 0.5, heatmap_overlay, 0.5, 0
                    )
                except Exception:
                    pass

            ballx = bally = 0
            # ball stuff - enhanced with Kalman filter
            if (
                mainball is not None
                and mainball.getlastpos() is not None
                and mainball.getlastpos() != (0, 0)
            ):
                ballx = mainball.getlastpos()[0]
                bally = mainball.getlastpos()[1]
                if ballx != 0 and bally != 0:
                    # Apply Kalman filter for smoothing
                    if USE_KALMAN_FILTER:
                        kf_x, kf_y = ball_kalman.update(float(ballx), float(bally))
                        ballx, bally = int(np.asarray(kf_x).flat[0]), int(np.asarray(kf_y).flat[0])
                    if [ballx, bally] not in ballxy:
                        ballxy.append([ballx, bally, frame_count])
                        # Also update in-memory buffer for ball prediction
                        ball_positions_buffer.append(
                            (ballx / frame_width, bally / frame_height)
                        )
            elif USE_KALMAN_FILTER and ball_kalman.is_tracking:
                # No detection - use Kalman prediction to fill gaps
                predicted = ball_kalman.predict()
                if predicted is not None:
                    ballx, bally = int(np.asarray(predicted[0]).flat[0]), int(np.asarray(predicted[1]).flat[0])
                    if 0 < ballx < frame_width and 0 < bally < frame_height:
                        ballxy.append([ballx, bally, frame_count])
                        ball_positions_buffer.append(
                            (ballx / frame_width, bally / frame_height)
                        )
                        # Draw predicted position differently (dashed circle)
                        cv2.circle(annotated_frame, (ballx, bally), 6, (0, 165, 255), 2)

            # Draw the ball trajectory
            if len(ballxy) > 2:
                for i in range(1, len(ballxy)):
                    if ballxy[i - 1] is None or ballxy[i] is None:
                        continue
                    if ballxy[i][2] - ballxy[i - 1][2] < 7:
                        if frame_count - ballxy[i][2] < 7:
                            cv2.line(
                                annotated_frame,
                                (ballxy[i - 1][0], ballxy[i - 1][1]),
                                (ballxy[i][0], ballxy[i][1]),
                                (0, 255, 0),
                                2,
                            )
                            cv2.circle(
                                annotated_frame,
                                (ballxy[i - 1][0], ballxy[i - 1][1]),
                                5,
                                (0, 255, 0),
                                -1,
                            )
                            cv2.circle(
                                annotated_frame,
                                (ballxy[i][0], ballxy[i][1]),
                                5,
                                (0, 255, 0),
                                -1,
                            )

            for ball_pos in ballxy:
                if frame_count - ball_pos[2] < 7:
                    cv2.circle(
                        annotated_frame, (ball_pos[0], ball_pos[1]), 5, (0, 255, 0), -1
                    )

            # PERF: Ball prediction - use in-memory buffer (not file I/O) and run periodically
            if running_frame % BALL_PREDICT_EVERY_N == 0 and len(ball_positions_buffer) > 11:
                positions = ball_positions_buffer
                input_sequence = np.array([positions[-10:]])
                input_sequence = input_sequence.reshape((1, 10, 2, 1))
                predicted_pos = ball_predict_model(input_sequence)
                cv2.circle(
                    annotated_frame,
                    (
                        int(predicted_pos[0][0] * frame_width),
                        int(predicted_pos[0][1] * frame_height),
                    ),
                    7,
                    (0, 0, 255),
                    7,
                )
                cv2.putText(
                    annotated_frame,
                    f"predicted ball position in 1 frame: {int(predicted_pos[0][0]*frame_width)},{int(predicted_pos[0][1]*frame_height)}",
                    (
                        int(predicted_pos[0][0] * frame_width),
                        int(predicted_pos[0][1] * frame_height),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                last9 = list(positions[-9:])
                last9.append([predicted_pos[0][0], predicted_pos[0][1]])
                sequence_and_predicted = np.array(last9)
                sequence_and_predicted = sequence_and_predicted.reshape((1, 10, 2, 1))
                future_predict = ball_predict_model(sequence_and_predicted)
                cv2.circle(
                    annotated_frame,
                    (
                        int(future_predict[0][0] * frame_width),
                        int(future_predict[0][1] * frame_height),
                    ),
                    7,
                    (255, 0, 0),
                    7,
                )
                cv2.putText(
                    annotated_frame,
                    f"predicted ball position in 3 frames: {int(future_predict[0][0]*frame_width)},{int(future_predict[0][1]*frame_height)}",
                    (
                        int(future_predict[0][0] * frame_width),
                        int(future_predict[0][1] * frame_height),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
            if (
                players.get(1) is not None
                and players.get(2) is not None
                and players.get(1).get_last_x_poses(3) is not None
                and players.get(2).get_last_x_poses(3) is not None
            ):
                players.get(1).get_last_x_poses(3).xyn[0]
                players.get(2).get_last_x_poses(3).xyn[0]
                rlp1postemp = [
                    players.get(1).get_last_x_poses(3).xyn[0][16][0] * frame_width,
                    players.get(1).get_last_x_poses(3).xyn[0][16][1] * frame_height,
                ]
                rlp2postemp = [
                    players.get(2).get_last_x_poses(3).xyn[0][16][0] * frame_width,
                    players.get(2).get_last_x_poses(3).xyn[0][16][1] * frame_height,
                ]
                rlworldp1 = Functions.pixel_to_3d(
                    rlp1postemp, homography, reference_points_3d
                )
                rlworldp2 = Functions.pixel_to_3d(
                    rlp2postemp, homography, reference_points_3d
                )
                text5 = f"Player 1: {rlworldp1}"
                text6 = f"Player 2: {rlworldp2}"

                cv2.putText(
                    annotated_frame,
                    text5,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    text6,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

            if len(ballxy) > 0:
                balltext = f"Ball position: {ballxy[-1][0]},{ballxy[-1][1]}"
                rlball = Functions.pixel_to_3d(
                    [ballxy[-1][0], ballxy[-1][1]], homography, reference_points_3d
                )
                text4 = f"Ball position in world: {rlball}"
                cv2.putText(
                    annotated_frame,
                    balltext,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    annotated_frame,
                    text4,
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
            
            # PERF: Removed per-frame print (was: print(f"finished writing frame {frame_count}"))
            
            player1rlworldpos = player2rlworldpos = []
            try:
                for position in players.get(1).get_latest_pose().xyn[0]:
                    player1rlworldpos.append(
                        Functions.pixel_to_3d(position, homography, reference_points_3d)
                    )
                for position in players.get(2).get_latest_pose().xyn[0]:
                    player2rlworldpos.append(
                        Functions.pixel_to_3d(position, homography, reference_points_3d)
                    )
                Functions.plot_coords(player1rlworldpos)
            except Exception:
                pass

            if past_ball_pos:
                try:
                    text = f"ball in rlworld: {Functions.pixel_to_3d([past_ball_pos[-1][0],past_ball_pos[-1][1]], homography, reference_points_3d)}"
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                except Exception:
                    pass
            
            # PERF: CSV write - periodic and using pre-opened file handle
            if running_frame % CSV_WRITE_EVERY_N == 0:
                try:
                    shot = shot_info.get("display", "None")
                    data = [
                        running_frame,
                        players.get(1).get_latest_pose().xyn[0].tolist(),
                        players.get(2).get_latest_pose().xyn[0].tolist(),
                        mainball.getloc(),
                        shot,
                        ball_speed.get("speed_kmh", 0),
                        rally_info.get("in_rally", False),
                        rally_info.get("rally_shots", 0),
                        shot_info.get("confidence", 0),
                        swing_phases.get(1, "idle"),
                        swing_phases.get(2, "idle"),
                    ]
                    csvwriter.writerow(data)
                except Exception:
                    pass
            
            if running_frame > end:
                try:
                    with open("final.txt", "a") as f:
                        f.write(
                            csvanalyze.parse_through(csvstart, end, "output/final.csv")
                        )
                    csvstart = end
                    end += 100
                except Exception:
                    pass
            # Add instructions to the frame
            cv2.putText(
                annotated_frame,
                "Press 'r' to update reference points, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            out.write(annotated_frame)
            cv2.imshow("Annotated Frame", annotated_frame)

            # PERF: Progress logging every 30 frames instead of every frame
            if running_frame % 30 == 0:
                elapsed = time.time() - process_start
                fps_actual = running_frame / max(0.01, elapsed)
                print(f"Frame {running_frame}/{total_frames} | ML frames: {frames_processed_ml} | {fps_actual:.1f} fps | {elapsed:.0f}s elapsed")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print("Updating reference points...")
                reference_points = Referencepoints.update_reference_points(
                    path=path, frame_width=frame_width, frame_height=frame_height, current_frame=frame
                )
                homography = Functions.generate_homography(
                    reference_points, reference_points_3d
                )
                print("Reference points updated successfully!")

        # Save final matplotlib plot
        if p1distancesfromT and p2distancesfromT:
            plt.figure(figsize=(10, 6))
            plt.plot(p1distancesfromT, color="blue", label="P1 Distance from T")
            plt.plot(p2distancesfromT, color="red", label="P2 Distance from T")
            plt.xlabel("Time (frames)")
            plt.ylabel("Distance from T")
            plt.title("Distance from T over Time")
            plt.legend()
            plt.savefig("output/distance_from_t_over_time.png")
            plt.close()

        # Flush and close CSV
        csv_file.close()
        
        total_time = time.time() - process_start
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total frames: {running_frame}")
        print(f"ML-processed frames: {frames_processed_ml} ({frames_processed_ml/max(1,running_frame)*100:.0f}%)")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {running_frame/max(0.01,total_time):.1f}")

        # ── Enhanced tracking summary ──
        rally_summary = rally_tracker.get_summary()
        print(f"\n=== ENHANCED TRACKING SUMMARY ===")
        print(f"Total rallies: {rally_summary['total_rallies']}")
        print(f"Score: P1 {rally_summary['p1_points']} - P2 {rally_summary['p2_points']}")
        print(f"Avg rally duration: {rally_summary['avg_rally_duration']}s")
        print(f"Avg shots/rally: {rally_summary['avg_shots_per_rally']}")
        print(f"Longest rally: {rally_summary['longest_rally_s']}s")
        print(f"Total bounces detected: {len(all_bounces)}")
        if ball_speeds:
            print(f"Avg ball speed: {sum(ball_speeds)/len(ball_speeds):.0f} km/h")
            print(f"Max ball speed: {max(ball_speeds):.0f} km/h")
        print(f"Court lines detected: {len(court_lines_detected)}")

        # Save enhanced summary JSON
        try:
            import json
            enhanced_data = {
                "rally_summary": rally_summary,
                "total_bounces": len(all_bounces),
                "avg_ball_speed_kmh": round(sum(ball_speeds) / max(1, len(ball_speeds)), 1) if ball_speeds else 0,
                "max_ball_speed_kmh": round(max(ball_speeds), 1) if ball_speeds else 0,
                "court_lines_found": len(court_lines_detected),
                "shot_history_count": len(shot_history),
            }
            with open("output/enhanced_tracking_summary.json", "w") as f:
                json.dump(enhanced_data, f, indent=2, default=str)
            print("Saved enhanced_tracking_summary.json")
        except Exception as e:
            print(f"Could not save enhanced summary: {e}")

        # ── Generate output visualizations for the web UI ──
        try:
            import json as _json

            # 1. Player heatmaps (from accumulated ankle data on heatmap_image)
            blurred = cv2.GaussianBlur(heatmap_image, (51, 51), 0)
            norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            heatmap_colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            cv2.imwrite("output/player_1_heatmap.png", heatmap_colored)
            cv2.imwrite("output/player_2_heatmap.png", heatmap_colored)
            print("Saved player heatmaps")

            # 2. Ball heatmap
            ball_heat = np.zeros((height, width), dtype=np.float32)
            for bx, by, *_ in ballxy:
                if 0 <= bx < width and 0 <= by < height:
                    cv2.circle(ball_heat, (int(bx), int(by)), 8, 1.0, -1)
            ball_heat_blur = cv2.GaussianBlur(ball_heat, (51, 51), 0)
            ball_heat_norm = cv2.normalize(ball_heat_blur, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            ball_heat_color = cv2.applyColorMap(ball_heat_norm, cv2.COLORMAP_HOT)
            cv2.imwrite("output/ball_heatmap.png", ball_heat_color)
            print("Saved ball heatmap")

            # 3. Shot distribution pie chart
            if shot_history:
                shot_types = {}
                for sh in shot_history:
                    st = sh.get("shot_type", "unknown")
                    shot_types[st] = shot_types.get(st, 0) + 1
                plt.figure(figsize=(8, 6))
                plt.pie(shot_types.values(), labels=shot_types.keys(), autopct="%1.1f%%")
                plt.title("Shot Distribution")
                plt.savefig("output/shot_distribution.png", bbox_inches="tight")
                plt.close()
                print("Saved shot distribution")

                # 4. Shot success rate bar chart
                plt.figure(figsize=(8, 6))
                plt.bar(shot_types.keys(), shot_types.values(), color="orange")
                plt.title("Shot Frequency")
                plt.xlabel("Shot Type")
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig("output/shot_success_rate.png", bbox_inches="tight")
                plt.close()
                print("Saved shot success rate")

            # 5. T-position distance (copy to expected name)
            if p1distancesfromT and p2distancesfromT:
                plt.figure(figsize=(10, 6))
                plt.plot(p1distancesfromT, color="blue", label="P1 Distance from T")
                plt.plot(p2distancesfromT, color="red", label="P2 Distance from T")
                plt.xlabel("Frame")
                plt.ylabel("Distance")
                plt.title("T-Position Distance Over Time")
                plt.legend()
                plt.savefig("output/t_position_distance.png", bbox_inches="tight")
                plt.close()
                print("Saved T-position distance chart")

            # 6. Match data summary JSON
            summary = {
                "total_frames": running_frame,
                "data_points": len(ballxy) + running_frame * 2,
                "players_tracked": 2,
                "ball_detections": len(ballxy),
                "total_shots": len(shot_history),
                "rally_summary": rally_tracker.get_summary() if rally_tracker else {},
                "avg_ball_speed_kmh": round(sum(ball_speeds) / max(1, len(ball_speeds)), 1) if ball_speeds else 0,
                "max_ball_speed_kmh": round(max(ball_speeds), 1) if ball_speeds else 0,
            }
            with open("output/match_data_summary.json", "w") as mf:
                _json.dump(summary, mf, indent=2, default=str)
            print("Saved match_data_summary.json")

        except Exception as viz_err:
            print(f"Warning: Could not generate some visualizations: {viz_err}")

        cap.release()
        try:
            out.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"error2: {e}")
        print(f"line was {e.__traceback__.tb_lineno}")
        print(f"other into about e: {e.__traceback__}")
        print(f"other info about e: {e.__traceback__.tb_frame}")
        print(f"other info about e: {e.__traceback__.tb_next}")
        print(f"other info about e: {e.__traceback__.tb_lasti}")


if __name__ == "__main__":
    try:
        main()
    # get keyboarinterrupt error
    except KeyboardInterrupt:
        print("keyboard interrupt")

        exit()
    except Exception:
        # print(f"error: {e}")
        pass
