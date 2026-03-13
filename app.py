import os
import json
import threading
import time
import glob
import subprocess
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, Response, make_response
from flask_cors import CORS

# Optional Supabase integration
try:
    from supabase import create_client, Client as SupabaseClient
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")  # Use service key for backend
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Optional[SupabaseClient] = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"[OK] Supabase connected: {SUPABASE_URL[:40]}")
    else:
        supabase = None
        print("[WARN] Supabase env vars not set - running in local-only mode")
except ImportError:
    supabase = None
    print("[WARN] supabase-py not installed - running in local-only mode")
    print("       Install with: pip install supabase")


app = Flask(__name__)
CORS(app)  # Allow the Vite dev server to call us

# Enable request logging so we can see every request that hits Flask
import logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

@app.after_request
def log_request(response):
    app.logger.info(f"{request.method} {request.path} → {response.status_code}")
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Simple in-memory job status
JOB = {
    "running": False,
    "started_at": None,
    "ended_at": None,
    "video_path": None,
    "error": None,
    "analysis_id": None,  # Supabase row ID
    "video_ready": False,  # Only True when a fresh analysis has completed
}


def push_to_supabase(analysis_id: str) -> None:
    """After processing, push all results to Supabase."""
    if not supabase or not analysis_id:
        return

    try:
        # 1. Read coaching report text
        coaching_report = None
        for txt_path in [
            os.path.join(OUTPUT_DIR, "enhanced_autonomous_coaching_report.txt"),
            os.path.join(OUTPUT_DIR, "autonomous_coaching_report.txt"),
            os.path.join(OUTPUT_DIR, "final.txt"),
            os.path.join(BASE_DIR, "final.txt"),
        ]:
            if os.path.isfile(txt_path):
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    coaching_report = f.read()[:50000]  # Limit size
                break

        # 2. Read match summary JSON
        match_summary = None
        summary_path = os.path.join(OUTPUT_DIR, "match_data_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                match_summary = json.load(f)

        # 3. Read enhanced coaching data (sample – full data can be very large)
        coaching_data = None
        coaching_path = os.path.join(OUTPUT_DIR, "enhanced_coaching_data.json")
        if os.path.isfile(coaching_path):
            with open(coaching_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
                # Store first + last 50 frames as summary, plus metadata
                if isinstance(full_data, list) and len(full_data) > 100:
                    coaching_data = {
                        "total_frames": len(full_data),
                        "sample_start": full_data[:50],
                        "sample_end": full_data[-50:],
                    }
                else:
                    coaching_data = full_data

        # 4. Count total frames
        total_frames = None
        if match_summary and "total_frames" in match_summary:
            total_frames = match_summary["total_frames"]
        elif coaching_data and isinstance(coaching_data, dict):
            total_frames = coaching_data.get("total_frames")

        # 5. Update the Supabase record
        update_data = {
            "status": "completed",
            "coaching_report": coaching_report,
            "match_summary": match_summary,
            "coaching_data": coaching_data,
            "total_frames": total_frames,
        }
        supabase.table("analyses").update(update_data).eq("id", analysis_id).execute()
        print(f"[OK] Supabase record updated: {analysis_id}")

        # 6. Upload output images to Supabase Storage
        image_types = [
            "player_1_heatmap", "player_2_heatmap",
            "shot_distribution", "shot_success_rate",
            "t_position_distance", "ball_heatmap",
            "shot_origins", "zone_occupancy", "front_wall_hits", "floor_bounces",
            "3d_heatmap_player_1", "3d_heatmap_player_2",
        ]
        for img_type in image_types:
            img_path = os.path.join(OUTPUT_DIR, f"{img_type}.png")
            if os.path.isfile(img_path):
                try:
                    storage_path = f"images/{analysis_id}/{img_type}.png"
                    with open(img_path, "rb") as img_file:
                        supabase.storage.from_("squash-files").upload(
                            storage_path, img_file.read(),
                            file_options={"content-type": "image/png", "upsert": "true"}
                        )
                    # Get public URL and save to analysis_images
                    public_url = supabase.storage.from_("squash-files").get_public_url(storage_path)
                    supabase.table("analysis_images").insert({
                        "analysis_id": analysis_id,
                        "image_type": img_type,
                        "image_url": public_url,
                    }).execute()
                    print(f"   [IMG] Uploaded {img_type}")
                except Exception as e:
                    print(f"   [WARN] Failed to upload {img_type}: {e}")

        # 7. Upload annotated video
        annotated_path = os.path.join(OUTPUT_DIR, "annotated.mp4")
        if os.path.isfile(annotated_path):
            try:
                vid_storage_path = f"videos/{analysis_id}/annotated.mp4"
                with open(annotated_path, "rb") as vid_file:
                    supabase.storage.from_("squash-files").upload(
                        vid_storage_path, vid_file.read(),
                        file_options={"content-type": "video/mp4", "upsert": "true"}
                    )
                vid_url = supabase.storage.from_("squash-files").get_public_url(vid_storage_path)
                supabase.table("analyses").update({
                    "annotated_video_url": vid_url
                }).eq("id", analysis_id).execute()
                print(f"   [VID] Uploaded annotated video")
            except Exception as e:
                print(f"   [WARN] Failed to upload video: {e}")

        print(f"[DONE] All results pushed to Supabase for analysis {analysis_id}")

    except Exception as e:
        print(f"[ERROR] Error pushing to Supabase: {e}")


def clean_output_dir() -> None:
    """Remove old analysis outputs before a new run, keeping reference files."""
    # Files that must NOT be deleted (reference images, model weights, etc.)
    keep_files = {"white.png", "court-topdown.png", "court-topdown.svg",
                  "annotated-squash-court.png", "yolo11n.pt"}
    extensions = (".mp4", ".png", ".txt", ".csv", ".json")
    for f in os.listdir(OUTPUT_DIR):
        if f.lower() in {k.lower() for k in keep_files}:
            continue
        if f.lower().endswith(extensions):
            try:
                os.remove(os.path.join(OUTPUT_DIR, f))
            except Exception:
                pass
    print("[OK] Cleared old output files (kept reference files)")


def run_analysis(video_path: str, analysis_id: Optional[str] = None,
                 width: int = 640, height: int = 360) -> None:
    try:
        import importlib
        import get_data  # local import so Flask can start without heavy deps loading

        JOB["running"] = True
        JOB["error"] = None
        JOB["started_at"] = time.time()
        JOB["video_path"] = video_path
        JOB["analysis_id"] = analysis_id

        # Clean old outputs so we don't serve stale results
        clean_output_dir()

        # Reload get_data module to reset all global state
        importlib.reload(get_data)

        # Update Supabase status to processing
        if supabase and analysis_id:
            supabase.table("analyses").update(
                {"status": "processing"}
            ).eq("id", analysis_id).execute()

        get_data.main(video_path, width, height)

        # Re-encode video to H.264 for browser playback
        reencode_video_to_h264()

        # Push results to Supabase
        push_to_supabase(analysis_id or "")

        # Mark video as ready ONLY after successful analysis
        JOB["video_ready"] = True

    except Exception as e:
        JOB["error"] = str(e)
        if supabase and analysis_id:
            try:
                supabase.table("analyses").update({
                    "status": "error",
                    "error_message": str(e),
                }).eq("id", analysis_id).execute()
            except Exception:
                pass
    finally:
        JOB["running"] = False
        JOB["ended_at"] = time.time()


@app.route("/")
def index():
    last_ready = os.path.isfile(os.path.join(OUTPUT_DIR, "annotated.mp4"))
    return render_template("index.html", job=JOB, last_ready=last_ready)


@app.route("/upload", methods=["POST"])
def upload():
    print("[UPLOAD] Received upload request")
    if "video" not in request.files:
        print("[UPLOAD] No video file in request")
        return jsonify({"error": "No video file"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save upload
    safe_name = f.filename.replace("..", "_")
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(save_path)
    print(f"[UPLOAD] Saved video to {save_path} ({os.path.getsize(save_path)} bytes)")

    # IMMEDIATELY clear old outputs so /status returns video_ready=false
    clean_output_dir()
    print("[UPLOAD] Cleared old outputs")

    # Get analysis_id from form data (sent by the web frontend)
    analysis_id = request.form.get("analysis_id") or request.args.get("analysis_id")

    # If no analysis_id provided but Supabase is available, create one
    if not analysis_id and supabase:
        try:
            result = supabase.table("analyses").insert({
                "video_name": safe_name,
                "status": "pending",
            }).execute()
            analysis_id = result.data[0]["id"]
        except Exception as e:
            print(f"[WARN] Could not create Supabase record: {e}")

    # Kick off background job (always allow new uploads — override stale state)
    JOB["running"] = True
    JOB["error"] = None
    JOB["video_ready"] = False  # Reset — only set True when this analysis finishes
    JOB["analysis_id"] = analysis_id
    t = threading.Thread(
        target=run_analysis,
        args=(save_path, analysis_id),
        daemon=True,
    )
    t.start()
    print(f"[UPLOAD] Started analysis thread for {safe_name}")

    # Return JSON for the API, redirect for the old HTML form
    if request.content_type and "json" in request.content_type:
        return jsonify({"ok": True, "analysis_id": analysis_id})
    if request.headers.get("Accept", "").startswith("application/json"):
        return jsonify({"ok": True, "analysis_id": analysis_id})
    return redirect(url_for("result"))


@app.route("/result")
def result():
    txt_paths = [
        os.path.join(OUTPUT_DIR, "final.txt"),
        os.path.join(BASE_DIR, "final.txt"),
    ]
    final_text: Optional[str] = None
    for p in txt_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    data = fh.read()
                final_text = data if len(data) < 200_000 else data[:200_000] + "\n... (truncated)"
                break
            except Exception:
                pass

    video_ready = os.path.isfile(os.path.join(OUTPUT_DIR, "annotated.mp4"))
    return render_template("result.html", job=JOB, video_ready=video_ready, final_text=final_text)


@app.route("/result-text")
def result_text():
    """Return the best available coaching report, combining multiple sources."""
    parts = []

    # 1. Match report (shot distribution, success rates, T-position)
    match_report_path = os.path.join(OUTPUT_DIR, "match_report.txt")
    if os.path.isfile(match_report_path):
        with open(match_report_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content:
                parts.append(content)

    # 2. Autonomous coaching report (filter out processing logs)
    coaching_path = os.path.join(OUTPUT_DIR, "autonomous_coaching_report.txt")
    if os.path.isfile(coaching_path):
        with open(coaching_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            # Filter out processing logs, warnings, GPU messages
            skip_patterns = [
                "Loading", "checkpoint", "WARNING", "UserWarning",
                "GPU memory", "coach model loaded", "Coach compute device",
                "accelerate", "generation error", "coaching analysis completed",
                "Enhanced report saved", "Enhanced data saved", "bounces analyzed",
                "GENERATING", "Loading enhanced", "output/ directory",
                "processing complete", "Traditional coaching", "Check output/",
                "enhanced_", "annotated.mp4", "Other traditional",
                "warnings.warn", "wrapper_CUDA", "at least two devices",
                "charmap", "codec can't encode", "INFO -",
                "?", "??", "100%|",
            ]
            lines = []
            for l in content.split("\n"):
                stripped = l.strip()
                if not stripped:
                    continue
                if any(p in stripped for p in skip_patterns):
                    continue
                if stripped.startswith("===") or stripped.startswith("---"):
                    lines.append(l)
                    continue
                lines.append(l)
            filtered = "\n".join(lines)
            # Only include if there's meaningful content beyond headers
            meaningful = [l for l in lines if not l.strip().startswith("=") and not l.strip().startswith("-") and len(l.strip()) > 5]
            if len(meaningful) > 3:
                parts.append(filtered)

    # 3. Frame analysis (first 50 + last 20 frames as sample)
    frame_path = os.path.join(OUTPUT_DIR, "frame_analysis.txt")
    if os.path.isfile(frame_path):
        with open(frame_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content:
                frames = content.split("--------------------------------------------------")
                total = len(frames)
                if total > 70:
                    sample = frames[:50] + [f"\n... ({total - 70} more frames) ...\n"] + frames[-20:]
                    parts.append("FRAME-BY-FRAME ANALYSIS (sample)\n" + "--------------------------------------------------".join(sample))
                else:
                    parts.append("FRAME-BY-FRAME ANALYSIS\n" + content)

    if parts:
        combined = "\n\n" + "=" * 50 + "\n\n".join(parts)
        return jsonify({"ok": True, "text": combined})

    # Fallback to any .txt file
    for p in [os.path.join(OUTPUT_DIR, "final.txt"), os.path.join(BASE_DIR, "final.txt")]:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return jsonify({"ok": True, "text": f.read()})

    return jsonify({"ok": False, "error": "No report text found"}), 404


@app.route("/api/coaching-context")
def api_coaching_context():
    """Return structured coaching context for the AI chatbot — real data only."""
    context_parts = []

    # Match report
    match_report_path = os.path.join(OUTPUT_DIR, "match_report.txt")
    if os.path.isfile(match_report_path):
        with open(match_report_path, "r", encoding="utf-8", errors="ignore") as f:
            context_parts.append(f"MATCH REPORT:\n{f.read().strip()}")

    # Frame analysis (sample for context window)
    frame_path = os.path.join(OUTPUT_DIR, "frame_analysis.txt")
    if os.path.isfile(frame_path):
        with open(frame_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            # Take first 30 + last 10 frames for AI context
            frames = content.split("--------------------------------------------------")
            if len(frames) > 40:
                sample = frames[:30] + frames[-10:]
            else:
                sample = frames
            context_parts.append(f"FRAME-BY-FRAME ANALYSIS (sample of {len(sample)} frames):\n" +
                                "--------------------------------------------------".join(sample))

    # Enhanced coaching data summary
    coaching_path = os.path.join(OUTPUT_DIR, "enhanced_coaching_data.json")
    if os.path.isfile(coaching_path):
        with open(coaching_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                # Count stats
                hits = sum(1 for d in data if d.get("ball_hit_detected"))
                active_frames = sum(1 for d in data if d.get("match_active"))
                movement_frames = sum(1 for d in data if d.get("player_movement"))
                shot_types = {}
                for d in data:
                    st = d.get("shot_type", [])
                    if st and len(st) >= 2:
                        key = f"{st[0]} {st[1]}"
                        shot_types[key] = shot_types.get(key, 0) + 1
                context_parts.append(
                    f"COACHING DATA SUMMARY:\n"
                    f"- Total frames: {len(data)}\n"
                    f"- Ball hits detected: {hits}\n"
                    f"- Match active frames: {active_frames}\n"
                    f"- Player movement frames: {movement_frames}\n"
                    f"- Shot type distribution: {json.dumps(shot_types, indent=2)}"
                )

    if context_parts:
        return jsonify({"ok": True, "context": "\n\n".join(context_parts)})
    return jsonify({"ok": False, "error": "No coaching data found"}), 404


@app.route("/status")
def status():
    # Use JOB flag instead of file existence — avoids stale results from
    # old files that Windows can't delete (file locking)
    resp = jsonify({
        "running": JOB["running"],
        "error": JOB["error"],
        "video_ready": JOB["video_ready"],
        "analysis_id": JOB.get("analysis_id"),
    })
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/outputs/<path:filename>")
def outputs(filename: str):
    """Serve output files with range request support for video playback."""
    filepath = os.path.join(OUTPUT_DIR, filename)

    # If requesting annotated.mp4, prefer the H.264 re-encoded version
    if filename == "annotated.mp4":
        h264_path = os.path.join(OUTPUT_DIR, "annotated_h264.mp4")
        if os.path.isfile(h264_path) and os.path.getsize(h264_path) > 100000:
            filepath = h264_path

    if not os.path.isfile(filepath):
        return "Not found", 404

    file_size = os.path.getsize(filepath)

    # Determine MIME type
    if filename.endswith(".mp4"):
        mime = "video/mp4"
    elif filename.endswith(".png"):
        mime = "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        mime = "image/jpeg"
    elif filename.endswith(".json"):
        mime = "application/json"
    else:
        mime = "application/octet-stream"

    # Handle range requests (required for video seeking in browsers)
    range_header = request.headers.get("Range")
    if range_header and filename.endswith(".mp4"):
        byte_start = 0
        byte_end = file_size - 1

        range_match = range_header.replace("bytes=", "").split("-")
        if range_match[0]:
            byte_start = int(range_match[0])
        if len(range_match) > 1 and range_match[1]:
            byte_end = int(range_match[1])

        content_length = byte_end - byte_start + 1

        def generate():
            with open(filepath, "rb") as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        resp = Response(generate(), status=206, mimetype=mime)
        resp.headers["Content-Range"] = f"bytes {byte_start}-{byte_end}/{file_size}"
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = str(content_length)
        return resp

    # Non-range request — serve the full file
    resp = make_response(send_from_directory(OUTPUT_DIR, filename, as_attachment=False))
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Type"] = mime
    # Prevent browser caching so new analysis results always show
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


# ── New API endpoints for the enhanced frontend ──────────────

@app.route("/api/coaching-data")
def api_coaching_data():
    """Return the enhanced coaching data as JSON."""
    for path in [
        os.path.join(OUTPUT_DIR, "enhanced_coaching_data.json"),
        os.path.join(OUTPUT_DIR, "detailed_coaching_data.json"),
    ]:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return jsonify({"ok": True, "data": data})
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": False, "error": "No coaching data found"}), 404


@app.route("/api/match-summary")
def api_match_summary():
    """Return match summary JSON."""
    path = os.path.join(OUTPUT_DIR, "match_data_summary.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"ok": True, "data": data})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": False, "error": "No match summary found"}), 404


@app.route("/api/available-outputs")
def api_available_outputs():
    """List all available output files."""
    files = []
    if os.path.isdir(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            fpath = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fpath):
                files.append({
                    "name": f,
                    "size": os.path.getsize(fpath),
                    "url": f"/outputs/{f}",
                })
    return jsonify({"ok": True, "files": files})


def reencode_video_to_h264(force: bool = False):
    """Re-encode the annotated video from FMP4 to H.264 for browser compatibility."""
    original = os.path.join(OUTPUT_DIR, "annotated.mp4")
    h264_path = os.path.join(OUTPUT_DIR, "annotated_h264.mp4")

    if not os.path.isfile(original):
        return

    # Already re-encoded and not forced?
    if not force and os.path.isfile(h264_path) and os.path.getsize(h264_path) > 100000:
        # Check if h264 is newer than original
        if os.path.getmtime(h264_path) >= os.path.getmtime(original):
            print(f"[OK] H.264 video already exists: {h264_path}")
            return

    # Find the ffmpeg binary bundled with imageio-ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("[WARN] imageio-ffmpeg not installed, cannot re-encode video")
        return

    print(f"[...] Re-encoding video to H.264 for browser playback...")
    try:
        result = subprocess.run([
            ffmpeg_exe,
            "-i", original,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",  # no audio
            "-y",
            h264_path,
        ], capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.isfile(h264_path):
            print(f"[OK] Video re-encoded successfully: {h264_path} ({os.path.getsize(h264_path)} bytes)")
        else:
            print(f"[WARN] FFmpeg returned code {result.returncode}")
            if result.stderr:
                print(result.stderr[-500:])
    except Exception as e:
        print(f"[WARN] Video re-encoding failed: {e}")


if __name__ == "__main__":
    # Re-encode video at startup
    reencode_video_to_h264()

    # Run: .\.venv\Scripts\python.exe app.py
    # Then visit http://127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=False)
