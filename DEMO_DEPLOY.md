# CourtSense – Demo Deployment

Choose based on your needs: **real deployment** (recommended when tunnels fail) or **local + tunnel**.

---

## Option C: Real Deployment (Recommended – No Tunnels)

Deploy the app on the internet. No tunnels, no SSL issues. Frontend on Vercel, backend on Render.

### 1. Push to GitHub

```bash
cd c:\Users\arjun\Squash\vision
git add .
git commit -m "Add Render deployment"
git push origin main
```

### 2. Deploy Backend on Render

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo (`arjun-biy/vision` or your repo)
3. Configure:
   - **Name:** `courtsense-api` (or any name)
   - **Root Directory:** leave blank (or `vision` if your repo root is `Squash`)
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
4. **Environment Variables:**
   - `SUPABASE_URL` = your Supabase project URL
   - `SUPABASE_SERVICE_KEY` = your Supabase service role key
5. Click **Create Web Service**
6. Wait for deploy. Copy the URL (e.g. `https://courtsense-api.onrender.com`)

### 3. Point Vercel to the Backend

1. Vercel → your project → **Settings** → **Environment Variables**
2. Add: `VITE_API_URL` = `https://courtsense-api.onrender.com` (no trailing slash)
3. **Redeploy** the frontend (Deployments → ⋮ → Redeploy)

### 4. Test

Open your Vercel URL. The app should load and call the Render backend over normal internet paths.

**Note:** Video processing uses heavy ML (YOLO, TensorFlow). Render free tier (512MB) may fail for processing. The API will still run; upload may return errors until you use a paid plan or run processing locally.

---

## Option A: One-Click Demo (Local + ngrok)

Expose your full local stack (frontend + backend) with one URL.

### Steps

1. **Start your app locally**
   ```bash
   cd c:\Users\arjun\Squash\vision
   python app.py
   ```
   In another terminal:
   ```bash
   cd c:\Users\arjun\Squash\vision\web
   npm run dev
   ```

2. **Install ngrok** (if needed)
   - Download from [ngrok.com/download](https://ngrok.com/download)
   - Or: `winget install ngrok`

3. **Expose the frontend**
   ```bash
   ngrok http 5173
   ```

4. **Share the URL** – ngrok shows something like `https://abc123.ngrok-free.app`
   - Open that URL in your browser
   - The frontend is served by Vite and proxies API calls to Flask
   - Upload, analysis, and video playback all work

5. **When you're done** – press Ctrl+C in the ngrok terminal to stop

---

## Option B: Vercel Frontend + ngrok Backend

If you prefer the frontend to stay on Vercel:

1. **Start Flask locally**
   ```bash
   cd c:\Users\arjun\Squash\vision
   python app.py
   ```

2. **Expose Flask with ngrok**
   ```bash
   ngrok http 5000
   ```

3. **Copy the ngrok URL** (e.g. `https://xyz.ngrok-free.app`)

4. **In Vercel** → Settings → Environment Variables:
   - Add `VITE_API_URL` = `https://xyz.ngrok-free.app` (no trailing slash)
   - Redeploy

5. **Use the Vercel URL** – it will call your local backend via ngrok

**Note:** On the free ngrok plan, the URL changes each time you restart ngrok, so you'll need to update `VITE_API_URL` in Vercel and redeploy after each new session.

---

## Summary

| Option | Effort | Use case |
|--------|--------|----------|
| **C: Render backend** | Highest | Stable, persistent, no tunnels; recommended when tunnels fail |
| **A: ngrok on 5173** | Lowest | Quick demos, share one link |
| **B: Vercel + ngrok on 5000** | Medium | Prefer Vercel URL, accept updating backend URL |

When tunnels fail (SSL errors), use **Option C**. For quick local demos, **Option A**.
