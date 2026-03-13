# CourtSense – Vercel Deployment Guide

Step-by-step guide to deploy the frontend to Vercel (free).

---

## Prerequisites

- GitHub account
- Code pushed to a GitHub repo (e.g. `github.com/arjun-biy/visionaj`)
- [Vercel account](https://vercel.com/signup) (free)

---

## Step 1: Push Your Code to GitHub

If you haven’t already:

```bash
cd c:\Users\arjun\Squash
git add .
git commit -m "Add Vercel deployment config"
git push origin main
```

---

## Step 2: Go to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Sign in with GitHub
3. Click **Add New…** → **Project**

---

## Step 3: Import Your Repository

1. Select your repo (e.g. `arjun-biy/visionaj`)
2. Click **Import**
3. On the project settings page, configure the build settings (Step 4)

---

## Step 4: Project Settings

Set these **before** clicking Deploy:

| Setting | Value |
|--------|--------|
| **Framework Preset** | Vite |
| **Root Directory** | `vision/web` (or the path to the folder with `package.json`) |
| **Build Command** | `npm run build` |
| **Output Directory** | `dist` |
| **Install Command** | `npm install` |

**Root Directory:**  
- If the repo root has `vision/web/`, set `vision/web`  
- If the repo root is directly the web app, leave Root Directory empty

---

## Step 5: Environment Variables (Optional)

If you use Supabase, add these under **Environment Variables**:

| Name | Value |
|------|-------|
| `VITE_SUPABASE_URL` | Your Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Your Supabase anonymous key |

If you later add a hosted backend (e.g. Render), add:

| Name | Value |
|------|-------|
| `VITE_API_URL` | Your backend URL (e.g. `https://your-app.onrender.com`) |

---

## Step 6: Deploy

1. Click **Deploy**
2. Wait for the build to finish (usually 1–2 minutes)
3. Vercel will give you a URL like `https://your-project.vercel.app`

---

## Step 7: Verify the Deployment

1. Open the deployment URL
2. You should see the CourtSense landing page
3. Video upload/analysis will not work until the backend is deployed (see below)

---

## Troubleshooting

### Build uses Python instead of Node.js

- Ensure **Root Directory** is set to `vision/web` so Vercel builds the frontend, not the Python backend.

### Build fails with "Cannot find module"

- Confirm Root Directory points to the folder with `package.json`.
- Re-check the repo structure and adjust Root Directory if needed.

### Deployed app shows 404 on refresh

- Vercel should handle SPA routing with the default Vite config. If you still see issues, add a `vercel.json` in the root of `vision/web`:

```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

---

## Backend (Video Processing)

The Flask backend with YOLO/ML is **not** deployed on Vercel. Vercel only hosts the frontend.

To enable video upload and analysis online:

1. Deploy the backend elsewhere (e.g. [Render](https://render.com) or [Railway](https://railway.app))
2. Get the backend URL (e.g. `https://courtsense-api.onrender.com`)
3. In Vercel: **Settings → Environment Variables**
4. Add `VITE_API_URL` = `https://courtsense-api.onrender.com`
5. Redeploy the frontend

---

## Summary

| Step | Action |
|------|--------|
| 1 | Push code to GitHub |
| 2 | Go to vercel.com, sign in |
| 3 | Import your repo |
| 4 | Set Root Directory to `vision/web` |
| 5 | Add env vars (optional) |
| 6 | Deploy |
