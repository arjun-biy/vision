/**
 * API base URL for production. In dev, empty string so Vite proxy works.
 * Set VITE_API_URL in Vercel (e.g. https://your-backend.onrender.com) when backend is deployed.
 */
export function getApiBase(): string {
  const url = import.meta.env.VITE_API_URL;
  if (!url || typeof url !== "string") return "";
  return url.replace(/\/$/, ""); // trailing slash removed
}
