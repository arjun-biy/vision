import { getApiBase } from "./apiBase";
import { getSupabase, isSupabaseConfigured } from "./supabase";
import type { Analysis, AnalysisImage, ChatMessage } from "@/types/supabase";

// ── Analyses ────────────────────────────────────────────────

export async function fetchAnalyses(): Promise<Analysis[]> {
  if (!isSupabaseConfigured) return [];
  const { data, error } = await getSupabase()
    .from("analyses")
    .select("*")
    .order("created_at", { ascending: false });
  if (error) throw error;
  return (data ?? []) as unknown as Analysis[];
}

export async function fetchAnalysis(id: string): Promise<Analysis> {
  if (!isSupabaseConfigured) throw new Error("Supabase not configured");
  const { data, error } = await getSupabase()
    .from("analyses")
    .select("*")
    .eq("id", id)
    .single();
  if (error) throw error;
  return data as unknown as Analysis;
}

export async function createAnalysis(videoName: string): Promise<Analysis> {
  if (!isSupabaseConfigured) {
    // Return a fake analysis so the upload flow still works via Flask
    return {
      id: crypto.randomUUID(),
      created_at: new Date().toISOString(),
      video_name: videoName,
      status: "pending",
      total_frames: null,
      coaching_report: null,
      match_summary: null,
      coaching_data: null,
      error_message: null,
      video_url: null,
      annotated_video_url: null,
    };
  }
  const { data, error } = await getSupabase()
    .from("analyses")
    .insert({ video_name: videoName, status: "pending" } as Record<string, unknown>)
    .select()
    .single();
  if (error) throw error;
  return data as unknown as Analysis;
}

export async function updateAnalysis(
  id: string,
  updates: Partial<Analysis>
): Promise<void> {
  if (!isSupabaseConfigured) return;
  const { error } = await getSupabase()
    .from("analyses")
    .update(updates as Record<string, unknown>)
    .eq("id", id);
  if (error) throw error;
}

// ── Analysis images ─────────────────────────────────────────

export async function fetchAnalysisImages(
  analysisId: string
): Promise<AnalysisImage[]> {
  if (!isSupabaseConfigured) return [];
  const { data, error } = await getSupabase()
    .from("analysis_images")
    .select("*")
    .eq("analysis_id", analysisId);
  if (error) throw error;
  return (data ?? []) as unknown as AnalysisImage[];
}

// ── Chat ────────────────────────────────────────────────────

export async function fetchChatHistory(
  analysisId: string
): Promise<ChatMessage[]> {
  if (!isSupabaseConfigured) return [];
  const { data, error } = await getSupabase()
    .from("chat_messages")
    .select("*")
    .eq("analysis_id", analysisId)
    .order("created_at", { ascending: true });
  if (error) throw error;
  return (data ?? []) as unknown as ChatMessage[];
}

export async function saveChatMessage(
  analysisId: string,
  role: "user" | "assistant",
  content: string
): Promise<ChatMessage> {
  if (!isSupabaseConfigured) {
    return {
      id: crypto.randomUUID(),
      analysis_id: analysisId,
      role,
      content,
      created_at: new Date().toISOString(),
    };
  }
  const { data, error } = await getSupabase()
    .from("chat_messages")
    .insert({
      analysis_id: analysisId,
      role,
      content,
    } as Record<string, unknown>)
    .select()
    .single();
  if (error) throw error;
  return data as unknown as ChatMessage;
}

// ── Storage helpers ─────────────────────────────────────────

export async function uploadFile(
  bucket: string,
  path: string,
  file: File
): Promise<string> {
  if (!isSupabaseConfigured) throw new Error("Supabase not configured");
  const { data, error } = await getSupabase().storage
    .from(bucket)
    .upload(path, file, { upsert: true });
  if (error) throw error;
  const {
    data: { publicUrl },
  } = getSupabase().storage.from(bucket).getPublicUrl(data.path);
  return publicUrl;
}

// ── Legacy Flask proxy (still used for video processing) ────

export async function getFlaskStatus() {
  const res = await fetch(`${getApiBase()}/status`, { cache: "no-store" });
  return res.json() as Promise<{
    running: boolean;
    error: string | null;
    video_ready: boolean;
    analysis_id?: string;
  }>;
}

export async function uploadVideoToFlask(file: File, analysisId?: string) {
  const form = new FormData();
  form.append("video", file);
  if (analysisId) {
    form.append("analysis_id", analysisId);
  }
  const res = await fetch(`${getApiBase()}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Upload failed");
}

export async function getFlaskText() {
  const res = await fetch(`${getApiBase()}/result-text`);
  return res.json() as Promise<{
    ok: boolean;
    text?: string;
    error?: string;
  }>;
}
