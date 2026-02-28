export interface Analysis {
  id: string;
  created_at: string;
  video_name: string;
  status: "pending" | "processing" | "completed" | "error";
  total_frames: number | null;
  coaching_report: string | null;
  match_summary: Record<string, unknown> | null;
  coaching_data: Record<string, unknown>[] | null;
  error_message: string | null;
  video_url: string | null;
  annotated_video_url: string | null;
}

export interface AnalysisImage {
  id: string;
  analysis_id: string;
  image_type: string;
  image_url: string;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  analysis_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface Database {
  public: {
    Tables: {
      analyses: {
        Row: Analysis;
        Insert: Omit<Analysis, "id" | "created_at">;
        Update: Partial<Omit<Analysis, "id" | "created_at">>;
      };
      analysis_images: {
        Row: AnalysisImage;
        Insert: Omit<AnalysisImage, "id" | "created_at">;
        Update: Partial<Omit<AnalysisImage, "id" | "created_at">>;
      };
      chat_messages: {
        Row: ChatMessage;
        Insert: Omit<ChatMessage, "id" | "created_at">;
        Update: Partial<Omit<ChatMessage, "id" | "created_at">>;
      };
    };
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
  };
}

