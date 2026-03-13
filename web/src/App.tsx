import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  Activity,
  Clock,
  Video,
  ChevronRight,
  Loader2,
  Zap,
  Eye,
  Users,
  LogOut,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { resetLocalChat } from "@/components/ChatPanel";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";

import {
  fetchAnalyses,
  createAnalysis,
  uploadFile,
  uploadVideoToFlask,
  getFlaskStatus,
  updateAnalysis,
} from "@/lib/api";
import { isSupabaseConfigured } from "@/lib/supabase";
import type { Analysis } from "@/types/supabase";

export default function App() {
  const navigate = useNavigate();
  const { signOut } = useAuth();
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [activeAnalysisId, setActiveAnalysisId] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Player names
  const [player1Name, setPlayer1Name] = useState(
    () => localStorage.getItem("squash_player1") || "Me"
  );
  const [player2Name, setPlayer2Name] = useState(
    () => localStorage.getItem("squash_player2") || "Opponent"
  );

  // Save player names to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem("squash_player1", player1Name);
    localStorage.setItem("squash_player2", player2Name);
  }, [player1Name, player2Name]);

  // Local-only mode state
  const [flaskReady, setFlaskReady] = useState(false);
  const [flaskVideoReady, setFlaskVideoReady] = useState(false);

  // Load past analyses from Supabase (if configured)
  const loadAnalyses = useCallback(async () => {
    try {
      const data = await fetchAnalyses();
      setAnalyses(data);
    } catch {
      // Supabase not configured — silently ignore
    }
  }, []);

  useEffect(() => {
    loadAnalyses();
  }, [loadAnalyses]);

  // Check Flask status on mount + poll while processing
  const checkFlask = useCallback(async () => {
    if (uploading) return; // Don't poll while upload is in flight
    try {
      const s = await getFlaskStatus();
      setFlaskReady(true);

      if (s.running) {
        // Flask is actively processing — track it, hide old results
        setProcessing(true);
        setFlaskVideoReady(false);
      } else if (processing) {
        // Transition: was processing, now done — show results
        setProcessing(false);
        if (s.error) setError(s.error);
        setFlaskVideoReady(s.video_ready);

        // Update Supabase status when Flask finishes
        const idToUpdate = s.analysis_id || activeAnalysisId;
        if (isSupabaseConfigured && idToUpdate) {
          try {
            await updateAnalysis(idToUpdate, {
              status: s.error ? "error" : "completed",
              ...(s.error ? { error_message: s.error } : {}),
            });
          } catch {
            // Supabase update failed silently — not critical
          }
        }

        // Refresh Supabase list too
        loadAnalyses();
      }
      // else: idle on page load — do NOT show stale results from old files
    } catch {
      setFlaskReady(false);
    }
  }, [processing, uploading, loadAnalyses]);

  useEffect(() => {
    checkFlask();
    const interval = setInterval(checkFlask, 3000);
    return () => clearInterval(interval);
  }, [checkFlask]);

  async function handleUpload() {
    if (!file) return;

    // IMMEDIATELY reset all result state so the UI shows "uploading"
    setError(null);
    setFlaskVideoReady(false);
    setUploading(true);
    resetLocalChat(); // Clear old chat so new analysis gets fresh chat

    try {
      // 1. Create Supabase record (fallback to local ID if Supabase fails)
      let analysisId = crypto.randomUUID();
      try {
        const analysis = await createAnalysis(file.name);
        analysisId = analysis.id;
      } catch {
        // Supabase not available or table missing — continue with local ID
        console.warn("Supabase createAnalysis failed, using local mode");
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      setActiveAnalysisId(analysisId as any);

      // 2. Upload video to Supabase Storage (if configured, non-blocking)
      if (isSupabaseConfigured) {
        try {
          const videoUrl = await uploadFile(
            "squash-files",
            `videos/${analysisId}/${file.name}`,
            file
          );
          await updateAnalysis(analysisId, { video_url: videoUrl });
          await updateAnalysis(analysisId, { status: "processing" });
        } catch {
          // Storage or table not available; continue with Flask
        }
      }

      // 3. Send to Flask for processing — this is the critical step
      await uploadVideoToFlask(file, analysisId);
      setProcessing(true);
      setFile(null);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setError(message);
      setProcessing(false);
      if (isSupabaseConfigured && activeAnalysisId) {
        try {
          await updateAnalysis(activeAnalysisId, {
            status: "error",
            error_message: message,
          });
        } catch {
          // ignore Supabase errors here
        }
      }
    } finally {
      setUploading(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped?.type.startsWith("video/")) setFile(dropped);
  }

  const statusColor: Record<string, string> = {
    pending: "secondary",
    processing: "default",
    completed: "default",
    error: "destructive",
  };

  return (
    <div className="min-h-screen bg-gradient-main">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-white/10 bg-black/70 backdrop-blur-lg">
        <div className="container flex h-16 items-center justify-between overflow-visible">
          <img src="/courtsense-logo.png" alt="CourtSense" className="h-44 w-auto object-contain mt-2 -ml-4" />
          <div className="flex items-center gap-2">
            {!flaskReady && (
              <Badge variant="destructive" className="gap-1.5">
                Flask Offline
              </Badge>
            )}
            <Badge variant="outline" className="gap-1.5">
              <Activity className="h-3 w-3" />
              {processing ? "Processing..." : "Ready"}
            </Badge>
            <Button variant="ghost" size="icon" onClick={signOut} title="Sign out">
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      <main className="container py-8 space-y-8">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Match Video
            </CardTitle>
            <CardDescription>
              Drop a match video to analyze player movement, ball tracking,
              shot classification, and get AI coaching insights.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-12 transition-colors ${
                dragActive
                  ? "border-primary bg-primary/5"
                  : "border-muted-foreground/25 hover:border-muted-foreground/50"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              <Video className="h-10 w-10 text-muted-foreground mb-4" />
              {file ? (
                <div className="text-center">
                  <p className="font-medium">{file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(file.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                </div>
              ) : (
                <div className="text-center">
                  <p className="text-muted-foreground">
                    Drag & drop your video here, or
                  </p>
                  <label className="mt-2 inline-block cursor-pointer text-primary hover:underline">
                    browse files
                    <input
                      type="file"
                      className="hidden"
                      accept="video/*"
                      onChange={(e) =>
                        setFile(e.target.files?.[0] ?? null)
                      }
                    />
                  </label>
                </div>
              )}
            </div>

            {/* Player Names */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-sm font-medium">
                  <Users className="h-3.5 w-3.5" />
                  Player 1 (Left Side)
                </label>
                <Input
                  value={player1Name}
                  onChange={(e) => setPlayer1Name(e.target.value)}
                  placeholder="e.g. Me, Your name..."
                />
              </div>
              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-sm font-medium">
                  <Users className="h-3.5 w-3.5" />
                  Player 2 (Right Side)
                </label>
                <Input
                  value={player2Name}
                  onChange={(e) => setPlayer2Name(e.target.value)}
                  placeholder="e.g. Opponent..."
                />
              </div>
            </div>

            {processing && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing video... This may take a few minutes.
                </div>
                <Progress value={undefined} className="h-2" />
              </div>
            )}

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <Button
              onClick={handleUpload}
              disabled={!file || uploading || processing || !flaskReady}
              className="w-full"
              size="lg"
            >
              {uploading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Uploading...
                </>
              ) : processing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" />
                  Analyze Video
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Local results (when Supabase is not configured) */}
        {flaskVideoReady && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold tracking-tight">
                Latest Results
              </h2>
            </div>
            <Separator />
            <Card
              className="cursor-pointer transition-colors hover:bg-accent/50"
              onClick={() => navigate(`/analysis/local?p1=${encodeURIComponent(player1Name)}&p2=${encodeURIComponent(player2Name)}&from=dashboard`)}
            >
              <CardContent className="flex items-center justify-between p-4">
                <div className="flex items-center gap-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <Video className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium">Latest Analysis</p>
                    <p className="text-xs text-muted-foreground">
                      Annotated video, heatmaps, coaching report & AI chat ready
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant="default">completed</Badge>
                  <Button variant="outline" size="sm" className="gap-1.5">
                    <Eye className="h-4 w-4" />
                    View Results
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Past Analyses from Supabase */}
        {isSupabaseConfigured && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold tracking-tight">
                Past Analyses
              </h2>
              <Button variant="ghost" size="sm" onClick={loadAnalyses}>
                Refresh
              </Button>
            </div>

            <Separator />

            {analyses.length === 0 ? (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-12">
                  <Video className="h-12 w-12 text-muted-foreground/40 mb-4" />
                  <p className="text-muted-foreground text-sm">
                    No analyses yet. Upload a video to get started!
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-3">
                {analyses.map((a) => (
                  <Card
                    key={a.id}
                    className="cursor-pointer transition-colors hover:bg-accent/50"
                    onClick={() => navigate(`/analysis/${a.id}`)}
                  >
                    <CardContent className="flex items-center justify-between p-4">
                      <div className="flex items-center gap-4">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
                          <Video className="h-5 w-5 text-muted-foreground" />
                        </div>
                        <div>
                          <p className="font-medium">{a.video_name}</p>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <Clock className="h-3 w-3" />
                            {new Date(a.created_at).toLocaleDateString(
                              undefined,
                              {
                                month: "short",
                                day: "numeric",
                                year: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                              }
                            )}
                            {a.total_frames && (
                              <>
                                <span>·</span>
                                <span>{a.total_frames} frames</span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge
                          variant={
                            statusColor[a.status] as
                              | "default"
                              | "secondary"
                              | "destructive"
                              | "outline"
                          }
                        >
                          {a.status}
                        </Badge>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
