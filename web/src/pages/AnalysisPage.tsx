import { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import {
  ArrowLeft,
  Video,
  BarChart3,
  MessageSquare,
  Image,
  FileText,
  Loader2,
  MapPin,
  Target,
  TrendingUp,
  CheckCircle2,
  AlertTriangle,
  Users,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatPanel } from "@/components/ChatPanel";

import { fetchAnalysis, fetchAnalysisImages, getFlaskText, getFlaskStatus, updateAnalysis } from "@/lib/api";
import { isSupabaseConfigured } from "@/lib/supabase";
import type { Analysis, AnalysisImage } from "@/types/supabase";

// Static output images served by the Flask backend
function getOutputImages(p1: string, p2: string) {
  return [
    { key: "player_1_heatmap", label: `${p1} Heatmap`, icon: MapPin },
    { key: "player_2_heatmap", label: `${p2} Heatmap`, icon: MapPin },
    { key: "shot_distribution", label: "Shot Distribution", icon: Target },
    { key: "shot_success_rate", label: "Shot Success Rate", icon: TrendingUp },
    { key: "t_position_distance", label: "T-Position Distance", icon: BarChart3 },
    { key: "ball_heatmap", label: "Ball Heatmap", icon: Target },
  ];
}

export default function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isLocal = id === "local";

  // Player names from URL params or localStorage
  const player1 = searchParams.get("p1") || localStorage.getItem("squash_player1") || "Player 1";
  const player2 = searchParams.get("p2") || localStorage.getItem("squash_player2") || "Player 2";
  const outputImages = getOutputImages(player1, player2);

  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [images, setImages] = useState<AnalysisImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [coachingText, setCoachingText] = useState<string | null>(null);
  const [aiContext, setAiContext] = useState<string | null>(null);
  const [matchSummary, setMatchSummary] = useState<Record<string, unknown> | null>(null);
  const [availableImages, setAvailableImages] = useState<string[]>([]);

  const load = useCallback(async () => {
    if (!id) return;

    try {
      // Always try to load Flask text data
      try {
        const res = await getFlaskText();
        if (res.ok && res.text) setCoachingText(res.text);
      } catch {
        // Flask might not be running
      }

      // Load match summary from Flask
      try {
        const res = await fetch("/api/match-summary");
        const data = await res.json();
        if (data.ok) setMatchSummary(data.data);
      } catch {
        // Ignore
      }

      // Load structured AI coaching context (real analysis data for chatbot)
      try {
        const res = await fetch("/api/coaching-context");
        const data = await res.json();
        if (data.ok) setAiContext(data.context);
      } catch {
        // Ignore
      }

      // Check which output images actually exist
      try {
        const res = await fetch("/api/available-outputs");
        const data = await res.json();
        if (data.ok) {
          const imgFiles = (data.files as { name: string }[])
            .filter((f) => f.name.endsWith(".png"))
            .map((f) => f.name.replace(".png", ""));
          setAvailableImages(imgFiles);
        }
      } catch {
        // Ignore
      }

      if (isLocal) {
        // Build a local analysis object from Flask data
        setAnalysis({
          id: "local",
          created_at: new Date().toISOString(),
          video_name: "Latest Analysis",
          status: "completed",
          total_frames: matchSummary?.total_frames as number | null ?? null,
          coaching_report: null,
          match_summary: matchSummary,
          coaching_data: null,
          error_message: null,
          video_url: null,
          annotated_video_url: null,
        });
      } else if (isSupabaseConfigured) {
        // Load from Supabase
        const [a, imgs] = await Promise.all([
          fetchAnalysis(id),
          fetchAnalysisImages(id),
        ]);

        // Fix stuck "processing" status: if Supabase says processing but Flask is done
        if (a.status === "processing") {
          try {
            const s = await getFlaskStatus();
            if (!s.running) {
              a.status = s.error ? "error" : "completed";
              // Persist the fix to Supabase
              await updateAnalysis(id, {
                status: a.status,
                ...(s.error ? { error_message: s.error } : {}),
              });
            }
          } catch {
            // Flask not reachable — leave as-is, polling will handle it
          }
        }

        setAnalysis(a);
        setImages(imgs);
      }
    } catch {
      if (isLocal) {
        // Still show local results even if some calls fail
        setAnalysis({
          id: "local",
          created_at: new Date().toISOString(),
          video_name: "Latest Analysis",
          status: "completed",
          total_frames: null,
          coaching_report: null,
          match_summary: null,
          coaching_data: null,
          error_message: null,
          video_url: null,
          annotated_video_url: null,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [id, isLocal]);

  useEffect(() => {
    load();
  }, [load]);

  // Poll Flask status when analysis is still "processing" and auto-update to "completed"
  useEffect(() => {
    if (!analysis || analysis.status !== "processing") return;

    const pollInterval = setInterval(async () => {
      try {
        const s = await getFlaskStatus();
        // Flask says it's no longer running — processing is done
        if (!s.running) {
          const newStatus = s.error ? "error" : "completed";
          // Update local state immediately so the UI refreshes
          setAnalysis((prev) =>
            prev ? { ...prev, status: newStatus, ...(s.error ? { error_message: s.error } : {}) } : prev
          );
          // Also update Supabase so it persists
          if (isSupabaseConfigured && id && id !== "local") {
            try {
              await updateAnalysis(id, {
                status: newStatus,
                ...(s.error ? { error_message: s.error } : {}),
              });
            } catch {
              // Not critical
            }
          }
          // Re-load all data (coaching report, images, etc. are now available)
          load();
          clearInterval(pollInterval);
        }
      } catch {
        // Flask not reachable — ignore
      }
    }, 3000);

    return () => clearInterval(pollInterval);
  }, [analysis?.status, id, load]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4">
        <p className="text-muted-foreground">Analysis not found.</p>
        <Button variant="outline" onClick={() => navigate("/dashboard")}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Button>
      </div>
    );
  }

  // Display text for the coaching report tab
  const reportText = analysis.coaching_report ?? coachingText ?? "No coaching report available. Check the Visuals tab for charts and heatmaps.";

  // AI context — use the structured real data for the chatbot (separate from display)
  const rawAiData = aiContext ?? analysis.coaching_report ?? coachingText ?? "No data available.";
  const coachingContext = `Player names: Player 1 (left side) = "${player1}", Player 2 (right side) = "${player2}".\n\n${rawAiData}`;

  // Resolve image URLs: prefer Supabase images, fall back to Flask /outputs/
  function getImageUrl(key: string): string {
    const sbImage = images.find((i) => i.image_type === key);
    if (sbImage) return sbImage.image_url;
    return `/outputs/${key}.png`;
  }

  // Check if an image key is available
  function isImageAvailable(key: string): boolean {
    if (images.find((i) => i.image_type === key)) return true;
    return availableImages.includes(key);
  }

  const totalFrames =
    analysis.total_frames ??
    (matchSummary?.total_frames as number | undefined) ??
    null;

  const dataPoints =
    (matchSummary?.enhanced_coaching_points as number | undefined) ?? null;

  return (
    <div className="min-h-screen bg-gradient-main">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur">
        <div className="container flex h-14 items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => navigate("/dashboard")}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex-1">
            <h1 className="text-sm font-semibold">{analysis.video_name}</h1>
            <p className="text-xs text-muted-foreground">
              {new Date(analysis.created_at).toLocaleString()}
            </p>
          </div>
          <Badge
            variant={
              analysis.status === "completed"
                ? "default"
                : analysis.status === "error"
                ? "destructive"
                : "secondary"
            }
          >
            {analysis.status}
          </Badge>
        </div>
      </header>

      <main className="container py-6">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview" className="gap-1.5">
              <BarChart3 className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="video" className="gap-1.5">
              <Video className="h-4 w-4" />
              Video
            </TabsTrigger>
            <TabsTrigger value="visuals" className="gap-1.5">
              <Image className="h-4 w-4" />
              Visuals
            </TabsTrigger>
            <TabsTrigger value="chat" className="gap-1.5">
              <MessageSquare className="h-4 w-4" />
              AI Coach
            </TabsTrigger>
          </TabsList>

          {/* -- Overview Tab -- */}
          <TabsContent value="overview" className="space-y-6">
            {/* Player info */}
            <Card>
              <CardContent className="flex items-center gap-6 pt-6">
                <Users className="h-5 w-5 text-muted-foreground" />
                <div className="flex gap-8">
                  <div>
                    <p className="text-xs text-muted-foreground">Player 1 (Left)</p>
                    <p className="font-semibold">{player1}</p>
                  </div>
                  <div className="text-muted-foreground font-light">vs</div>
                  <div>
                    <p className="text-xs text-muted-foreground">Player 2 (Right)</p>
                    <p className="font-semibold">{player2}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <p className="text-2xl font-bold">
                    {totalFrames ?? "--"}
                  </p>
                  <p className="text-xs text-muted-foreground">Total Frames</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-2xl font-bold">
                    {dataPoints ?? "--"}
                  </p>
                  <p className="text-xs text-muted-foreground">Data Points</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-2xl font-bold capitalize">
                    {analysis.status}
                  </p>
                  <p className="text-xs text-muted-foreground">Status</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-2xl font-bold">2</p>
                  <p className="text-xs text-muted-foreground">Players Tracked</p>
                </CardContent>
              </Card>
            </div>

            {/* Data verification */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  <CheckCircle2 className="h-5 w-5 text-primary" />
                  Data Verification
                </CardTitle>
                <CardDescription>
                  What the computer vision system detected and tracked
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      {totalFrames ? (
                        <CheckCircle2 className="h-4 w-4 text-orange-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      )}
                      <span>
                        <strong>Frames processed:</strong> {totalFrames ?? "Unknown"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {availableImages.some((i) => i.includes("heatmap")) ? (
                        <CheckCircle2 className="h-4 w-4 text-orange-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      )}
                      <span>
                        <strong>Player pose estimation:</strong> 17 keypoints per player
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {availableImages.includes("ball_heatmap") ? (
                        <CheckCircle2 className="h-4 w-4 text-orange-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      )}
                      <span>
                        <strong>Ball tracking:</strong> YOLO detection + trajectory prediction
                      </span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      {availableImages.includes("shot_distribution") ? (
                        <CheckCircle2 className="h-4 w-4 text-orange-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      )}
                      <span>
                        <strong>Shot classification:</strong> Type, direction, success rate
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {availableImages.includes("t_position_distance") ? (
                        <CheckCircle2 className="h-4 w-4 text-orange-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      )}
                      <span>
                        <strong>T-position analysis:</strong> Distance from T over time
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-orange-500" />
                      <span>
                        <strong>Player labels:</strong> {player1} = Player 1 (left), {player2} = Player 2 (right)
                      </span>
                    </div>
                  </div>
                </div>
                <div className="mt-4 space-y-1 text-xs text-muted-foreground">
                  <p>
                    <strong>How to verify:</strong> Watch the annotated video and compare with the heatmaps. Player 1 is detected on the left side of the court, Player 2 on the right.
                  </p>
                  <p>
                    <strong>Note on shot success rate:</strong> The current model classifies shot outcomes based on ball trajectory continuation. A "100%" rate means the model detected all shots as in-play (not errored). This does not mean every shot was a "winner" — it means the ball stayed in play after each detected shot.
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Coaching report */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Coaching Report
                </CardTitle>
                <CardDescription>
                  AI-generated coaching insights from the analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px] rounded-md border p-4">
                  <pre className="whitespace-pre-wrap text-sm leading-relaxed font-mono">
                    {reportText}
                  </pre>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>

          {/* -- Video Tab -- */}
          <TabsContent value="video">
            <Card>
              <CardHeader>
                <CardTitle>Annotated Video</CardTitle>
                <CardDescription>
                  Video with ball tracking, player pose estimation, and shot
                  classifications overlaid.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {analysis.annotated_video_url ? (
                  <video
                    controls
                    preload="metadata"
                    className="w-full rounded-lg"
                    src={analysis.annotated_video_url}
                  />
                ) : (
                  <video
                    controls
                    preload="metadata"
                    playsInline
                    className="w-full rounded-lg"
                    src="/outputs/annotated.mp4"
                  />
                )}
                <div className="mt-3 flex gap-2">
                  <Button variant="outline" size="sm" asChild>
                    <a href="/outputs/annotated.mp4" download>
                      Download MP4
                    </a>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* -- Visuals Tab -- */}
          <TabsContent value="visuals" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {outputImages.filter(({ key }) => isImageAvailable(key)).map(
                ({ key, label, icon: Icon }) => (
                  <Card key={key}>
                    <CardHeader className="pb-3">
                      <CardTitle className="flex items-center gap-2 text-base">
                        <Icon className="h-4 w-4" />
                        {label}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <img
                        src={getImageUrl(key)}
                        alt={label}
                        className="w-full rounded-md border"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = "none";
                        }}
                      />
                    </CardContent>
                  </Card>
                )
              )}
              {availableImages.length === 0 && images.length === 0 && (
                <Card className="md:col-span-2">
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <Image className="h-12 w-12 text-muted-foreground/40 mb-4" />
                    <p className="text-muted-foreground text-sm">
                      No visualizations available yet.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* -- Chat Tab -- */}
          <TabsContent value="chat" className="space-y-4">
            {aiContext && (
              <Card>
                <CardContent className="flex items-center gap-3 py-3 px-4">
                  <CheckCircle2 className="h-4 w-4 text-orange-500 shrink-0" />
                  <p className="text-xs text-muted-foreground">
                    <strong>AI is using real match data:</strong> match report, frame-by-frame analysis ({totalFrames ?? "?"} frames),
                    shot distribution, player positions, and T-position metrics.
                  </p>
                </CardContent>
              </Card>
            )}
            <ChatPanel analysisId={id!} coachingContext={coachingContext} />
          </TabsContent>
        </Tabs>
      </main>

      <Separator className="mt-8" />
      <footer className="container py-4 text-center text-xs text-muted-foreground">
        CourtSense - Powered by YOLO, PyTorch & AI
      </footer>
    </div>
  );
}
