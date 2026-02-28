import { useEffect, useRef, useState, useCallback } from "react";
import {
  Send,
  Bot,
  User,
  Loader2,
  Sparkles,
  RotateCcw,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

import { fetchChatHistory, saveChatMessage } from "@/lib/api";
import type { ChatMessage } from "@/types/supabase";

interface ChatPanelProps {
  analysisId: string;
  coachingContext: string;
}

const SUGGESTED_QUESTIONS = [
  "What are the key areas of improvement for Player 1?",
  "Summarize the shot distribution in this match.",
  "How was the T-position discipline for both players?",
  "What patterns did you notice in the ball tracking data?",
  "Give me a training plan based on this analysis.",
];

// For "local" mode, create a NEW chat ID for each analysis run
function getLocalChatId(): string {
  const key = "squash_local_chat_analysis_id";
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
}

// Call this when a new analysis starts to reset chat
export function resetLocalChat(): void {
  sessionStorage.removeItem("squash_local_chat_analysis_id");
}

export function ChatPanel({ analysisId, coachingContext }: ChatPanelProps) {
  // Use a real UUID for Supabase chat persistence, even in local mode
  const chatId = analysisId === "local" ? getLocalChatId() : analysisId;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Helper: save messages to localStorage as fallback
  const localKey = `squash_chat_${chatId}`;
  const saveToLocal = useCallback((msgs: ChatMessage[]) => {
    try {
      localStorage.setItem(localKey, JSON.stringify(msgs));
    } catch { /* quota */ }
  }, [localKey]);

  // Load chat history from Supabase, then fall back to localStorage
  const loadHistory = useCallback(async () => {
    try {
      const history = await fetchChatHistory(chatId);
      if (history.length > 0) {
        setMessages(history);
        return;
      }
    } catch {
      // Supabase not configured
    }
    // Fallback: load from localStorage
    try {
      const stored = localStorage.getItem(localKey);
      if (stored) setMessages(JSON.parse(stored));
    } catch { /* ignore */ }
  }, [chatId, localKey]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  async function sendMessage(text?: string) {
    const userText = (text ?? input).trim();
    if (!userText || loading) return;
    setInput("");
    setLoading(true);

    // Optimistically add user message
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      analysis_id: chatId,
      role: "user",
      content: userText,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => {
      const updated = [...prev, userMsg];
      saveToLocal(updated);
      return updated;
    });

    try {
      // Save user message to Supabase (don't block on failure)
      saveChatMessage(chatId, "user", userText).catch(() => {});

      // Call AI endpoint
      let assistantContent: string;
      try {
        assistantContent = await callAI(userText, messages, coachingContext);
      } catch (aiErr: unknown) {
        console.error("[Chat] AI call failed:", aiErr);
        const errMsg = aiErr instanceof Error ? aiErr.message : String(aiErr);
        if (errMsg.includes("insufficient_quota") || errMsg.includes("429")) {
          assistantContent = `⚠️ **OpenAI Quota Exceeded**\n\nYour OpenAI API key has run out of credits. Consider using **Google Gemini** instead (free!):\n\n1. Get a free key at **https://aistudio.google.com/apikey**\n2. Add it to your .env file as VITE_GEMINI_API_KEY\n3. Restart the dev server`;
        } else if (errMsg.includes("401") || errMsg.includes("invalid_api_key")) {
          assistantContent = `⚠️ **Invalid API Key**\n\nThe API key appears to be invalid. Please double-check it.`;
        } else {
          assistantContent = `⚠️ **AI Error**: ${errMsg}\n\nFalling back to basic analysis:\n\n${generateLocalResponse(userText, coachingContext)}`;
        }
      }

      // Save assistant reply to Supabase (don't block on failure)
      saveChatMessage(chatId, "assistant", assistantContent).catch(() => {});

      const replyMsg: ChatMessage = {
        id: crypto.randomUUID(),
        analysis_id: chatId,
        role: "assistant",
        content: assistantContent,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => {
        const updated = [...prev, replyMsg];
        saveToLocal(updated);
        return updated;
      });
    } catch (err) {
      console.error("Chat error:", err);
      const fallback = generateLocalResponse(userText, coachingContext);
      const fallbackMsg: ChatMessage = {
        id: crypto.randomUUID(),
        analysis_id: chatId,
        role: "assistant",
        content: fallback,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => {
        const updated = [...prev, fallbackMsg];
        saveToLocal(updated);
        return updated;
      });
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <Card className="flex flex-col" style={{ height: "calc(100vh - 220px)" }}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              CourtSense AI Coach
            </CardTitle>
            <CardDescription>
              Ask questions about the match analysis, player performance, and get
              coaching advice.
            </CardDescription>
          </div>
          <Button variant="ghost" size="icon" onClick={loadHistory} title="Refresh">
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <Separator />

      {/* Messages area */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef as React.RefObject<HTMLDivElement>}>
        <div className="space-y-4">
          {messages.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Bot className="h-12 w-12 text-muted-foreground/40 mb-4" />
              <p className="text-sm text-muted-foreground mb-6">
                Ask me anything about the match analysis!
              </p>
              <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                {SUGGESTED_QUESTIONS.map((q) => (
                  <Badge
                    key={q}
                    variant="outline"
                    className="cursor-pointer hover:bg-accent transition-colors py-1.5 px-3"
                    onClick={() => sendMessage(q)}
                  >
                    {q}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <Avatar className="h-8 w-8 shrink-0">
                  <AvatarFallback className="bg-primary/10 text-primary">
                    <Bot className="h-4 w-4" />
                  </AvatarFallback>
                </Avatar>
              )}
              <div
                className={`rounded-lg px-4 py-3 max-w-[80%] text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted"
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.content}</div>
                <p className="text-[10px] opacity-50 mt-1">
                  {new Date(msg.created_at).toLocaleTimeString()}
                </p>
              </div>
              {msg.role === "user" && (
                <Avatar className="h-8 w-8 shrink-0">
                  <AvatarFallback>
                    <User className="h-4 w-4" />
                  </AvatarFallback>
                </Avatar>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex gap-3">
              <Avatar className="h-8 w-8 shrink-0">
                <AvatarFallback className="bg-primary/10 text-primary">
                  <Bot className="h-4 w-4" />
                </AvatarFallback>
              </Avatar>
              <div className="rounded-lg bg-muted px-4 py-3">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      <Separator />

      {/* Input area */}
      <CardContent className="p-4">
        <div className="flex gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the match analysis…"
            className="min-h-[44px] max-h-[120px] resize-none"
            rows={1}
          />
          <Button
            size="icon"
            onClick={() => sendMessage()}
            disabled={!input.trim() || loading}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// ── AI Call — tries Groq (free) → Gemini (free) → OpenAI → Supabase Edge Fn → local fallback ──

const SYSTEM_PROMPT_PREFIX = `You are CourtSense, an expert squash coach AI assistant. You have access to a detailed match analysis report from a computer vision system that tracked ball movement, player positions, shot types, and more.

Here is the analysis data:
---
`;
const SYSTEM_PROMPT_SUFFIX = `
---

Use this data to provide specific, actionable coaching insights. Reference specific numbers and observations from the data. Be encouraging but honest about areas for improvement. Format responses clearly with bullet points when listing multiple items.`;

function buildSystemPrompt(coachingContext: string): string {
  return SYSTEM_PROMPT_PREFIX + coachingContext.slice(0, 8000) + SYSTEM_PROMPT_SUFFIX;
}

async function callAI(
  userMessage: string,
  history: ChatMessage[],
  coachingContext: string
): Promise<string> {
  // 1. Try Groq (free tier — Llama 3.3 70B, 30 RPM)
  const groqKey = import.meta.env.VITE_GROQ_API_KEY as string | undefined;
  if (groqKey) {
    return callGroq(userMessage, history, coachingContext, groqKey);
  }

  // 2. Try Google Gemini (free tier)
  const geminiKey = import.meta.env.VITE_GEMINI_API_KEY as string | undefined;
  if (geminiKey) {
    return callGemini(userMessage, history, coachingContext, geminiKey);
  }

  // 3. Try OpenAI (paid)
  const openaiKey = import.meta.env.VITE_OPENAI_API_KEY as string | undefined;
  if (openaiKey) {
    return callOpenAI(userMessage, history, coachingContext, openaiKey);
  }

  // 3. Try the Supabase Edge Function
  const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string | undefined;
  const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;
  if (supabaseUrl && supabaseKey) {
    const res = await fetch(`${supabaseUrl}/functions/v1/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${supabaseKey}`,
      },
      body: JSON.stringify({
        message: userMessage,
        history: history.map((m) => ({ role: m.role, content: m.content })),
        context: coachingContext,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      return data.reply as string;
    }
  }

  // 4. Fallback to local response
  return generateLocalResponse(userMessage, coachingContext);
}

// ── Groq (free: 30 RPM, Llama 3.3 70B) ──

async function callGroq(
  userMessage: string,
  history: ChatMessage[],
  coachingContext: string,
  apiKey: string
): Promise<string> {
  const systemPrompt = buildSystemPrompt(coachingContext);

  const messages = [
    { role: "system" as const, content: systemPrompt },
    ...history.slice(-10).map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user" as const, content: userMessage },
  ];

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "llama-3.3-70b-versatile",
      messages,
      max_tokens: 1024,
      temperature: 0.7,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    console.error("[Chat] Groq HTTP error:", res.status, err);
    throw new Error(`Groq error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.choices[0].message.content;
}

// ── Google Gemini (free: 15 RPM, 1M tokens/day) ──

async function callGemini(
  userMessage: string,
  history: ChatMessage[],
  coachingContext: string,
  apiKey: string
): Promise<string> {
  const systemPrompt = buildSystemPrompt(coachingContext);

  // Build Gemini contents array (Gemini uses "user"/"model" roles)
  const contents: { role: string; parts: { text: string }[] }[] = [];

  // Add system instruction as first user+model exchange
  contents.push({ role: "user", parts: [{ text: systemPrompt }] });
  contents.push({ role: "model", parts: [{ text: "Understood. I have the match analysis data and I'm ready to provide expert squash coaching insights. What would you like to know?" }] });

  // Add conversation history
  for (const m of history.slice(-10)) {
    contents.push({
      role: m.role === "user" ? "user" : "model",
      parts: [{ text: m.content }],
    });
  }

  // Add current user message
  contents.push({ role: "user", parts: [{ text: userMessage }] });

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 1024,
        },
      }),
    }
  );

  if (!res.ok) {
    const err = await res.text();
    console.error("[Chat] Gemini HTTP error:", res.status, err);
    throw new Error(`Gemini error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text ?? "No response from Gemini.";
}

// ── OpenAI (paid) ──

async function callOpenAI(
  userMessage: string,
  history: ChatMessage[],
  coachingContext: string,
  apiKey: string
): Promise<string> {
  const systemPrompt = buildSystemPrompt(coachingContext);

  const messages = [
    { role: "system" as const, content: systemPrompt },
    ...history.slice(-10).map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user" as const, content: userMessage },
  ];

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      messages,
      max_tokens: 1024,
      temperature: 0.7,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    console.error("[Chat] OpenAI HTTP error:", res.status, err);
    throw new Error(`OpenAI error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.choices[0].message.content;
}

// ── Fallback local response when no AI API is configured ──

function generateLocalResponse(
  question: string,
  context: string
): string {
  const q = question.toLowerCase();

  if (q.includes("improve") || q.includes("better") || q.includes("training")) {
    return `Based on the match analysis, here are some key areas for improvement:

• **T-Position Discipline**: Work on returning to the T after each shot. The data shows inconsistent T-position recovery.
• **Shot Variety**: Consider mixing in more lobs and drops to keep your opponent guessing.
• **Movement Efficiency**: Focus on ghosting drills to improve court coverage and reduce wasted steps.
• **Ball Tracking**: Your confidence tracking suggests some shots may be going to predictable areas.

`;
  }

  if (q.includes("shot") || q.includes("distribution")) {
    return `From the analysis data:

• The match shows a mix of **straight drives** and **crosscourt** patterns.
• Shot classification detected both drives and lobs throughout the match.
• The dominant shot type appears to be straight drives — consider adding more variation.

Check the **Visuals** tab for the shot distribution chart.

`;
  }

  if (q.includes("t-position") || q.includes("t position") || q.includes("movement")) {
    return `Regarding T-position and movement:

• The analysis tracks both players' ankle positions throughout the match.
• Distance from the T is plotted over time — check the Visuals tab for the graph.
• Good T-position discipline means consistently returning to the center of the court.

`;
  }

  if (q.includes("player 1") || q.includes("player 2") || q.includes("player")) {
    return `Player insights from the analysis:

• Both players' movements were tracked via pose estimation (17 keypoints per player).
• Heatmaps show court coverage for each player — check the Visuals tab.
• Ball hit detection identifies which player struck each shot.

The analysis processed ${context.includes("716") ? "716" : "multiple"} frames of match data.

`;
  }

  return `I can help answer questions about the match analysis! Here's what I can see from the data:

• **${context.length > 100 ? "Detailed" : "Basic"} analysis data** is available
• The system tracked ball position, player movements, and shot types
• Heatmaps and visualizations are available in the Visuals tab

Try asking about:
- Player improvement areas
- Shot distribution patterns  
- T-position discipline
- Movement efficiency
- Training recommendations

`;
}

