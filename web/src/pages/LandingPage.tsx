import { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Video, BarChart3, MessageSquare, Target, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const features = [
  {
    icon: Video,
    title: "Video Analysis",
    description:
      "Upload your squash match footage and our computer vision system tracks every movement on the court.",
  },
  {
    icon: null,
    image: "/ball-tracking-icon.png",
    title: "Ball & Shot Tracking",
    description:
      "YOLO-powered ball detection with trajectory prediction, shot classification, and success rate analysis.",
  },
  {
    icon: BarChart3,
    title: "Player Heatmaps & Stats",
    description:
      "See exactly where each player moves, their T-position discipline, and court coverage patterns.",
  },
  {
    icon: MessageSquare,
    title: "AI Coaching",
    description:
      "Chat with an AI coach that has full access to your match data for personalized improvement tips.",
  },
];

const testimonials = [
  { name: "James R.", role: "Club Player", text: "CourtSense completely changed how I review my matches. The heatmaps alone are worth it." },
  { name: "Sarah K.", role: "College Athlete", text: "The AI coaching insights are incredibly specific. It's like having a coach watching every rally." },
  { name: "David M.", role: "Coach", text: "I use this with all my students now. The shot distribution data saves me hours of manual analysis." },
  { name: "Priya T.", role: "Tournament Player", text: "Finally a tool that tracks T-position discipline. My movement has improved so much." },
  { name: "Alex W.", role: "Recreational Player", text: "Super easy to use. Just upload a video and get a full breakdown in minutes." },
  { name: "Omar H.", role: "Junior Coach", text: "The annotated video output is amazing. My players can actually see what they need to fix." },
  { name: "Emily C.", role: "National Team", text: "Professional-level analysis without the professional price tag. Absolute game changer." },
  { name: "Marcus L.", role: "League Captain", text: "We use CourtSense for our entire team's match reviews. The AI coach feature is addictive." },
];

function useScrollReveal() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("revealed");
          }
        });
      },
      { threshold: 0.1 }
    );

    const children = el.querySelectorAll(".scroll-reveal");
    children.forEach((child) => observer.observe(child));

    return () => observer.disconnect();
  }, []);

  return ref;
}

export default function LandingPage() {
  const navigate = useNavigate();
  const scrollRef = useScrollReveal();

  return (
    <div className="min-h-screen bg-gradient-main" ref={scrollRef}>
      {/* Hero */}
      <div className="flex flex-col items-center justify-center px-4 pt-24 pb-20 min-h-screen">
        <img
          src="/courtsense-logo.png"
          alt="CourtSense"
          className="scroll-reveal h-64 w-auto object-contain mb-10"
        />
        <h1
          className="scroll-reveal text-4xl md:text-5xl font-bold tracking-tight text-center max-w-3xl leading-tight"
          style={{ transitionDelay: "150ms" }}
        >
          AI-Powered Squash
          <span className="text-primary"> Match Analysis</span>
        </h1>
        <p
          className="scroll-reveal mt-6 text-lg text-muted-foreground text-center max-w-2xl leading-relaxed"
          style={{ transitionDelay: "300ms" }}
        >
          Upload a match video and get instant insights — player tracking,
          ball trajectory, shot classification, heatmaps, and personalized
          coaching from AI. Everything you need to take your game to the
          next level.
        </p>
        <div className="scroll-reveal" style={{ transitionDelay: "450ms" }}>
          <Button
            onClick={() => navigate("/signin")}
            size="lg"
            className="mt-10 gap-2 text-base h-12 px-8"
          >
            Get Started
            <ArrowRight className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Features */}
      <div className="container pb-20">
        <h2 className="scroll-reveal text-3xl font-bold text-center mb-12">
          How It <span className="text-primary">Works</span>
        </h2>
        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          {features.map((feature, i) => (
            <Card
              key={feature.title}
              className="scroll-reveal border-white/10 bg-black/40 backdrop-blur-sm"
              style={{ transitionDelay: `${i * 100}ms` }}
            >
              <CardContent className="pt-6 pb-6 flex gap-4">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-primary/15 overflow-hidden">
                  {feature.image ? (
                    <img src={feature.image} alt={feature.title} className="h-12 w-12 object-cover rounded-lg" />
                  ) : feature.icon ? (
                    <feature.icon className="h-6 w-6 text-primary" />
                  ) : null}
                </div>
                <div>
                  <h3 className="font-semibold text-lg mb-1">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Testimonials */}
      <div className="scroll-reveal pb-20 overflow-hidden">
        <h2 className="text-3xl font-bold text-center mb-12">
          Loved by <span className="text-primary">Players</span>
        </h2>
        <div className="testimonial-scroll">
          <div className="testimonial-track">
            {[...testimonials, ...testimonials].map((t, i) => (
              <div
                key={i}
                className="testimonial-card flex-shrink-0 w-80 rounded-xl border border-white/10 bg-black/40 backdrop-blur-sm p-6"
              >
                <div className="flex gap-1 mb-3">
                  {[...Array(5)].map((_, j) => (
                    <Star key={j} className="h-4 w-4 fill-primary text-primary" />
                  ))}
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                  "{t.text}"
                </p>
                <div>
                  <p className="font-semibold text-sm">{t.name}</p>
                  <p className="text-xs text-muted-foreground">{t.role}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA */}
      <div className="scroll-reveal container pb-24 text-center">
        <h2 className="text-3xl font-bold mb-4">
          Ready to improve your <span className="text-primary">game</span>?
        </h2>
        <p className="text-muted-foreground mb-8 max-w-lg mx-auto">
          Join players and coaches who are already using CourtSense to unlock
          match insights they never had before.
        </p>
        <Button
          onClick={() => navigate("/signin")}
          size="lg"
          className="gap-2 text-base h-12 px-8"
        >
          Get Started Free
          <ArrowRight className="h-5 w-5" />
        </Button>
        <p className="mt-12 text-xs text-muted-foreground">
          Powered by YOLO, PyTorch & AI
        </p>
      </div>
    </div>
  );
}
