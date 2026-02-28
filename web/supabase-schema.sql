-- ============================================================
-- Supabase Schema for Squash Analyzer
-- Run this in the Supabase SQL Editor (Dashboard > SQL Editor)
-- ============================================================

-- 1. Analyses table – one row per video upload / analysis run
create table if not exists public.analyses (
  id            uuid primary key default gen_random_uuid(),
  created_at    timestamptz default now(),
  video_name    text not null,
  status        text not null default 'pending'
                  check (status in ('pending','processing','completed','error')),
  total_frames  int,
  coaching_report text,
  match_summary jsonb,
  coaching_data jsonb,
  error_message text,
  video_url     text,
  annotated_video_url text
);

-- 2. Analysis images – heatmaps, court overlays, shot charts, etc.
create table if not exists public.analysis_images (
  id            uuid primary key default gen_random_uuid(),
  analysis_id   uuid references public.analyses(id) on delete cascade,
  image_type    text not null,   -- e.g. 'player_1_heatmap', 'shot_distribution'
  image_url     text not null,
  created_at    timestamptz default now()
);

-- 3. Chat messages – conversation history per analysis
create table if not exists public.chat_messages (
  id            uuid primary key default gen_random_uuid(),
  analysis_id   uuid references public.analyses(id) on delete cascade,
  role          text not null check (role in ('user','assistant')),
  content       text not null,
  created_at    timestamptz default now()
);

-- Indexes for fast lookups
create index if not exists idx_analysis_images_analysis on public.analysis_images(analysis_id);
create index if not exists idx_chat_messages_analysis   on public.chat_messages(analysis_id);
create index if not exists idx_analyses_created         on public.analyses(created_at desc);

-- Row Level Security (open for now – tighten for production)
alter table public.analyses        enable row level security;
alter table public.analysis_images enable row level security;
alter table public.chat_messages   enable row level security;

-- Allow all operations for anon / authenticated (adjust for prod)
create policy "Allow all on analyses"        on public.analyses        for all using (true) with check (true);
create policy "Allow all on analysis_images" on public.analysis_images for all using (true) with check (true);
create policy "Allow all on chat_messages"   on public.chat_messages   for all using (true) with check (true);

-- Storage bucket for videos and output images
-- (Run separately in Supabase Dashboard > Storage > New Bucket)
-- Bucket name: squash-files
-- Public: true

