#!/usr/bin/env bun
import { randomBytes, timingSafeEqual } from "node:crypto";
import { mkdir, readdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
/**
 * omp agent bridge.
 *
 * Serves the subset of the Cursor Cloud Agents v1 wire format that Rakazo's
 * cloud-agent adapter speaks (`CLOUD_AGENT_PROVIDER=omp`), backed by headless
 * `omp -p` sessions on this host. One agent = one working directory and one omp
 * session; every follow-up is a new run that resumes that session.
 *
 * Statuses follow the Cursor wire: runs are CREATING | RUNNING | FINISHED |
 * ERROR | CANCELLED, agents ACTIVE | ARCHIVED. A run still active when the
 * bridge restarts lost its process and reads as ERROR.
 */
import type { Subprocess } from "bun";

const TOKEN = process.env.OMP_BRIDGE_TOKEN?.trim() ?? "";
if (TOKEN.length < 32) throw new Error("OMP_BRIDGE_TOKEN must be at least 32 characters");
const HOST = process.env.OMP_BRIDGE_HOST ?? "127.0.0.1";
const PORT = Number(process.env.OMP_BRIDGE_PORT ?? 7480);
const DATA = process.env.OMP_BRIDGE_DATA ?? join(homedir(), ".rakazo-omp");
const PUBLIC_URL = (process.env.OMP_BRIDGE_PUBLIC_URL ?? `http://${HOST}:${PORT}`).replace(
  /\/+$/,
  "",
);
const OMP = process.env.OMP_BIN ?? "omp";
const MODEL = process.env.OMP_BRIDGE_MODEL?.trim();
const MAX_TIME = process.env.OMP_BRIDGE_MAX_TIME ?? "2h";
const MAX_RUNNING = Number(process.env.OMP_BRIDGE_MAX_RUNNING ?? 4);
const AGENTS_DIR = join(DATA, "agents");

type RunStatus = "CREATING" | "RUNNING" | "FINISHED" | "ERROR" | "CANCELLED";
interface Run {
  id: string;
  status: RunStatus;
  prompt: string;
  createdAt: string;
  finishedAt?: string;
  error?: string;
  branch?: string;
  prUrl?: string;
  /** Last assistant text of the run, for the agent page. */
  summary?: string;
}
interface Agent {
  id: string;
  name: string;
  status: "ACTIVE" | "ARCHIVED";
  repository?: string;
  autoCreatePR: boolean;
  createdAt: string;
  sessionFile?: string;
  /** Oldest first. */
  runs: Run[];
}
type Image = { data: string; mimeType: string } | { url: string };

const agents = new Map<string, Agent>();
const live = new Map<string, { proc?: Subprocess; cancelled: boolean }>();
const ACTIVE = new Set<RunStatus>(["CREATING", "RUNNING"]);
const ID = /^[A-Za-z0-9_-]{1,100}$/;
const EXT: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
};

const agentDir = (id: string) => join(AGENTS_DIR, id);
const now = () => new Date().toISOString();
const latest = (agent: Agent) => agent.runs[agent.runs.length - 1]!;
const runningCount = () =>
  [...agents.values()].reduce((n, a) => n + a.runs.filter((r) => ACTIVE.has(r.status)).length, 0);

async function save(agent: Agent) {
  const file = join(agentDir(agent.id), "agent.json");
  await writeFile(`${file}.tmp`, JSON.stringify(agent, null, 2));
  await rename(`${file}.tmp`, file);
}

async function load() {
  await mkdir(AGENTS_DIR, { recursive: true });
  for (const id of await readdir(AGENTS_DIR)) {
    const raw = await readFile(join(agentDir(id), "agent.json"), "utf8").catch(() => null);
    if (!raw) continue;
    const agent = JSON.parse(raw) as Agent;
    let lost = false;
    for (const run of agent.runs) {
      if (!ACTIVE.has(run.status)) continue;
      Object.assign(run, { status: "ERROR", error: "bridge restarted mid-run", finishedAt: now() });
      lost = true;
    }
    agents.set(agent.id, agent);
    if (lost) await save(agent);
  }
}

function wireAgent(agent: Agent) {
  return {
    id: agent.id,
    name: agent.name,
    status: agent.status,
    url: `${PUBLIC_URL}/agents/${agent.id}`,
    latestRunId: latest(agent).id,
  };
}

function wireRun(run: Run) {
  return {
    id: run.id,
    status: run.status,
    ...(run.branch
      ? { git: { branches: [{ branch: run.branch, ...(run.prUrl ? { prUrl: run.prUrl } : {}) }] } }
      : {}),
  };
}

async function exec(cmd: string[], cwd: string, runId?: string) {
  const proc = Bun.spawn(cmd, { cwd, stdout: "pipe", stderr: "pipe" });
  const handle = runId ? live.get(runId) : undefined;
  if (handle) handle.proc = proc;
  const [out, err, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { out: out.trim(), err: err.trim(), code };
}

function promptFor(agent: Agent, run: Run, first: boolean, imageUrls: string[]) {
  const parts = [run.prompt];
  if (imageUrls.length)
    parts.push(`Reference images:\n${imageUrls.map((u) => `- ${u}`).join("\n")}`);
  if (agent.repository && first) {
    const branch = `omp/${agent.id.replace(/^bc-/, "").slice(0, 8)}`;
    parts.push(
      agent.autoCreatePR
        ? `When the work is done: commit it on a new branch named \`${branch}\`, push that branch to origin, and open a pull request with \`gh pr create\`. Print the pull request URL.`
        : `Commit your work on a new local branch named \`${branch}\`. Do not push and do not open a pull request.`,
    );
  }
  return parts.join("\n\n");
}

/** Last assistant text and session id from an `omp --mode json` event log. */
function readLog(log: string) {
  let sessionId: string | undefined;
  let summary: string | undefined;
  for (const line of log.split("\n")) {
    if (!line.startsWith("{")) continue;
    let event: { type?: string; id?: string; message?: { role?: string; content?: unknown } };
    try {
      event = JSON.parse(line);
    } catch {
      continue;
    }
    if (event.type === "session" && event.id) sessionId = event.id;
    if (event.type === "message_end" && event.message?.role === "assistant") {
      const text = (Array.isArray(event.message.content) ? event.message.content : [])
        .filter((c: { type?: string }) => c.type === "text")
        .map((c: { text?: string }) => c.text ?? "")
        .join("")
        .trim();
      if (text) summary = text;
    }
  }
  return { sessionId, summary };
}

async function execute(agent: Agent, run: Run, images: Image[]) {
  const dir = agentDir(agent.id);
  const work = join(dir, "work");
  const runsDir = join(dir, "runs");
  const first = agent.runs.length === 1;
  const handle = live.get(run.id)!;
  try {
    await mkdir(runsDir, { recursive: true });
    if (first && agent.repository) {
      const clone = await exec(["git", "clone", agent.repository, work], dir, run.id);
      if (handle.cancelled) return finish(agent, run, "CANCELLED");
      if (clone.code !== 0)
        return finish(agent, run, "ERROR", `git clone failed: ${clone.err.slice(-500)}`);
    } else {
      await mkdir(work, { recursive: true });
    }
    const baseBranch = agent.repository
      ? (await exec(["git", "rev-parse", "--abbrev-ref", "HEAD"], work)).out
      : undefined;

    const files: string[] = [];
    const urls: string[] = [];
    for (const [i, image] of images.entries()) {
      if ("url" in image) urls.push(image.url);
      else {
        const file = join(runsDir, `${run.id}-${i}.${EXT[image.mimeType] ?? "png"}`);
        await writeFile(file, Buffer.from(image.data, "base64"));
        files.push(file);
      }
    }
    const promptFile = join(runsDir, `${run.id}.prompt.md`);
    await writeFile(promptFile, promptFor(agent, run, first, urls));

    run.status = "RUNNING";
    await save(agent);
    if (handle.cancelled) return finish(agent, run, "CANCELLED");

    const args = [
      OMP,
      "-p",
      "--mode",
      "json",
      "--auto-approve",
      "--session-dir",
      join(dir, "sessions"),
      "--cwd",
      work,
      "--max-time",
      MAX_TIME,
      ...(MODEL ? ["--model", MODEL] : []),
      ...(agent.sessionFile ? ["--resume", agent.sessionFile] : []),
      `@${promptFile}`,
      ...files.map((f) => `@${f}`),
    ];
    const proc = Bun.spawn(args, {
      cwd: work,
      stdin: "ignore",
      stdout: Bun.file(join(runsDir, `${run.id}.jsonl`)),
      stderr: Bun.file(join(runsDir, `${run.id}.stderr.log`)),
    });
    handle.proc = proc;
    const code = await proc.exited;

    const { sessionId, summary } = readLog(
      await readFile(join(runsDir, `${run.id}.jsonl`), "utf8").catch(() => ""),
    );
    if (summary) run.summary = summary;
    if (sessionId && !agent.sessionFile) {
      const sessions = await readdir(join(dir, "sessions")).catch(() => [] as string[]);
      const file = sessions.find((name) => name.includes(sessionId));
      if (file) agent.sessionFile = join(dir, "sessions", file);
    }
    if (agent.repository) {
      const branch = (await exec(["git", "rev-parse", "--abbrev-ref", "HEAD"], work)).out;
      if (branch && branch !== baseBranch && branch !== "HEAD") {
        run.branch = branch;
        if (agent.autoCreatePR) {
          const pr = await exec(["gh", "pr", "view", branch, "--json", "url", "-q", ".url"], work);
          if (pr.code === 0 && pr.out.startsWith("https://")) run.prUrl = pr.out;
        }
      }
    }
    if (handle.cancelled) return finish(agent, run, "CANCELLED");
    if (code === 0) return finish(agent, run, "FINISHED");
    const stderr = await readFile(join(runsDir, `${run.id}.stderr.log`), "utf8").catch(() => "");
    return finish(agent, run, "ERROR", `omp exited ${code}: ${stderr.trim().slice(-500)}`);
  } catch (error) {
    return finish(agent, run, "ERROR", error instanceof Error ? error.message : String(error));
  }
}

async function finish(agent: Agent, run: Run, status: RunStatus, error?: string) {
  Object.assign(run, { status, finishedAt: now(), ...(error ? { error } : {}) });
  live.delete(run.id);
  await save(agent);
}

function startRun(agent: Agent, prompt: string, images: Image[]) {
  const run: Run = {
    id: `run-${Date.now().toString(36)}-${randomBytes(4).toString("hex")}`,
    status: "CREATING",
    prompt,
    createdAt: now(),
  };
  agent.runs.push(run);
  live.set(run.id, { cancelled: false });
  void save(agent).then(() => execute(agent, run, images));
  return run;
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

function parseImage(value: unknown): Image | null {
  if (!isObject(value)) return null;
  if (typeof value.url === "string" && value.url.startsWith("https://")) return { url: value.url };
  if (typeof value.data === "string" && typeof value.mimeType === "string" && value.mimeType in EXT)
    return { data: value.data, mimeType: value.mimeType };
  return null;
}

function parsePrompt(body: unknown): { text: string; images: Image[] } | null {
  if (!isObject(body) || !isObject(body.prompt)) return null;
  const { text, images: raw = [] } = body.prompt;
  if (typeof text !== "string" || !text.trim() || !Array.isArray(raw) || raw.length > 5)
    return null;
  const images = raw.map(parseImage);
  return images.every((image): image is Image => image !== null) ? { text, images } : null;
}

function authorized(request: Request) {
  const given = Buffer.from(request.headers.get("authorization") ?? "");
  const expected = Buffer.from(`Bearer ${TOKEN}`);
  return given.length === expected.length && timingSafeEqual(given, expected);
}

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  Response.json(body, { status, headers });
const error = (status: number, message: string) => json({ code: status, message }, status);
const busy = () =>
  json({ code: 429, message: "omp bridge is at its run limit" }, 429, { "Retry-After": "30" });

async function api(request: Request, path: string): Promise<Response> {
  if (!authorized(request)) return error(401, "Unauthorized");
  const method = request.method;
  if (path === "/v1/agents" && method === "POST") {
    const body: unknown = await request.json().catch(() => null);
    const prompt = parsePrompt(body);
    if (!isObject(body) || typeof body.agentId !== "string" || !ID.test(body.agentId) || !prompt)
      return error(400, "Invalid agent request");
    const id = body.agentId;
    if (agents.has(id)) return error(409, "Agent already exists");
    if (runningCount() >= MAX_RUNNING) return busy();
    const repo = Array.isArray(body.repos) ? body.repos[0] : undefined;
    const repository = isObject(repo) && typeof repo.url === "string" ? repo.url : undefined;
    if (repository && !repository.startsWith("https://"))
      return error(400, "Repository must be https");
    const agent: Agent = {
      id,
      name: prompt.text.trim().split("\n")[0]!.slice(0, 80),
      status: "ACTIVE",
      ...(repository ? { repository } : {}),
      autoCreatePR: body.autoCreatePR === true,
      createdAt: now(),
      runs: [],
    };
    await mkdir(agentDir(id), { recursive: true });
    agents.set(id, agent);
    const run = startRun(agent, prompt.text, prompt.images);
    return json({ agent: wireAgent(agent), run: wireRun(run) });
  }
  const match = /^\/v1\/agents\/([^/]+)(?:\/(runs)(?:\/([^/]+)(\/cancel)?)?)?$/.exec(path);
  const agent = match ? agents.get(decodeURIComponent(match[1]!)) : undefined;
  if (!match || !agent) return error(404, "Not found");
  const [, , runs, runId, cancel] = match;
  if (!runs && method === "GET") return json(wireAgent(agent));
  if (runs && !runId && method === "GET") {
    const limit = Math.max(1, Number(new URL(request.url).searchParams.get("limit") ?? 20) || 20);
    return json({ items: [...agent.runs].reverse().slice(0, limit).map(wireRun) });
  }
  if (runs && !runId && method === "POST") {
    if (agent.status === "ARCHIVED") return error(409, "Agent is archived");
    if (ACTIVE.has(latest(agent).status)) return error(409, "Agent is busy with another run");
    const prompt = parsePrompt(await request.json().catch(() => null));
    if (!prompt) return error(400, "Invalid follow-up");
    if (runningCount() >= MAX_RUNNING) return busy();
    return json({ run: wireRun(startRun(agent, prompt.text, prompt.images)) });
  }
  const run = runId ? agent.runs.find((r) => r.id === decodeURIComponent(runId)) : undefined;
  if (!run) return error(404, "Not found");
  if (!cancel && method === "GET") return json(wireRun(run));
  if (cancel && method === "POST") {
    const handle = live.get(run.id);
    if (!handle || !ACTIVE.has(run.status)) return error(409, "Run already ended");
    handle.cancelled = true;
    handle.proc?.kill("SIGTERM");
    setTimeout(() => live.get(run.id)?.proc?.kill("SIGKILL"), 10_000).unref();
    return json({});
  }
  return error(405, "Method not allowed");
}

const escapeHtml = (s: string) =>
  s.replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!,
  );

function page(agent: Agent) {
  const running = ACTIVE.has(latest(agent).status);
  const runs = agent.runs
    .map(
      (r) => `<section><h3>${escapeHtml(r.status)} <small>${escapeHtml(r.createdAt)}</small></h3>
<pre class="prompt">${escapeHtml(r.prompt)}</pre>
${r.branch ? `<p>Branch: <code>${escapeHtml(r.branch)}</code>${r.prUrl ? ` · <a href="${escapeHtml(r.prUrl)}">${escapeHtml(r.prUrl)}</a>` : ""}</p>` : ""}
${r.error ? `<pre class="error">${escapeHtml(r.error)}</pre>` : ""}
${r.summary ? `<pre>${escapeHtml(r.summary)}</pre>` : ""}</section>`,
    )
    .join("\n");
  return `<!doctype html><html><head><meta charset="utf-8"><title>${escapeHtml(agent.name)} · omp</title>
${running ? '<meta http-equiv="refresh" content="5">' : ""}
<style>body{font:14px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}pre{white-space:pre-wrap;background:#f4f4f5;padding:.75rem;border-radius:6px}.prompt{background:#eef2ff}.error{background:#fef2f2}small{color:#71717a;font-weight:normal}</style>
</head><body><h1>${escapeHtml(agent.name)}</h1>
<p>omp agent <code>${escapeHtml(agent.id)}</code>${agent.repository ? ` on <a href="${escapeHtml(agent.repository)}">${escapeHtml(agent.repository)}</a>` : ""}</p>
${runs}</body></html>`;
}

await load();
Bun.serve({
  hostname: HOST,
  port: PORT,
  maxRequestBodySize: 128 * 1024 * 1024,
  async fetch(request) {
    const path = new URL(request.url).pathname;
    if (path === "/healthz") return new Response("ok");
    if (path.startsWith("/v1/")) return api(request, path);
    const match = /^\/agents\/([^/]+)$/.exec(path);
    const agent = match ? agents.get(decodeURIComponent(match[1]!)) : undefined;
    if (agent && request.method === "GET")
      return new Response(page(agent), { headers: { "Content-Type": "text/html; charset=utf-8" } });
    return new Response("Not found", { status: 404 });
  },
});
console.log(`omp agent bridge on http://${HOST}:${PORT} (${agents.size} agents, data ${DATA})`);
