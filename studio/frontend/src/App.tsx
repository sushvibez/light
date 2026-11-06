import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Events } from "@wailsio/runtime";
import { StudioService } from "../bindings/light/studio";
import type { Entry, GenOpts, Message, Spec, Stats, Status } from "../bindings/light/engine/models";

type ChatMsg = Message & { stats?: Stats; pending?: boolean };
type Session = { id: string; title: string; messages: ChatMsg[]; created: number };

const SYSTEM_DEFAULT = "You are a helpful assistant.";

function fmtGB(b: number) {
  return (b / 2 ** 30).toFixed(b < 2 ** 30 ? 2 : 1) + " GB";
}

function newSession(): Session {
  return { id: crypto.randomUUID(), title: "New chat", messages: [], created: Date.now() };
}

function loadSessions(): Session[] {
  try {
    const raw = localStorage.getItem("light.sessions");
    if (raw) return JSON.parse(raw);
  } catch {}
  return [newSession()];
}

export default function App() {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [status, setStatus] = useState<Status | null>(null);
  const [apiAddr, setApiAddr] = useState("");
  const [sessions, setSessions] = useState<Session[]>(loadSessions);
  const [active, setActive] = useState<string>(() => loadSessions()[0]?.id);
  const [input, setInput] = useState("");
  const [system, setSystem] = useState(SYSTEM_DEFAULT);
  const [opts, setOpts] = useState<GenOpts>({ max_tokens: 512, temperature: 0.7, top_p: 0.9, top_k: 0, seed: 0 });
  const [device, setDevice] = useState<"gpu" | "cpu">("gpu");
  const [dtype, setDtype] = useState<"bf16" | "f32">("bf16");
  const [error, setError] = useState("");
  const [busyLoad, setBusyLoad] = useState("");
  const [showSettings, setShowSettings] = useState(true);
  const requestRef = useRef<number>(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  const session = useMemo(() => sessions.find((s) => s.id === active) ?? sessions[0], [sessions, active]);
  const generating = session?.messages.some((m) => m.pending) ?? false;

  useEffect(() => {
    try {
      localStorage.setItem("light.sessions", JSON.stringify(sessions));
    } catch {}
  }, [sessions]);

  const refreshStatus = useCallback(() => {
    StudioService.Status().then(setStatus).catch(() => {});
  }, []);

  useEffect(() => {
    StudioService.Models().then((m) => setEntries(m ?? [])).catch((e) => setError(String(e)));
    StudioService.APIAddr().then(setApiAddr).catch(() => {});
    refreshStatus();
    const t = setInterval(refreshStatus, 3000);
    return () => clearInterval(t);
  }, [refreshStatus]);

  // Streaming events from Go.
  useEffect(() => {
    const offTok = Events.On("token", (ev: { data: { request: number; text: string } }) => {
      const { request, text } = ev.data;
      if (request !== requestRef.current) return;
      setSessions((ss) =>
        ss.map((s) =>
          s.id !== active
            ? s
            : {
                ...s,
                messages: s.messages.map((m, i) =>
                  i === s.messages.length - 1 && m.pending ? { ...m, content: m.content + text } : m,
                ),
              },
        ),
      );
    });
    const offDone = Events.On("done", (ev: { data: { request: number; error?: string; stats: Stats } }) => {
      const { request, error: err, stats } = ev.data;
      if (request !== requestRef.current) return;
      if (err) setError(err);
      setSessions((ss) =>
        ss.map((s) =>
          s.id !== active
            ? s
            : {
                ...s,
                messages: s.messages.map((m, i) =>
                  i === s.messages.length - 1 && m.pending ? { ...m, pending: false, stats } : m,
                ),
              },
        ),
      );
      refreshStatus();
    });
    return () => {
      offTok();
      offDone();
    };
  }, [active, refreshStatus]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [session?.messages]);

  async function loadModel(e: Entry) {
    setError("");
    setBusyLoad(e.spec.id);
    const spec: Spec = { ...e.spec, device, dtype: e.kind === "gguf" ? "" : dtype };
    try {
      setStatus(await StudioService.Load(spec));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusyLoad("");
      refreshStatus();
    }
  }

  async function loadSpeech(e: Entry) {
    setError("");
    setBusyLoad(e.spec.id);
    try {
      setStatus(await StudioService.LoadSpeech({ ...e.spec, device }));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusyLoad("");
      refreshStatus();
    }
  }

  async function loadSTT(e: Entry) {
    setError("");
    setBusyLoad(e.spec.id);
    try {
      setStatus(await StudioService.LoadSTT({ ...e.spec, device }));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusyLoad("");
      refreshStatus();
    }
  }

  const [transcribing, setTranscribing] = useState(false);
  async function transcribe() {
    setError("");
    setTranscribing(true);
    try {
      const res = await StudioService.TranscribeFile("");
      if (res?.text) {
        setInput((v) => (v ? v + " " : "") + res.text);
        setLastSTT(`${res.audio_sec.toFixed(1)} s audio in ${((res.mel_ms + res.encode_ms + res.decode_ms) / 1000).toFixed(2)} s (${res.real_time_x.toFixed(1)}× real time)`);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setTranscribing(false);
      refreshStatus();
    }
  }
  const [lastSTT, setLastSTT] = useState("");

  const [conversing, setConversing] = useState(false);
  const [lastTalk, setLastTalk] = useState("");
  async function converse() {
    setError("");
    setConversing(true);
    try {
      const res = await StudioService.ConverseFile(2);
      if (res?.reply_path) {
        setLastTalk(`"${res.text}" · ${res.audio_sec.toFixed(1)} s reply in ${res.frames} frames (${res.step_ms.toFixed(0)} ms/frame, ${res.real_time_x.toFixed(1)}× real time) → ${res.reply_path}`);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setConversing(false);
      refreshStatus();
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || generating || !status?.loaded || !session) return;
    setError("");
    setInput("");
    const userMsg: ChatMsg = { role: "user", content: text };
    const history: Message[] = [
      ...(system.trim() ? [{ role: "system", content: system.trim() } as Message] : []),
      ...session.messages.filter((m) => !m.pending).map(({ role, content }) => ({ role, content })),
      userMsg,
    ];
    setSessions((ss) =>
      ss.map((s) =>
        s.id !== session.id
          ? s
          : {
              ...s,
              title: s.messages.length === 0 ? text.slice(0, 40) : s.title,
              messages: [...s.messages, userMsg, { role: "assistant", content: "", pending: true }],
            },
      ),
    );
    try {
      requestRef.current = await StudioService.Chat(history, opts);
    } catch (err) {
      setError(String(err));
      setSessions((ss) =>
        ss.map((s) => (s.id !== session.id ? s : { ...s, messages: s.messages.filter((m) => !m.pending) })),
      );
    }
  }

  function stop() {
    StudioService.Stop();
  }

  function addSession() {
    const s = newSession();
    setSessions((ss) => [s, ...ss]);
    setActive(s.id);
  }

  function removeSession(id: string) {
    setSessions((ss) => {
      const rest = ss.filter((s) => s.id !== id);
      const next = rest.length ? rest : [newSession()];
      if (id === active) setActive(next[0].id);
      return next;
    });
  }

  const loadedId = status?.spec?.id;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">◐</span> Light Studio
        </div>
        <button className="primary wide" onClick={addSession}>
          + New chat
        </button>
        <div className="sessions">
          {sessions.map((s) => (
            <div key={s.id} className={"session" + (s.id === session?.id ? " active" : "")} onClick={() => setActive(s.id)}>
              <span className="title">{s.title}</span>
              <button
                className="icon"
                title="Delete"
                onClick={(ev) => {
                  ev.stopPropagation();
                  removeSession(s.id);
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
        <div className="models">
          <div className="section-title">Models on disk</div>
          {entries.length === 0 && <div className="muted small">No supported checkpoints found in the HF cache or ~/.cache/light/gguf.</div>}
          {entries.map((e) => {
            const isLoaded =
              e.kind === "stt"
                ? e.spec.id === status?.stt?.id
                : e.kind === "embedding"
                  ? e.spec.id === status?.embedder?.id
                  : e.kind === "speech"
                    ? e.spec.id === status?.speech?.id
                    : e.spec.id === loadedId;
            return (
              <div key={e.spec.id} className={"model" + (isLoaded ? " loaded" : "")}>
                <div className="model-main">
                  <div className="model-name" title={e.spec.id}>
                    {e.spec.id}
                  </div>
                  <div className="model-meta">
                    <span className={"tag " + e.kind}>{e.kind === "gguf" ? "GGUF" : e.kind === "stt" ? "speech→text" : e.kind === "embedding" ? "embeddings" : e.kind === "speech" ? "speech↔speech" : "safetensors"}</span>
                    {e.params && <span className="tag">{e.params}</span>}
                    <span className="tag">{e.model_type}</span>
                    <span className="muted">{fmtGB(e.size_bytes)}</span>
                  </div>
                </div>
                <button
                  className={isLoaded ? "ghost" : "secondary"}
                  disabled={!!busyLoad || generating}
                  onClick={() =>
                    e.kind === "stt"
                      ? isLoaded
                        ? StudioService.UnloadSTT().then(refreshStatus)
                        : loadSTT(e)
                      : e.kind === "embedding"
                        ? isLoaded
                          ? StudioService.UnloadEmbedder().then(refreshStatus)
                          : StudioService.LoadEmbedder({ ...e.spec, device }).then(setStatus)
                        : e.kind === "speech"
                          ? isLoaded
                            ? StudioService.UnloadSpeech().then(refreshStatus)
                            : loadSpeech(e)
                        : isLoaded
                        ? StudioService.Unload().then(refreshStatus)
                        : loadModel(e)
                  }
                >
                  {busyLoad === e.spec.id ? "Loading…" : isLoaded ? "Unload" : "Load"}
                </button>
              </div>
            );
          })}
        </div>
        <div className="footer">
          <div className="small muted">
            {status?.loaded ? (
              <>
                <b>{status.spec?.id}</b>
                <br />
                {status.params_m.toFixed(0)}M params · ctx {status.context}
                {status.sliding_window ? ` · window ${status.sliding_window}` : ""} · {status.spec?.family} template
                <br />
              </>
            ) : status?.loading ? (
              <>Loading model…</>
            ) : (
              <>No model loaded</>
            )}
            {status?.stt ? (
              <>
                STT {status.stt.id}
                {lastSTT ? ` · ${lastSTT}` : ""}
                <br />
              </>
            ) : null}
            {status?.speech ? (
              <>
                Moshi {status.speech.id}
                {lastTalk ? ` · ${lastTalk}` : ""}
                <br />
              </>
            ) : null}
            {status?.gpu_total_mb ? (
              <>
                GPU {status.gpu_used_mb} / {status.gpu_total_mb} MB
                <br />
              </>
            ) : null}
            API {apiAddr}
          </div>
        </div>
      </aside>

      <main className="chat">
        <header className="topbar">
          <div className="topbar-title">{session?.title}</div>
          <div className="spacer" />
          <button className="ghost" onClick={() => setShowSettings((v) => !v)}>
            {showSettings ? "Hide settings" : "Settings"}
          </button>
        </header>
        {error && (
          <div className="error" onClick={() => setError("")}>
            {error}
          </div>
        )}
        <div className="messages" ref={scrollRef}>
          {session?.messages.length === 0 && (
            <div className="empty">
              <div className="empty-title">Runs on your GPU, in-process.</div>
              <div className="muted">
                Load a model on the left, then ask anything. Every reply is generated by Light's own Go kernels.
                <br />
                Other apps can use the same model through the OpenAI-compatible API at <code>{apiAddr}</code>.
              </div>
            </div>
          )}
          {session?.messages.map((m, i) => (
            <div key={i} className={"msg " + m.role}>
              <div className="role">{m.role === "user" ? "You" : loadedId ?? "Assistant"}</div>
              <div className="bubble">
                {m.content}
                {m.pending && <span className="cursor">▍</span>}
              </div>
              {m.stats && (
                <div className="stats muted small">
                  {m.stats.tokens} tok · {m.stats.tok_per_sec.toFixed(1)} tok/s · prefill {m.stats.prompt_tokens} tok in{" "}
                  {(m.stats.prefill_ms / 1000).toFixed(2)} s{m.stats.stop !== "eos" ? ` · stopped: ${m.stats.stop}` : ""}
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="composer">
          <textarea
            value={input}
            placeholder={status?.loaded ? "Message… (Enter to send, Shift+Enter for newline)" : "Load a model first"}
            disabled={!status?.loaded}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={2}
          />
          <button className="ghost" title="Transcribe a WAV file with Whisper into the message box" disabled={transcribing || generating} onClick={transcribe}>
            {transcribing ? "Transcribing…" : "🎤 Audio"}
          </button>
          <button className="ghost" title="Send a WAV of your turn to Moshi; the spoken reply is saved next to it" disabled={conversing || generating || !!busyLoad} onClick={converse}>
            {conversing ? "Moshi is talking…" : "🗣 Talk"}
          </button>
          {generating ? (
            <button className="danger" onClick={stop}>
              Stop
            </button>
          ) : (
            <button className="primary" onClick={send} disabled={!status?.loaded || !input.trim()}>
              Send
            </button>
          )}
        </div>
      </main>

      {showSettings && (
        <aside className="settings">
          <div className="section-title">Generation</div>
          <label>
            Max tokens <span>{opts.max_tokens}</span>
            <input type="range" min={16} max={4096} step={16} value={opts.max_tokens} onChange={(e) => setOpts({ ...opts, max_tokens: +e.target.value })} />
          </label>
          <label>
            Temperature <span>{opts.temperature.toFixed(2)}</span>
            <input type="range" min={0} max={2} step={0.05} value={opts.temperature} onChange={(e) => setOpts({ ...opts, temperature: +e.target.value })} />
          </label>
          <label>
            Top-p <span>{opts.top_p.toFixed(2)}</span>
            <input type="range" min={0} max={1} step={0.01} value={opts.top_p} onChange={(e) => setOpts({ ...opts, top_p: +e.target.value })} />
          </label>
          <label>
            Top-k <span>{opts.top_k || "off"}</span>
            <input type="range" min={0} max={200} step={1} value={opts.top_k} onChange={(e) => setOpts({ ...opts, top_k: +e.target.value })} />
          </label>
          <label>
            Seed (0 = random)
            <input type="number" value={opts.seed} onChange={(e) => setOpts({ ...opts, seed: Math.max(0, +e.target.value || 0) })} />
          </label>
          <div className="section-title">System prompt</div>
          <textarea rows={5} value={system} onChange={(e) => setSystem(e.target.value)} />
          <div className="section-title">Loading</div>
          <label>
            Device
            <select value={device} onChange={(e) => setDevice(e.target.value as "gpu" | "cpu")}>
              <option value="gpu">GPU (CUDA)</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
          <label>
            Safetensors weights
            <select value={dtype} onChange={(e) => setDtype(e.target.value as "bf16" | "f32")}>
              <option value="bf16">BF16 (half memory)</option>
              <option value="f32">F32 (reference precision)</option>
            </select>
          </label>
          <div className="muted small">GGUF files keep their quantized weights and run through the fused dequant kernels.</div>
        </aside>
      )}
    </div>
  );
}
