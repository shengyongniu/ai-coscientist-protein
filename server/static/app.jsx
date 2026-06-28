const { useState, useEffect, useRef, useCallback } = React;

const AGENTS = ["supervisor", "generation", "reflection", "proximity", "ranking", "evolution", "meta_review"];

// Minimal, safe-ish markdown renderer (headings, lists, code, bold/italic).
function renderMarkdown(md) {
  if (!md) return "";
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const lines = md.split("\n");
  let html = "", inCode = false, inList = false;
  for (let line of lines) {
    if (line.trim().startsWith("```")) {
      if (!inCode) { html += "<pre><code>"; inCode = true; }
      else { html += "</code></pre>"; inCode = false; }
      continue;
    }
    if (inCode) { html += esc(line) + "\n"; continue; }
    let l = esc(line);
    l = l.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>")
         .replace(/`(.+?)`/g, "<code>$1</code>");
    if (/^### /.test(l)) { html += `<h3>${l.slice(4)}</h3>`; }
    else if (/^## /.test(l)) { html += `<h2>${l.slice(3)}</h2>`; }
    else if (/^# /.test(l)) { html += `<h1>${l.slice(2)}</h1>`; }
    else if (/^- /.test(l)) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${l.slice(2)}</li>`; }
    else { if (inList) { html += "</ul>"; inList = false; } html += l ? `<p>${l}</p>` : ""; }
  }
  if (inList) html += "</ul>";
  if (inCode) html += "</code></pre>";
  return html;
}

function Leaderboard({ board }) {
  const maxElo = Math.max(1300, ...board.map((h) => h.elo || 0));
  const minElo = Math.min(1100, ...board.map((h) => h.elo || 0));
  return (
    <table>
      <thead>
        <tr><th>#</th><th>Hypothesis</th><th className="num">Elo</th><th className="num">Score</th><th className="num">W/L</th></tr>
      </thead>
      <tbody>
        {board.map((h, i) => (
          <tr key={h.id} className={i === 0 ? "top1" : ""}>
            <td>{i + 1}</td>
            <td>
              {h.title}
              <div className="elo-bar" style={{ width: `${Math.round(100 * ((h.elo - minElo) / (maxElo - minElo + 1)))}%` }} />
            </td>
            <td className="num">{Math.round(h.elo)}</td>
            <td className="num">{h.score != null ? h.score.toFixed(3) : "—"}</td>
            <td className="num">{h.wins}/{h.losses}</td>
          </tr>
        ))}
        {board.length === 0 && <tr><td colSpan="5" style={{ color: "var(--muted)" }}>Awaiting first tournament…</td></tr>}
      </tbody>
    </table>
  );
}

function App() {
  const [configs, setConfigs] = useState([]);
  const [goal, setGoal] = useState("");
  const [config, setConfig] = useState("protein_binder");
  const [provider, setProvider] = useState("");
  const [scorer, setScorer] = useState("");
  const [rounds, setRounds] = useState("");
  const [events, setEvents] = useState([]);
  const [board, setBoard] = useState([]);
  const [overview, setOverview] = useState(null);
  const [usage, setUsage] = useState(null);
  const [running, setRunning] = useState(false);
  const [round, setRound] = useState(-1);
  const feedRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    fetch("/api/configs").then((r) => r.json()).then((d) => setConfigs(d.configs || []));
  }, []);

  useEffect(() => {
    const c = configs.find((c) => c.name === config);
    if (c && !goal) setGoal(c.goal || "");
  }, [config, configs]);

  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [events]);

  const start = useCallback(() => {
    setEvents([]); setBoard([]); setOverview(null); setUsage(null); setRunning(true); setRound(-1);
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/run`);
    wsRef.current = ws;
    ws.onopen = () => ws.send(JSON.stringify({
      goal, config, provider: provider || null, scorer: scorer || null,
      rounds: rounds ? parseInt(rounds) : null,
    }));
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.control) {
        if (m.control === "overview") setOverview(m.data);
        else if (m.control === "usage") setUsage(m.data);
        else if (m.control === "done") { setRunning(false); ws.close(); }
        return;
      }
      if (m.kind === "leaderboard") setBoard(m.data.leaderboard || []);
      if (m.agent) {
        if (typeof m.round === "number") setRound(m.round);
        setEvents((prev) => [...prev.slice(-200), m]);
      }
    };
    ws.onerror = () => setRunning(false);
    ws.onclose = () => setRunning(false);
  }, [goal, config, provider, scorer, rounds]);

  return (
    <div className="app">
      <header>
        <h1>AI <span className="accent">Co-Scientist</span> — Protein Binder Design</h1>
        <p className="sub">A multi-agent coalition generates, debates, ranks (Elo), and evolves protein designs — grounded by ESM2 / a fine-tuned predictor.</p>
      </header>

      <div className="controls">
        <textarea value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="Research goal…" />
        <div className="field">Config
          <select value={config} onChange={(e) => setConfig(e.target.value)}>
            {configs.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
          </select>
        </div>
        <div className="field">Provider
          <select value={provider} onChange={(e) => setProvider(e.target.value)}>
            <option value="">(default)</option><option value="bedrock">bedrock</option><option value="mock">mock</option>
          </select>
        </div>
        <div className="field">Scorer
          <select value={scorer} onChange={(e) => setScorer(e.target.value)}>
            <option value="">(config)</option><option value="heuristic">heuristic</option><option value="esm">esm</option><option value="predictor">predictor</option>
          </select>
        </div>
        <div className="field">Rounds
          <input type="number" min="1" max="10" value={rounds} onChange={(e) => setRounds(e.target.value)} placeholder="auto" style={{ width: 64 }} />
        </div>
        <button className="run" onClick={start} disabled={running}>{running ? "Running…" : "Run"}</button>
      </div>

      <div className="phase">
        {AGENTS.map((a) => (
          <span key={a} className={"chip" + (events.some((e) => e.agent === a) ? " active" : "")}>{a}</span>
        ))}
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Live Agent Activity</h2>
          <div className="feed" ref={feedRef}>
            {events.map((e, i) => (
              <div className="event" key={i}>
                <span className={"badge " + e.agent}>{e.agent}</span>
                <span className="rnd">r{e.round}</span>
                <span className="msg">{e.message}</span>
              </div>
            ))}
            {events.length === 0 && <div style={{ color: "var(--muted)" }}>Press Run to start a session.</div>}
          </div>
        </div>
        <div className="panel">
          <h2>Hypothesis Leaderboard</h2>
          <Leaderboard board={board} />
        </div>
      </div>

      <div className="statusbar">
        <span className={"dot" + (running ? " live" : "")} />
        <span>{running ? `Running — round ${round + 1}` : overview ? "Complete" : "Idle"}</span>
        {usage && <span>· {usage.calls} LLM calls · {usage.input_tokens}/{usage.output_tokens} tok · ~${usage.cost_usd}</span>}
      </div>

      {overview && (
        <div className="panel overview">
          <h2>Research Overview</h2>
          <div className="md" dangerouslySetInnerHTML={{ __html: renderMarkdown(overview.markdown) }} />
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
