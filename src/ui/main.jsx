import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, ArrowUpRight, Check, ChevronDown, Clock3, Film,
  Gauge, Languages, Menu, Play, Sparkles, UploadCloud, WandSparkles, X,
} from "lucide-react";
import "./styles.css";

const presets = [
  ["Binary Search Trees", "Today we will analyze balanced binary search trees and their asymptotic time complexity."],
  ["Thermodynamics", "The second law of thermodynamics states that total entropy of an isolated system always increases over time."],
  ["Linear Transformations", "An eigenvector of a square matrix represents a direction invariant under linear transformations."],
];

function Metric({ label, value, note, tone = "" }) {
  return <div className="metric"><span>{label}</span><strong className={tone}>{value}</strong><small>{note}</small></div>;
}

function App() {
  const [active, setActive] = useState("playground");
  const [text, setText] = useState(presets[0][1]);
  const [duration, setDuration] = useState(4);
  const [source, setSource] = useState("English");
  const [target, setTarget] = useState("Hindi");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    let mounted = true;
    fetch("/api/health")
      .then(response => {
        if (!response.ok) throw new Error("Health check failed");
        return response.json();
      })
      .then(data => {
        if (mounted) setApiStatus(data.status === "ok" ? "online" : "offline");
      })
      .catch(() => {
        if (mounted) setApiStatus("offline");
      });
    return () => { mounted = false; };
  }, []);

  async function translate() {
    if (loading || !text.trim()) {
      setNotice("Enter a sentence before starting translation.");
      return;
    }
    setLoading(true); setNotice("");
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 120000);
    try {
      const response = await fetch("/api/translate", { method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text, duration, srcLang: source === "English" ? "en" : "hi", tgtLang: target === "Hindi" ? "hi" : "en" }),
        signal: controller.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Translation failed");
      setResult(data);
    } catch (error) {
      const message = error.name === "AbortError"
        ? "Translation timed out. The first model load can take a while; please try again."
        : error.message;
      setNotice(`${message}. Start the Python backend to enable live translation.`);
    } finally {
      window.clearTimeout(timeout);
      setLoading(false);
    }
  }

  const nav = [
    ["playground", <Sparkles size={17} />, "Model playground"],
    ["studio", <Film size={17} />, "Video studio"],
    ["evaluation", <Gauge size={17} />, "Evaluation"],
  ];
  return <div className="app">
    <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
      <div className="brand"><div className="brand-mark"><WandSparkles size={19} /></div><div><b>Subtitle AI</b><span>Localization studio</span></div><button className="close" onClick={() => setSidebarOpen(false)}><X size={18}/></button></div>
      <div className="workspace"><span className="eyebrow">Workspace</span><div className="workspace-btn">Local translation workspace <ChevronDown size={15}/></div></div>
      <nav>{nav.map(([id, icon, label]) => <button key={id} className={active === id ? "active" : ""} onClick={() => {setActive(id); setSidebarOpen(false)}}>{icon}<span>{label}</span>{id === "playground" && <i>Live</i>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="status"><span className={`pulse ${apiStatus === "offline" ? "offline" : ""}`}/><div><b>{apiStatus === "online" ? "API online" : apiStatus === "offline" ? "API offline" : "Checking API"}</b><small>Translation service status</small></div></div></div>
    </aside>
    <main>
      <header className="topbar"><button className="menu" onClick={() => setSidebarOpen(true)}><Menu size={21}/></button><div className="crumb"><span>Workspace</span><b>/</b><strong>{nav.find(n => n[0] === active)?.[2]}</strong></div><div className="top-actions"><span className="online"><span className={`pulse ${apiStatus === "offline" ? "offline" : ""}`}/> {apiStatus === "online" ? "API connected" : apiStatus === "offline" ? "API unavailable" : "Checking API"}</span><button className="icon-btn"><Activity size={17}/></button></div></header>
      <div className="content">
        <section className="hero"><div className="hero-copy"><span className="eyebrow light">Neural subtitle workspace</span><h1>Make every line<br/><em>land perfectly.</em></h1><p>Translate text and inspect the reading-speed metrics returned by your local model.</p><div className="hero-actions"><button className="button primary" onClick={() => setActive("studio")}>Open video studio <ArrowUpRight size={16}/></button><span className="hero-meta"><Check size={15}/> Metrics appear after translation</span></div></div><div className="hero-art"><div className="orb orb-one"/><div className="orb orb-two"/><div className="float-card"><span>LAST RESULT</span><strong>{result ? `${result.constrained.metrics.cps} CPS` : "—"}</strong><small>{result ? `${result.constrained.metrics.max_line_cpl} CPL · from API` : "No translation yet"}</small></div><div className="grid-lines"/></div></section>
        {active === "playground" && <><div className="section-head"><div><span className="eyebrow">Compare · refine · ship</span><h2>Model playground</h2><p>See the trade-off between raw translation and subtitle-ready phrasing.</p></div><div className="live-pill"><span className="pulse"/> Live inference</div></div>
          <section className="grid two"><div className="panel input-panel"><div className="panel-title"><div><span className="step">01</span><h3>Source sentence</h3></div><button className="ghost"><UploadCloud size={15}/> Import</button></div><label>Educational sample</label><div className="preset-row">{presets.map(([label, value]) => <button key={label} className={text === value ? "selected" : ""} onClick={() => setText(value)}>{label}</button>)}</div><textarea value={text} onChange={e => setText(e.target.value)} /><div className="input-footer"><span>{text.length} characters</span><label><Clock3 size={14}/> Cue duration <input type="number" min="1" max="12" value={duration} onChange={e => setDuration(e.target.value)}/> sec</label></div><div className="language-row"><div><Languages size={15}/><select value={source} onChange={e=>setSource(e.target.value)}><option>English</option><option>Hindi</option></select></div><span>→</span><div><Languages size={15}/><select value={target} onChange={e=>setTarget(e.target.value)}><option>Hindi</option><option>English</option></select></div></div><button className="button primary full" onClick={translate} disabled={loading}>{loading ? "Translating..." : "Translate & compare"} <Sparkles size={16}/></button>{notice && <div className="notice">{notice}</div>}</div>
          <div className="panel output-panel"><div className="panel-title"><div><span className="step">02</span><h3>Output comparison</h3></div><span className="ai-label"><Sparkles size={14}/> AI optimized</span></div>{result ? <div className="outputs"><OutputCard label="Baseline · unconstrained" text={result.baseline.text} metrics={result.baseline.metrics} /><OutputCard label="Constraint-aware NMT" text={result.constrained.text} metrics={result.constrained.metrics} good /></div> : <div className="empty-state"><div className="empty-icon"><WandSparkles size={23}/></div><h3>Your comparison appears here</h3><p>Run the sentence through the backend to compare meaning preservation with screen-fit constraints.</p><div className="empty-line"><span/><span/><span/></div></div>}</div></section>
          <div className="section-head compact"><div><span className="eyebrow">Current result</span><h2>Measured constraints</h2></div></div><section className="metrics"><Metric label="CPL compliance" value={result ? (result.constrained.metrics.cpl_compliant ? "Pass" : "Fail") : "—"} note="From current API result" tone="green"/><Metric label="CPS compliance" value={result ? (result.constrained.metrics.cps_compliant ? "Pass" : "Fail") : "—"} note="From current API result" tone="blue"/><Metric label="Maximum line length" value={result ? `${result.constrained.metrics.max_line_cpl} ch` : "—"} note="Current constrained output" /><Metric label="Character rate" value={result ? `${result.constrained.metrics.cps} CPS` : "—"} note="Current constrained output" tone="orange"/></section>
        </>}
        {active === "studio" && <Studio onBack={() => setActive("playground")} />}
        {active === "evaluation" && <Evaluation />}
      </div>
    </main>
  </div>;
}

function OutputCard({label, text, metrics, good}) { return <div className={`output-card ${good ? "good" : ""}`}><div className="output-label"><span className="dot"/>{label}<span className="fit">{good ? "✓ Fits constraints" : "Needs review"}</span></div><p>{text}</p><div className="output-stats"><span><b>{metrics.max_line_cpl}</b> CPL</span><span><b>{metrics.cps}</b> CPS</span><span><b>{metrics.char_count}</b> chars</span></div></div> }
function Studio({onBack}) { return <><div className="section-head"><div><span className="eyebrow">Production workspace</span><h2>Video localization studio</h2><p>File upload and video preview are not connected yet.</p></div><button className="button secondary" onClick={onBack}>Back to playground</button></div><div className="studio-grid"><div className="dropzone"><div className="upload-icon"><UploadCloud size={28}/></div><h3>Subtitle file import</h3><p>Upload processing is not available in this version.</p><button className="button primary" disabled>Choose file</button><span>Use the Model playground for text translation.</span></div><div className="panel video-preview"><div className="preview-top"><span><Film size={16}/> Video preview</span><span className="tag">NOT CONNECTED</span></div><div className="video-placeholder"><Play size={28}/><div className="subtitles">No video loaded</div></div><div className="timeline"><span/><i/><b/></div><div className="preview-controls"><span>—</span><span>—</span></div></div></div></> }
function Evaluation() { return <><div className="section-head"><div><span className="eyebrow">Evaluation</span><h2>Evaluation dashboard</h2><p>No benchmark data is connected to the frontend.</p></div></div><div className="panel chart-panel empty-state"><div className="empty-icon"><Gauge size={23}/></div><h3>Evaluation data unavailable</h3><p>BLEU, chrF, and aggregate benchmark values will appear here when an evaluation endpoint is connected.</p></div></> }
createRoot(document.getElementById("root")).render(<App />);
