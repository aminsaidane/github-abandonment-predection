import React, { useEffect, useState } from "react";

const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(1)) : v);
const typ = (c) => (c.feature === "is_org" ? (c.typical ? "Yes" : "No") : fmt(c.typical));
const signed = (x) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(1)}`;

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
  return data;
}

export default function App() {
  const [tab, setTab] = useState("forecast");
  const [health, setHealth] = useState(null);
  useEffect(() => { api("/api/health").then(setHealth).catch(() => {}); }, []);
  return (
    <div className="wrap">
      <header>
        <h1>Will this GitHub repository be abandoned?</h1>
        <p>Paste a repository link. The model reads its first six months of activity and estimates the chance that it ends up with no push for more than a year.</p>
      </header>
      <nav>
        <button className={tab === "forecast" ? "on" : ""} onClick={() => setTab("forecast")}>Forecast</button>
        <button className={tab === "perf" ? "on" : ""} onClick={() => setTab("perf")}>How good is the model?</button>
      </nav>
      {tab === "forecast" ? <Forecast health={health} /> : <Performance />}
      <footer>
        Logistic regression on {health ? health.trained_on : "…"} · public repositories with 5+ stars created {health ? `${health.training_years[0]}–${health.training_years[1]}` : "…"} ·
        abandoned = no push for more than 365 days.
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ forecast */
function Forecast({ health }) {
  const [url, setUrl] = useState("");
  const [examples, setExamples] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => { api("/api/examples").then(setExamples).catch(() => {}); }, []);

  const run = async (call) => {
    setBusy(true); setError(null);
    try { setResult(await call()); } catch (e) { setResult(null); setError(e.message); }
    setBusy(false);
  };
  const submit = (e) => { e.preventDefault(); if (url.trim()) run(() => api("/api/predict", { url })); };
  const useExample = (name) => { setUrl(`https://github.com/${name}`); run(() => api("/api/score_example", { name })); };

  const live = health ? health.live_lookup : true;
  const groups = [["flagged and abandoned", "Correctly flagged"], ["flagged but active", "Flagged, but still active"],
    ["not flagged and active", "Correctly not flagged"], ["missed", "Missed (abandoned, not flagged)"]];

  return (
    <>
      <div className="card">
        <h2>Repository</h2>
        <p className="sub">Example: https://github.com/owner/name or just owner/name. The repository must be at least 6 months old.</p>
        <form className="lookup" onSubmit={submit}>
          <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://github.com/owner/repository" aria-label="Repository link" />
          <button className="primary" disabled={busy || !url.trim() || !live}>{busy ? "Working…" : "Forecast"}</button>
        </form>
        {!live && <div className="notice warn" style={{ marginTop: 12 }}>Live lookup is off (no GitHub token on the server). The examples below still work.</div>}
        <div className="chips">
          <span className="lbl">Try a repository the model never saw:</span>
          {examples.map((e) => (
            <button key={e.name} className="chip" onClick={() => useExample(e.name)} title={groups.find((g) => g[0] === e.case)?.[1]}>
              {e.name}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="notice err">{error}</div>}
      {busy && <div className="spinner">Reading the repository's first six months from GitHub… this can take 10–60 seconds for a busy repository.</div>}
      {result && !busy && <Result r={result} />}
    </>
  );
}

function Result({ r }) {
  const p = r.p_abandoned;
  const avg = r.average_repo_p_abandoned;
  const info = r.repo;
  const status = info.observed || (info.today ? (info.today.abandoned_by_definition ? "abandoned" : "active") : null);
  return (
    <>
      {r.warnings.map((w) => <div key={w} className="notice warn">{w}</div>)}
      <div className="card">
        <div className="hero">
          <div className="big">{pct(p)}</div>
          <div>
            <span className={`verdict ${r.flagged ? "flag" : "ok"}`}>{r.flagged ? "Flagged: high abandonment risk" : "Not flagged"}</span>
            <h2 style={{ margin: "2px 0 0" }}><a href={info.url} target="_blank" rel="noreferrer" style={{ color: "inherit" }}>{info.full_name}</a></h2>
            <p>
              Estimated chance of being abandoned. The average repository in the training data scores {pct(avg)}; the model flags a repository from {pct(r.threshold)}.
            </p>
          </div>
        </div>
        <div className="scale" aria-hidden>
          <div className="tick" style={{ left: `${r.threshold * 100}%` }} />
          <div className="dot" style={{ left: `${p * 100}%` }} />
        </div>
        <div className="scale-lbl" aria-hidden>
          <span style={{ left: "0%", transform: "none" }}>0%</span>
          <span style={{ left: `${r.threshold * 100}%` }}>cut-off {pct(r.threshold)}</span>
        </div>
      </div>

      <div className="card">
        <h2>The repository</h2>
        <div className="meta" style={{ marginTop: 10 }}>
          <div>Created<b>{info.created_at}</b></div>
          <div>Stars<b>{fmt(info.stars)}</b></div>
          <div>Main language<b>{info.language}</b></div>
          {info.today && <div>Last push<b>{info.today.last_push} ({fmt(info.today.days_since_last_push)} days ago)</b></div>}
          {status && (
            <div>Actual status today
              <b><span className={`status-pill ${status}`}>{status === "abandoned" ? "Abandoned" : "Active"}</span></b>
            </div>
          )}
          <div>Data source<b style={{ fontSize: 13 }}>{r.source}</b></div>
        </div>
        {info.observed && <p className="sub" style={{ margin: "12px 0 0" }}>This repository is in the held-out test set: the model never saw it during training, and its real outcome is known.</p>}
        {info.today && !info.observed && <p className="sub" style={{ margin: "12px 0 0" }}>The model forecasts from the first six months only, so the status above is the real outcome so far and tells you whether the forecast was right.</p>}
      </div>

      <Why r={r} />
    </>
  );
}

/* ------------------------------------------------------------------ why this score */
function Why({ r }) {
  const [view, setView] = useState("chart");
  const [tip, setTip] = useState(null);
  const rows = r.contributions;
  const max = Math.max(3, ...rows.map((c) => Math.abs(c.points)));
  const total = rows.reduce((s, c) => s + c.points, 0);

  const show = (e, c) => setTip({ x: e.clientX + 14, y: e.clientY + 14, c });
  return (
    <div className="card">
      <div className="head-row">
        <div>
          <h2>Why this score</h2>
          <p className="sub">
            Each bar shows how much a feature moves this repository's risk away from the average repository ({pct(r.average_repo_p_abandoned)}), in percentage points.
            Red raises the risk, blue lowers it. The bars add up to the {signed(total)} point gap between this repository and the average.
          </p>
        </div>
        <div className="toggle">
          <button className={view === "chart" ? "on" : ""} onClick={() => setView("chart")}>Chart</button>
          <button className={view === "table" ? "on" : ""} onClick={() => setView("table")}>Table</button>
        </div>
      </div>

      {view === "chart" ? (
        <div className="diverge" onMouseLeave={() => setTip(null)}>
          <div className="axis-lbl"><span /><div><span>← lowers risk</span><span>raises risk →</span></div><span /></div>
          {rows.map((c) => {
            const w = (Math.abs(c.points) / max) * 50;
            const up = c.points > 0;
            return (
              <div className="row" key={c.feature} onMouseMove={(e) => show(e, c)} onMouseLeave={() => setTip(null)}>
                <div className="name">{c.label}<small>{fmt(c.value)} <span title="Typical repository in the training data">(typical {typ(c)})</span></small></div>
                <div className="track"><div className={`bar ${up ? "raise" : "lower"}`} style={{ width: `${w}%` }} /></div>
                <div className={`pts ${up ? "raise" : "lower"}`}>{signed(c.points)}</div>
              </div>
            );
          })}
          {tip && (
            <div className="tip" style={{ left: tip.x, top: tip.y }}>
              <b>{tip.c.label}</b><br />
              This repository: {fmt(tip.c.value)}<br />
              Typical repository: {typ(tip.c)}<br />
              Effect: {signed(tip.c.points)} points ({tip.c.effect} the risk)
            </div>
          )}
        </div>
      ) : (
        <table>
          <thead><tr><th>Feature</th><th className="num">This repo</th><th className="num">Typical</th><th className="num">Effect (points)</th></tr></thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.feature}><td>{c.label}</td><td className="num">{fmt(c.value)}</td><td className="num">{typ(c)}</td>
                <td className="num">{signed(c.points)} ({c.effect})</td></tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="sub" style={{ margin: "14px 0 0" }}>
        “Typical” is the median repository of the training data. The explanation is exact for this model (a logistic regression), not an approximation.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ performance */
function Tile({ k, v, d }) {
  return <div className="tile"><div className="v">{v}</div><div className="k">{k}</div><div className="d">{d}</div></div>;
}

function Performance() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api("/api/performance").then(setD).catch((e) => setErr(e.message)); }, []);
  if (err) return <div className="notice err">{err}</div>;
  if (!d) return <div className="spinner">Loading…</div>;
  const m = d.metrics, cm = m.confusion;
  const captions = {
    "final_roc_curve.png": "ROC curve: how well the score separates the two groups",
    "final_calibration.png": "Calibration: predicted risk versus what really happened",
    "final_confusion_matrix.png": "Confusion matrix at the 0.84 cut-off",
    "final_threshold_tradeoff.png": "What each possible cut-off would trade off",
  };
  return (
    <>
      <div className="card">
        <h2>Result on a final test set the model never saw</h2>
        <p className="sub">{m.n_test.toLocaleString()} repositories, evaluated once, after every modelling choice was fixed. {pct(m.abandon_rate_test, 1)} of them were abandoned.</p>
        <div className="tiles">
          <Tile k="Ranking quality (ROC AUC)" v={m.roc_auc.toFixed(2)} d={`95% CI ${m.roc_auc_ci[0].toFixed(2)}–${m.roc_auc_ci[1].toFixed(2)}. 0.5 is a coin flip; a single simple rule scores ${m.baseline_roc_auc_single_rule.toFixed(2)}.`} />
          <Tile k="Balanced accuracy" v={pct(m.balanced_accuracy)} d={`Right on ${pct(m.recall_abandoned)} of abandoned and ${pct(m.recall_active)} of active repositories at the cut-off.`} />
          <Tile k="Precision, abandoned" v={pct(m.precision_abandoned)} d={`Of flagged repositories, ${pct(m.precision_abandoned)} were abandoned. The base rate is already ${pct(m.abandon_rate_test)}.`} />
          <Tile k="Precision, active" v={pct(m.precision_active)} d="Of repositories NOT flagged, only about one in three stayed active. Weak: see limits." />
          <Tile k="Calibration error (ECE)" v={m.ece.toFixed(3)} d={`Predicted ${pct(m.mean_predicted, 1)} abandoned on average, observed ${pct(m.abandon_rate_test, 1)}.`} />
        </div>
      </div>

      <div className="card">
        <h2>What the model does at the {d.decisions.threshold} cut-off</h2>
        <div className="cm" style={{ marginTop: 10 }}>
          <span /><span className="h">Not flagged</span><span className="h">Flagged</span>
          <span className="r">Really active</span>
          <div className="c good"><b>{cm.true_active_pred_active}</b>correctly left alone</div>
          <div className="c bad"><b>{cm.true_active_flagged}</b>wrongly flagged</div>
          <span className="r">Really abandoned</span>
          <div className="c bad"><b>{cm.true_abandoned_missed}</b>missed</div>
          <div className="c good"><b>{cm.true_abandoned_flagged}</b>correctly flagged</div>
        </div>
      </div>

      <div className="card">
        <h2>Limits to keep in mind</h2>
        <ul className="limits">
          <li><b>Most repositories get abandoned.</b> {pct(m.abandon_rate_test)} of the test set was abandoned, so a high score is not surprising. The useful signal is the <i>difference</i> from the average, which is why the chart explains the gap to the average repository.</li>
          <li><b>“Not flagged” is a weak guarantee.</b> Only about a third of unflagged repositories stayed active (precision {pct(m.precision_active)}, against a {pct(1 - m.abandon_rate_test)} base rate). The model is better at ranking than at certifying a project as healthy.</li>
          <li><b>Older repositories look worse.</b> A repository created in 2012 has had far longer to go quiet than one created in 2022, so the observed abandonment rate falls with creation year. The model ranks well within each period, but its raw probabilities are less reliable for very recent repositories.</li>
          <li><b>Scope.</b> Trained on public repositories with at least 5 stars, created 2009–2023. Younger repositories, repositories with fewer stars, and private projects are outside what it was built for.</li>
          <li><b>Correlation, not cause.</b> The bars describe patterns in past repositories; they do not say that changing one number would change the outcome.</li>
        </ul>
        <h2 style={{ marginTop: 18 }}>By creation period</h2>
        <table style={{ marginTop: 6 }}>
          <thead><tr><th>Created</th><th className="num">Repositories</th><th className="num">Abandoned (observed)</th><th className="num">Predicted (mean)</th><th className="num">ROC AUC</th></tr></thead>
          <tbody>{m.by_creation_period.map((p) => (
            <tr key={p.created}><td>{p.created}</td><td className="num">{p.n}</td><td className="num">{pct(p.observed_abandoned, 1)}</td>
              <td className="num">{pct(p.mean_predicted, 1)}</td><td className="num">{p.roc_auc != null ? p.roc_auc.toFixed(2) : "–"}</td></tr>
          ))}</tbody>
        </table>
      </div>

      <div className="card">
        <h2>Figures</h2>
        <div className="figs" style={{ marginTop: 10 }}>
          {d.figures.map((f) => (
            <figure key={f} style={{ margin: 0 }}><img src={`/figures/${f}`} alt={captions[f] || f} /><figcaption>{captions[f]}</figcaption></figure>
          ))}
        </div>
      </div>
    </>
  );
}
