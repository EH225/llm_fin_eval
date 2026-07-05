"""
Generates a self-contained interactive HTML report with agent comparison.

Usage:
  python dashboard.py                  # uses latest summary
  python dashboard.py --summary path   # specific summary
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def get_latest_summary_filename() -> Path:
    """
    Returns the latest summary JSON file saved in the results folder by sorting by name since summary
    results are saved with date and time labels.
    """
    files = sorted(RESULTS_DIR.glob("*_summary.json"))
    if not files:
        print("No summary found. Run: python analyze.py")
        sys.exit(1)
    return files[-1]


def build_html(summary: dict, run_name: str) -> str:
    """
    Builds an HTML summary of the results based on an input summary (a JSON) read from disk.
    """
    agents = summary.get("agents", ["claude"])
    agent_labels = summary.get("agent_labels", {"claude": "Claude Sonnet 4.6"})
    agent_stats = summary.get("agent_stats", {})
    h2h = summary.get("head_to_head", [])
    multi = len(agents) >= 2

    # Colours per agent slot
    AGENT_COLORS = ["#4f8ef7", "#f7954f", "#4ff787", "#f74f8e"]
    agent_color = {a: AGENT_COLORS[i % len(AGENT_COLORS)] for i, a in enumerate(agents)}

    agents_js = json.dumps(agents)
    agent_labels_js = json.dumps(agent_labels)
    agent_stats_js = json.dumps(agent_stats, default=str)
    h2h_js = json.dumps(h2h, default=str)
    agent_color_js = json.dumps(agent_color)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FinEval — Agent Comparison Dashboard</title>
<style>
  :root {{
    --bg:#0f1117;--surface:#1a1d2e;--surface2:#252842;--border:#2e3150;
    --accent:#4f8ef7;--accent2:#7c5cbf;--green:#22c55e;--red:#ef4444;
    --yellow:#f59e0b;--text:#e2e8f0;--muted:#64748b;
    --mono:'JetBrains Mono','Fira Code',monospace;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;min-height:100vh}}

  header{{background:linear-gradient(135deg,#0f1117,#1a1d2e,#151829);border-bottom:1px solid var(--border);padding:28px 48px;display:flex;align-items:flex-start;justify-content:space-between}}
  header h1{{font-size:1.6rem;font-weight:700;letter-spacing:-.02em}}
  header h1 span{{color:var(--accent)}}
  .run-tag{{font-family:var(--mono);font-size:.7rem;color:var(--muted);background:var(--surface2);border:1px solid var(--border);padding:3px 9px;border-radius:5px;margin-top:5px;display:inline-block}}

  /* Tabs */
  .tabs{{display:flex;gap:4px;padding:20px 32px 0;border-bottom:1px solid var(--border);background:var(--surface)}}
  .tab{{padding:10px 22px;cursor:pointer;font-size:.85rem;font-weight:500;color:var(--muted);border-radius:6px 6px 0 0;border:1px solid transparent;border-bottom:none;transition:.15s}}
  .tab.active{{color:var(--text);background:var(--bg);border-color:var(--border)}}
  .tab:hover:not(.active){{color:var(--text)}}

  main{{max-width:1350px;margin:0 auto;padding:36px 32px}}

  /* KPI row */
  .kpi-row{{display:grid;gap:14px;margin-bottom:36px}}
  .kpi{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px;position:relative;overflow:hidden}}
  .kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px}}
  .kpi .label{{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}}
  .kpi .value{{font-size:1.8rem;font-weight:700;font-family:var(--mono)}}
  .kpi .sub{{font-size:.75rem;color:var(--muted);margin-top:3px}}
  .green{{color:var(--green)}}.yellow{{color:var(--yellow)}}.red{{color:var(--red)}}

  h2{{font-size:1rem;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px}}
  .badge{{font-size:.68rem;background:var(--surface2);border:1px solid var(--border);padding:2px 8px;border-radius:20px;color:var(--muted);font-weight:400}}
  .section{{margin-bottom:36px}}

  /* Tables */
  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  th{{text-align:left;padding:9px 14px;font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);border-bottom:1px solid var(--border);background:var(--surface2)}}
  td{{padding:11px 14px;border-bottom:1px solid var(--border);vertical-align:top}}
  tr:hover td{{background:var(--surface2);cursor:pointer}}
  .mono{{font-family:var(--mono);font-size:.82rem}}
  .muted{{color:var(--muted);font-size:.75rem}}

  .tier-badge{{display:inline-block;font-size:.63rem;font-weight:600;padding:2px 6px;border-radius:4px;text-transform:uppercase}}
  .tier-1{{background:#1a3a1a;color:var(--green);border:1px solid #22c55e44}}
  .tier-2{{background:#3a2a0a;color:var(--yellow);border:1px solid #f59e0b44}}
  .tier-3{{background:#3a0a0a;color:var(--red);border:1px solid #ef444444}}

  .pass-bar{{height:5px;border-radius:3px;background:var(--surface2);overflow:hidden;margin-top:5px;width:80px}}
  .pass-bar-fill{{height:100%;border-radius:3px}}
  .grader-chip{{display:inline-block;font-size:.65rem;font-weight:600;padding:2px 7px;border-radius:4px;text-transform:uppercase;background:var(--surface2);border:1px solid var(--border);color:var(--muted)}}

  /* Comparison bar */
  .cmp-row{{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid var(--border)}}
  .cmp-row:last-child{{border-bottom:none}}
  .cmp-task{{width:240px;font-size:.82rem}}
  .cmp-task .tid{{font-family:var(--mono);color:var(--accent);font-size:.75rem}}
  .cmp-bars{{flex:1;display:flex;flex-direction:column;gap:5px}}
  .cmp-agent-row{{display:flex;align-items:center;gap:8px}}
  .cmp-agent-label{{width:130px;font-size:.72rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  .cmp-bar-bg{{flex:1;height:14px;background:var(--surface2);border-radius:3px;overflow:hidden}}
  .cmp-bar-fill{{height:100%;border-radius:3px;display:flex;align-items:center;padding-left:6px;font-size:.65rem;font-weight:600;color:#000}}
  .cmp-pct{{width:48px;font-family:var(--mono);font-size:.75rem;text-align:right}}
  .winner-badge{{font-size:.65rem;padding:2px 7px;border-radius:10px;font-weight:700;white-space:nowrap}}

  /* Failure cards */
  .failure-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}}
  .failure-card{{background:var(--surface);border:1px solid var(--border);border-radius:9px;padding:18px}}
  .fm-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
  .fm-category{{font-weight:600;font-size:.88rem}}
  .fm-count{{font-family:var(--mono);font-size:.72rem;background:#ef444422;color:var(--red);border:1px solid #ef444444;padding:2px 8px;border-radius:20px}}
  .failure-example{{background:var(--surface2);border-radius:5px;padding:9px 11px;font-size:.75rem;color:var(--muted);margin-top:7px;line-height:1.5}}
  .ex-task{{color:var(--accent);font-family:var(--mono);margin-bottom:3px;font-size:.7rem}}

  /* Drawer */
  #drawer{{position:fixed;right:0;top:0;bottom:0;width:500px;background:var(--surface);border-left:1px solid var(--border);padding:26px;overflow-y:auto;transform:translateX(100%);transition:.22s ease;z-index:100}}
  #drawer.open{{transform:translateX(0)}}
  #drawer-close{{float:right;cursor:pointer;font-size:1.1rem;color:var(--muted);background:none;border:none;padding:3px 7px}}
  .drawer-meta{{font-size:.76rem;color:var(--muted);margin-bottom:18px}}
  .stat-row{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border);font-size:.83rem}}
  .stat-val{{font-family:var(--mono)}}
  .fail-item{{background:var(--surface2);border-radius:5px;padding:9px 11px;margin-bottom:7px;font-size:.76rem;line-height:1.5}}
  .fi-trial{{color:var(--muted);font-family:var(--mono);margin-bottom:3px}}
  .fi-exp{{color:var(--red);margin-top:3px;font-size:.7rem}}

  .page{{display:none}}.page.active{{display:block}}

  footer{{text-align:center;padding:36px;color:var(--muted);font-size:.78rem}}
</style>
</head>
<body>

<header>
  <div>
    <h1>Fin<span>Eval</span> — Agent Comparison Dashboard</h1>
    <div class="run-tag">{run_name}</div>
  </div>
  <div style="text-align:right;font-size:.78rem;color:var(--muted)">LLM Financial Analysis Benchmark</div>
</header>

<div class="tabs" id="tabs"></div>

<main>
  <div class="page active" id="page-overview"></div>
  <div class="page" id="page-compare"></div>
  <div class="page" id="page-failures"></div>
</main>

<div id="drawer">
  <button id="drawer-close" onclick="closeDrawer()">✕</button>
  <h3 id="drawer-title" style="font-size:.95rem;margin-bottom:3px"></h3>
  <div class="drawer-meta" id="drawer-meta"></div>
  <div id="drawer-stats"></div>
  <div id="drawer-fails" style="margin-top:14px"></div>
</div>

<footer>FinEval · Graders: exact · tolerance · range · llm_rubric</footer>

<script>
const AGENTS = {agents_js};
const AGENT_LABELS = {agent_labels_js};
const AGENT_STATS = {agent_stats_js};
const H2H = {h2h_js};
const AGENT_COLOR = {agent_color_js};
const MULTI = {str(multi).lower()};

function passColor(r) {{
  return r >= .8 ? 'var(--green)' : r >= .5 ? 'var(--yellow)' : 'var(--red)';
}}
function tierLabel(t) {{ return ['','Easy','Medium','Hard'][t]||t; }}

// ── Tabs ──────────────────────────────────────────────────────────────────────
const tabDefs = [
  {{id:'overview', label:'Per-Agent Results'}},
  ...(MULTI ? [{{id:'compare', label:'Head-to-Head'}},] : []),
  {{id:'failures', label:'Failure Analysis'}},
];
const tabsEl = document.getElementById('tabs');
tabDefs.forEach((td,i) => {{
  const el = document.createElement('div');
  el.className = 'tab' + (i===0?' active':'');
  el.textContent = td.label;
  el.onclick = () => {{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('page-'+td.id).classList.add('active');
  }};
  tabsEl.appendChild(el);
}});

// ── Overview page ─────────────────────────────────────────────────────────────
const overviewEl = document.getElementById('page-overview');

AGENTS.forEach(agentKey => {{
  const s = AGENT_STATS[agentKey];
  if (!s) return;
  const color = AGENT_COLOR[agentKey];
  const label = AGENT_LABELS[agentKey];
  const pr = s.overall_pass_rate * 100;
  const ci = s.overall_pass_ci_95;
  const scoreColor = pr >= 70 ? 'green' : pr >= 40 ? 'yellow' : 'red';

  // Agent section header
  const sec = document.createElement('div');
  sec.style.cssText = 'margin-bottom:40px';
  sec.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px">
      <div style="width:10px;height:10px;border-radius:50%;background:${{color}}"></div>
      <h2 style="margin:0">${{label}}</h2>
    </div>
    <div class="kpi-row" style="grid-template-columns:repeat(4,1fr)">
      <div class="kpi" style="--accent:${{color}}">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:${{color}}"></div>
        <div class="label">Pass Rate</div>
        <div class="value ${{scoreColor}}">${{pr.toFixed(1)}}%</div>
        <div class="sub">95% CI: ${{(ci[0]*100).toFixed(1)}}%–${{(ci[1]*100).toFixed(1)}}%</div>
      </div>
      <div class="kpi">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:${{color}}"></div>
        <div class="label">Mean Score</div>
        <div class="value">${{s.overall_score_mean.toFixed(3)}}</div>
        <div class="sub">σ = ${{s.overall_score_std.toFixed(3)}}</div>
      </div>
      <div class="kpi">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:${{color}}"></div>
        <div class="label">Trials</div>
        <div class="value">${{s.total_trials}}</div>
        <div class="sub">${{Object.keys(s.task_stats).length}} tasks</div>
      </div>
      <div class="kpi">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:${{color}}"></div>
        <div class="label">Failures</div>
        <div class="value ${{s.failure_modes.length > 0 ? 'red' : 'green'}}">${{s.failure_modes.reduce((a,f)=>a+f.count,0)}}</div>
        <div class="sub">${{s.failure_modes.length}} failure mode(s)</div>
      </div>
    </div>
  `;

  // Task table for this agent
  const tableWrap = document.createElement('div');
  tableWrap.className = 'section';
  tableWrap.innerHTML = `<h2>Task Results <span class="badge">click row for failures</span></h2>`;
  const table = document.createElement('table');
  table.innerHTML = `<thead><tr>
    <th>ID</th><th>Task</th><th>Tier</th><th>Grader</th>
    <th>Pass Rate</th><th>Score mean ± σ</th><th>Avg Turns</th><th>Latency</th>
  </tr></thead>`;
  const tbody = document.createElement('tbody');

  Object.values(s.task_stats).forEach(t => {{
    const tpr = t.pass_rate * 100;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="mono" style="color:var(--accent)">${{t.task_id}}</span></td>
      <td style="font-weight:500">${{t.task_name}}</td>
      <td><span class="tier-badge tier-${{t.tier}}">${{tierLabel(t.tier)}}</span></td>
      <td><span class="grader-chip">${{t.grader_type.replace('_',' ')}}</span></td>
      <td>
        <span class="mono" style="color:${{passColor(t.pass_rate)}}">${{tpr.toFixed(1)}}%</span>
        <div class="pass-bar"><div class="pass-bar-fill" style="width:${{tpr}}%;background:${{passColor(t.pass_rate)}}"></div></div>
      </td>
      <td class="mono">${{t.score_mean.toFixed(3)}} <span class="muted">± ${{t.score_std.toFixed(3)}}</span></td>
      <td class="mono">${{t.avg_turns.toFixed(1)}}</td>
      <td class="mono">${{(t.avg_latency_ms/1000).toFixed(1)}}s</td>
    `;
    tr.onclick = () => openDrawer(t, label);
    tbody.appendChild(tr);
  }});

  table.appendChild(tbody);
  tableWrap.appendChild(table);
  sec.appendChild(tableWrap);
  overviewEl.appendChild(sec);
}});

// ── Compare page ──────────────────────────────────────────────────────────────
if (MULTI) {{
  const compareEl = document.getElementById('page-compare');
  compareEl.innerHTML = `<div class="section"><h2>Head-to-Head: Pass Rate by Task</h2></div>`;
  const wrap = compareEl.querySelector('.section');

  // Overall summary row
  const summaryRow = document.createElement('div');
  summaryRow.style.cssText = 'display:flex;gap:20px;margin-bottom:28px;flex-wrap:wrap';
  AGENTS.forEach(a => {{
    const s = AGENT_STATS[a];
    const pr = (s.overall_pass_rate*100).toFixed(1);
    const color = AGENT_COLOR[a];
    summaryRow.innerHTML += `
      <div style="display:flex;align-items:center;gap:8px;padding:10px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;border-left:3px solid ${{color}}">
        <div style="width:8px;height:8px;border-radius:50%;background:${{color}}"></div>
        <span style="font-size:.82rem;font-weight:600">${{AGENT_LABELS[a]}}</span>
        <span class="mono" style="color:${{passColor(s.overall_pass_rate)}};font-size:.9rem">${{pr}}%</span>
        <span class="muted">overall</span>
      </div>`;
  }});
  wrap.appendChild(summaryRow);

  H2H.forEach(row => {{
    const div = document.createElement('div');
    div.className = 'cmp-row';
    div.innerHTML = `
      <div class="cmp-task">
        <div class="tid">${{row.task_id}}</div>
        <div style="font-size:.8rem;font-weight:500;margin-top:2px">${{row.task_name}}</div>
        <div style="margin-top:4px"><span class="tier-badge tier-${{row.tier}}">${{tierLabel(row.tier)}}</span></div>
      </div>
      <div class="cmp-bars">
        ${{AGENTS.map(a => {{
          const pr = row[a+'_pass_rate'];
          const sm = row[a+'_score_mean'];
          const pct = pr !== null ? pr*100 : 0;
          const color = AGENT_COLOR[a];
          return `<div class="cmp-agent-row">
            <div class="cmp-agent-label">${{AGENT_LABELS[a]}}</div>
            <div class="cmp-bar-bg">
              <div class="cmp-bar-fill" style="width:${{pct}}%;background:${{color}}">${{pct >= 20 ? pct.toFixed(0)+'%' : ''}}</div>
            </div>
            <div class="cmp-pct" style="color:${{color}}">${{pr !== null ? pct.toFixed(0)+'%' : 'n/a'}}</div>
            <div class="muted" style="width:52px;font-size:.72rem">${{sm !== null ? 'μ='+sm.toFixed(2) : ''}}</div>
          </div>`;
        }}).join('')}}
      </div>
      <div style="width:100px;text-align:right">
        ${{row.winner && row.winner !== 'tie'
          ? `<span class="winner-badge" style="background:${{AGENT_COLOR[row.winner]}}22;color:${{AGENT_COLOR[row.winner]}};border:1px solid ${{AGENT_COLOR[row.winner]}}44">
              ${{AGENT_LABELS[row.winner].split(' ')[0]}} +${{(row.margin*100).toFixed(0)}}pp
             </span>`
          : `<span class="muted" style="font-size:.72rem">tie</span>`
        }}
      </div>
    `;
    wrap.appendChild(div);
  }});
}}

// ── Failures page ─────────────────────────────────────────────────────────────
const failEl = document.getElementById('page-failures');
AGENTS.forEach(agentKey => {{
  const s = AGENT_STATS[agentKey];
  if (!s || !s.failure_modes.length) return;
  const color = AGENT_COLOR[agentKey];
  const sec = document.createElement('div');
  sec.className = 'section';
  sec.innerHTML = `<h2>${{AGENT_LABELS[agentKey]}} — Failure Modes</h2>`;
  const grid = document.createElement('div');
  grid.className = 'failure-grid';
  s.failure_modes.forEach(fm => {{
    const examples = fm.examples.map(ex => `
      <div class="failure-example">
        <div class="ex-task">${{ex.task_id}} · Trial ${{ex.trial}} · score=${{ex.score.toFixed(2)}}</div>
        <div>${{ex.explanation}}</div>
      </div>`).join('');
    grid.innerHTML += `
      <div class="failure-card" style="border-top:2px solid ${{color}}">
        <div class="fm-header">
          <span class="fm-category">${{fm.category}}</span>
          <span class="fm-count">${{fm.count}}×</span>
        </div>
        ${{examples}}
      </div>`;
  }});
  sec.appendChild(grid);
  failEl.appendChild(sec);
}});
if (!failEl.innerHTML) {{
  failEl.innerHTML = '<p style="color:var(--green);padding:24px">🎉 No failures in this run.</p>';
}}

// ── Drawer ────────────────────────────────────────────────────────────────────
function openDrawer(t, agentLabel) {{
  document.getElementById('drawer-title').textContent = `[${{t.task_id}}] ${{t.task_name}}`;
  document.getElementById('drawer-meta').textContent =
    `${{agentLabel}} · Tier ${{t.tier}} (${{tierLabel(t.tier)}}) · ${{t.grader_type.replace('_',' ')}} · ${{t.n_trials}} trials`;
  const stats = [
    ['Pass Rate', `${{(t.pass_rate*100).toFixed(1)}}% (CI: ${{(t.pass_rate_ci_95[0]*100).toFixed(1)}}–${{(t.pass_rate_ci_95[1]*100).toFixed(1)}}%)`],
    ['Score Mean ± σ', `${{t.score_mean.toFixed(4)}} ± ${{t.score_std.toFixed(4)}}`],
    ['Score CI 95%', `[${{t.score_ci_95[0].toFixed(3)}}, ${{t.score_ci_95[1].toFixed(3)}}]`],
    ['Avg Turns', t.avg_turns.toFixed(1)],
    ['Avg Latency', `${{(t.avg_latency_ms/1000).toFixed(2)}}s`],
    ['Agent Errors', t.n_agent_errors],
  ];
  document.getElementById('drawer-stats').innerHTML = stats.map(([k,v]) =>
    `<div class="stat-row"><span>${{k}}</span><span class="stat-val">${{v}}</span></div>`
  ).join('');
  const fails = t.failures || [];
  document.getElementById('drawer-fails').innerHTML = fails.length === 0
    ? '<p style="color:var(--green);margin-top:12px">All trials passed ✓</p>'
    : `<div class="muted" style="margin-bottom:8px">${{fails.length}} failure(s)</div>`
      + fails.map(f => `
        <div class="fail-item">
          <div class="fi-trial">Trial ${{f.trial}} · score=${{f.score.toFixed(3)}}</div>
          <div>${{f.answer_preview || '(no answer)'}}</div>
          <div class="fi-exp">${{f.explanation}}</div>
        </div>`).join('');
  document.getElementById('drawer').classList.add('open');
}}
function closeDrawer() {{ document.getElementById('drawer').classList.remove('open'); }}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    path = Path(args.summary) if args.summary else get_latest_summary_filename()
    run_name = path.stem.replace("_summary", "")

    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    html = build_html(summary, run_name)

    out = REPORTS_DIR / f"{run_name}_dashboard.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard → {out}")
    return str(out)


if __name__ == "__main__":
    main()
