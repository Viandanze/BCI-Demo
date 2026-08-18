"""
Visual Feedback - Web-based real-time display for BCI-LLM collaborative reasoning.

Architecture:
  Flask + Server-Sent Events (SSE)
  - Main pipeline pushes state updates via update_state()
  - Browser receives updates via /events SSE stream
  - EEG waveform rendered on canvas with rolling buffer
  - LLM candidates displayed as selectable cards
  - State machine status shown with color-coded indicators

No external JS dependencies — pure vanilla JavaScript + CSS.
Self-contained: HTML template embedded in Python string.
"""

import json
import time
import threading
from queue import Queue, Empty, Full
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from flask import Flask, Response, render_template_string
except ImportError:
    Flask = None
    logger.warning("Flask not installed. Install with: pip install flask")


class VisualFeedback:
    """Web-based visual feedback server for BCI-LLM interaction."""

    HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEURODECODE</title>
<style>
  :root {
    --bg: #08080a;
    --bg-card: #0e0e10;
    --bg-deep: #050506;
    --yellow: #FCEE0A;
    --yellow-dim: rgba(252, 238, 10, 0.6);
    --yellow-faint: rgba(252, 238, 10, 0.1);
    --cyan: #00F0FF;
    --cyan-dim: rgba(0, 240, 255, 0.5);
    --red: #FF003C;
    --text: #d0d0d4;
    --text-dim: #555;
    --text-mute: #2a2a2e;
    --border: rgba(252, 238, 10, 0.08);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  html, body { height: 100%; }

  body {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.6;
    overflow: hidden;
  }

  /* === Atmosphere === */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent 0px, transparent 3px,
      rgba(255, 255, 255, 0.008) 3px, rgba(255, 255, 255, 0.008) 4px
    );
    pointer-events: none; z-index: 999;
  }
  body::after {
    content: '';
    position: fixed; inset: 0;
    background:
      radial-gradient(ellipse 70% 60% at 15% 10%, rgba(252, 238, 10, 0.04) 0%, transparent 60%),
      radial-gradient(ellipse 50% 50% at 85% 90%, rgba(0, 240, 255, 0.03) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
  }

  .app { position: relative; z-index: 1; display: flex; flex-direction: column; height: 100vh; }

  /* === Header === */
  header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0 24px; height: 52px; flex-shrink: 0;
    background: linear-gradient(180deg, #0c0c0e 0%, #08080a 100%);
    border-bottom: 1px solid var(--yellow-faint);
  }
  .logo { display: flex; align-items: center; gap: 12px; }
  .logo-mark {
    width: 20px; height: 20px;
    background: linear-gradient(135deg, var(--yellow), #FF8C00);
    clip-path: polygon(0 0, 70% 0, 100% 30%, 100% 100%, 30% 100%, 0 70%);
    box-shadow: 0 0 15px rgba(252, 238, 10, 0.4);
  }
  .logo-text {
    font-size: 16px; font-weight: 700; letter-spacing: 3px;
    color: var(--yellow);
    text-shadow: 0 0 12px rgba(252, 238, 10, 0.3);
  }
  .logo-sub { font-size: 9px; color: var(--text-dim); letter-spacing: 4px; margin-top: -2px; }

  .header-right { display: flex; align-items: center; gap: 20px; }
  .header-meta { display: flex; gap: 16px; font-size: 10px; color: var(--text-dim); letter-spacing: 1px; }
  .header-meta b { color: var(--text); font-weight: 600; }

  .status-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 14px; font-size: 10px; font-weight: 700; letter-spacing: 2px;
    clip-path: polygon(0 0, calc(100% - 6px) 0, 100% 6px, 100% 100%, 6px 100%, 0 calc(100% - 6px));
    transition: all 0.3s;
  }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .st-idle { color: var(--text-dim); border: 1px solid var(--text-mute); }
  .st-idle .status-dot { background: var(--text-dim); }
  .st-detecting { color: var(--cyan); border: 1px solid rgba(0,240,255,0.3); background: rgba(0,240,255,0.05); }
  .st-detecting .status-dot { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
  .st-intent_locked { color: var(--yellow); border: 1px solid rgba(252,238,10,0.4); background: rgba(252,238,10,0.06); }
  .st-intent_locked .status-dot { background: var(--yellow); box-shadow: 0 0 8px var(--yellow); }
  .st-awaiting_llm { color: var(--yellow-dim); border: 1px solid var(--yellow-faint); background: rgba(252,238,10,0.03); }
  .st-awaiting_llm .status-dot { background: var(--yellow-dim); }
  .st-presenting_candidates { color: var(--yellow); border: 1px solid rgba(252,238,10,0.3); background: rgba(252,238,10,0.04); }
  .st-presenting_candidates .status-dot { background: var(--yellow); box-shadow: 0 0 8px var(--yellow); }
  .st-selecting { color: var(--cyan); border: 1px solid rgba(0,240,255,0.3); background: rgba(0,240,255,0.05); }
  .st-selecting .status-dot { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
  .st-completed { color: var(--cyan); border: 1px solid rgba(0,240,255,0.2); }
  .st-completed .status-dot { background: var(--cyan); }

  /* === Main layout: 2 columns === */
  .main {
    flex: 1; display: grid;
    grid-template-columns: 1fr 300px;
    gap: 12px; padding: 12px;
    overflow: hidden;
  }

  .left-col { display: flex; flex-direction: column; gap: 12px; overflow: hidden; }
  .right-col { display: flex; flex-direction: column; gap: 12px; overflow: hidden; }

  /* === Card base === */
  .card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.015) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 16px 18px;
    position: relative;
    backdrop-filter: blur(10px);
  }
  .card-title {
    font-size: 10px; text-transform: uppercase; letter-spacing: 3px; font-weight: 700;
    color: var(--yellow-dim);
    margin-bottom: 12px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .card-badge {
    font-size: 8px; padding: 2px 8px; color: var(--text-dim);
    border: 1px solid var(--text-mute); letter-spacing: 1px; font-weight: 400;
  }

  /* === EEG Monitor === */
  .eeg-card { flex: 0 0 auto; }
  #eeg-canvas {
    width: 100%; height: 320px; display: block;
    background: #0c0c10;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 2px;
  }
  .eeg-footer {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 8px; font-size: 10px; color: var(--text-dim); letter-spacing: 1px;
  }
  .eeg-channels { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
  .ch-tag { font-size: 9px; padding: 1px 6px; font-weight: 600; border: 1px solid; letter-spacing: 1px; }

  /* === Intent Decoder === */
  .intent-card { flex: 0 0 auto; }
  .intent-display {
    display: flex; align-items: center; gap: 16px;
    padding: 14px 18px;
    background: linear-gradient(135deg, #0a0a0c 0%, #060608 100%);
    border-left: 3px solid var(--yellow);
    margin-bottom: 10px;
  }
  .intent-icon { font-size: 24px; color: var(--yellow); opacity: 0.5; }
  .intent-text { font-size: 24px; font-weight: 700; letter-spacing: 2px; color: var(--text); }
  .intent-text.active { color: var(--yellow); text-shadow: 0 0 20px rgba(252, 238, 10, 0.3); }
  .intent-desc { font-size: 11px; color: var(--text-dim); margin-left: auto; text-align: right; }

  .intent-data { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 10px; }
  .data-item { }
  .data-label { font-size: 9px; color: var(--text-mute); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 3px; }
  .data-value { font-size: 13px; font-weight: 600; color: var(--text); }
  .data-value.yellow { color: var(--yellow); }
  .data-value.cyan { color: var(--cyan); }

  .conf-section { margin-top: 8px; }
  .conf-label { display: flex; justify-content: space-between; font-size: 9px; color: var(--text-dim); letter-spacing: 1px; margin-bottom: 4px; }
  .conf-bar { height: 4px; background: var(--bg-deep); overflow: hidden; }
  .conf-fill { height: 100%; background: linear-gradient(90deg, #FF8C00, var(--yellow)); transition: width 0.3s; width: 0%; box-shadow: 0 0 8px rgba(252, 238, 10, 0.4); }

  .prob-row { display: flex; gap: 3px; margin-top: 6px; height: 3px; }
  .prob-seg { transition: width 0.3s; min-width: 0; }
  .prob-labels { display: flex; gap: 3px; margin-top: 4px; font-size: 8px; color: var(--text-dim); letter-spacing: 1px; }
  .prob-labels span { flex: 1; text-align: center; }

  /* === Pipeline visualizer === */
  .pipeline { display: flex; align-items: center; gap: 4px; margin: 10px 0; }
  .pipe-node {
    flex: 1; text-align: center; padding: 5px 2px;
    font-size: 8px; letter-spacing: 1px; text-transform: uppercase;
    border: 1px solid var(--text-mute); color: var(--text-dim);
    transition: all 0.3s;
  }
  .pipe-node.active { color: var(--yellow); border-color: rgba(252,238,10,0.4); background: rgba(252,238,10,0.06); }
  .pipe-node.done { color: var(--cyan-dim); border-color: rgba(0,240,255,0.15); }
  .pipe-arrow { color: var(--text-mute); font-size: 8px; }

  /* === Candidates === */
  .cand-card { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
  .cand-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; padding-right: 4px; }
  .cand-list::-webkit-scrollbar { width: 3px; }
  .cand-list::-webkit-scrollbar-track { background: transparent; }
  .cand-list::-webkit-scrollbar-thumb { background: var(--yellow-faint); }

  .candidate {
    background: linear-gradient(135deg, #0a0a0c 0%, #08080a 100%);
    border: 1px solid var(--border);
    border-left: 2px solid transparent;
    padding: 12px 14px; cursor: pointer; transition: all 0.2s;
  }
  .candidate:hover {
    border-color: rgba(252, 238, 10, 0.2);
    border-left-color: var(--yellow);
    background: linear-gradient(135deg, rgba(252,238,10,0.04) 0%, #08080a 100%);
  }
  .candidate.selected {
    border-color: rgba(0, 240, 255, 0.3);
    border-left-color: var(--cyan);
    background: linear-gradient(135deg, rgba(0,240,255,0.05) 0%, #08080a 100%);
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.05);
  }
  .candidate.auto-selected {
    border-color: rgba(252, 238, 10, 0.2);
    border-left-color: var(--yellow);
    background: linear-gradient(135deg, rgba(252,238,10,0.04) 0%, #08080a 100%);
  }
  .cand-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .cand-num { font-size: 10px; font-weight: 700; color: var(--yellow); }
  .cand-source { font-size: 9px; color: var(--text-dim); letter-spacing: 1px; }
  .cand-text { font-size: 11px; line-height: 1.6; color: var(--text); word-break: break-word; }

  /* === Right sidebar === */
  .metric-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; font-size: 11px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.02);
  }
  .metric-row:last-child { border-bottom: none; }
  .metric-label { color: var(--text-dim); letter-spacing: 1px; }
  .metric-value { color: var(--text); font-weight: 600; }
  .metric-value.yellow { color: var(--yellow); }
  .metric-value.cyan { color: var(--cyan); }

  .stat-block { display: flex; justify-content: space-between; align-items: baseline; padding: 6px 0; }
  .stat-label { font-size: 9px; color: var(--text-mute); letter-spacing: 2px; text-transform: uppercase; }
  .stat-val { font-size: 16px; font-weight: 700; color: var(--yellow); }
  .stat-val.cyan { color: var(--cyan); }

  .data-tags { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 6px; }
  .data-tag { font-size: 8px; padding: 2px 5px; font-weight: 600; border: 1px solid; letter-spacing: 1px; }

  .history-card { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
  .history-list { flex: 1; overflow-y: auto; }
  .history-list::-webkit-scrollbar { width: 3px; }
  .history-list::-webkit-scrollbar-track { background: transparent; }
  .history-list::-webkit-scrollbar-thumb { background: var(--yellow-faint); }
  .history-item {
    padding: 8px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.02);
    font-size: 10px; display: flex; gap: 8px;
  }
  .history-item:last-child { border-bottom: none; }
  .history-meta { min-width: 60px; flex-shrink: 0; }
  .history-mode { font-size: 8px; font-weight: 700; color: var(--yellow-dim); letter-spacing: 1px; }
  .history-time { font-size: 8px; color: var(--text-mute); }
  .history-response { color: var(--text-dim); line-height: 1.4; word-break: break-word; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }

  .empty { color: var(--text-mute); text-align: center; padding: 20px 0; font-size: 11px; letter-spacing: 1px; }
  .loading::after { content: '...'; animation: dots 1.2s steps(4) infinite; }
  @keyframes dots { 0%,20% { content: ''; } 40% { content: '.'; } 60% { content: '..'; } 80%,100% { content: '...'; } }

  /* === Footer === */
  footer {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0 24px; height: 28px; flex-shrink: 0;
    background: #060608; border-top: 1px solid var(--border);
    font-size: 9px; color: var(--text-dim); letter-spacing: 1px;
  }
  .blink { animation: blink 1.5s infinite; color: var(--yellow); }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>
<div class="app">

  <header>
    <div class="logo">
      <div class="logo-mark"></div>
      <div>
        <div class="logo-text">NEURODECODE</div>
        <div class="logo-sub">BCI / LLM COLLABORATIVE REASONING</div>
      </div>
    </div>
    <div class="header-right">
      <div class="header-meta">
        <span>BOARD: <b id="board-type">Synthetic</b></span>
        <span>CH: <b id="ch-count">16</b></span>
        <span>FS: <b>250Hz</b></span>
      </div>
      <div id="status-pill" class="status-pill st-idle">
        <span class="status-dot"></span>
        <span id="status-text">IDLE</span>
      </div>
    </div>
  </header>

  <div class="main">

    <!-- ===== LEFT COLUMN ===== -->
    <div class="left-col">

      <div class="card eeg-card">
        <div class="card-title">
          <span>EEG Signal Monitor</span>
          <span class="card-badge" id="latency">&mdash; ms</span>
        </div>
        <canvas id="eeg-canvas"></canvas>
        <div class="eeg-footer">
          <span id="eeg-info">BrainFlow Synthetic Board</span>
          <span id="eeg-rate">250 Hz | 8ch | 2.0s rolling</span>
        </div>
        <div class="eeg-channels" id="eeg-channels"></div>
      </div>

      <div class="card intent-card">
        <div class="card-title">
          <span>Intent Decoder</span>
          <span class="card-badge" id="intent-badge">EEGNet v1</span>
        </div>
        <div class="intent-display">
          <span id="intent-icon" class="intent-icon" style="opacity:0.2"></span>
          <span id="intent-text" class="intent-text" style="opacity:0.3">&mdash;</span>
          <span id="intent-desc" class="intent-desc">Awaiting signal...</span>
        </div>

        <div class="pipeline" id="pipeline">
          <div class="pipe-node" id="pipe-1">ACQUIRE</div>
          <span class="pipe-arrow">/</span>
          <div class="pipe-node" id="pipe-2">PREPROC</div>
          <span class="pipe-arrow">/</span>
          <div class="pipe-node" id="pipe-3">DECODE</div>
          <span class="pipe-arrow">/</span>
          <div class="pipe-node" id="pipe-4">LLM</div>
          <span class="pipe-arrow">/</span>
          <div class="pipe-node" id="pipe-5">OUTPUT</div>
        </div>

        <div class="intent-data">
          <div class="data-item">
            <div class="data-label">Classifier</div>
            <div class="data-value yellow" id="ic-classifier">EEGNet 4-CL MI</div>
          </div>
          <div class="data-item">
            <div class="data-label">Confidence</div>
            <div class="data-value yellow" id="conf-val">&mdash; %</div>
          </div>
          <div class="data-item">
            <div class="data-label">Source</div>
            <div class="data-value cyan" id="ic-source">BrainFlow</div>
          </div>
        </div>

        <div class="conf-section">
          <div class="conf-label"><span>CONFIDENCE</span><span id="conf-pct">&mdash;</span></div>
          <div class="conf-bar"><div id="conf-fill" class="conf-fill"></div></div>
        </div>
        <div class="prob-row" id="prob-bars"></div>
        <div class="prob-labels" id="prob-labels"></div>
      </div>

      <div class="card cand-card">
        <div class="card-title">
          <span>LLM Candidates</span>
          <span class="card-badge" id="cand-count">&mdash;</span>
        </div>
        <div class="cand-list" id="cand-list">
          <div class="empty">// Awaiting intent lock</div>
        </div>
      </div>

    </div>

    <!-- ===== RIGHT COLUMN ===== -->
    <div class="right-col">

      <div class="card">
        <div class="card-title"><span>System Status</span><span class="card-badge">LIVE</span></div>
        <div class="metric-row"><span class="metric-label">UPLINK</span><span class="metric-value cyan">ACTIVE</span></div>
        <div class="metric-row"><span class="metric-label">ENCRYPT</span><span class="metric-value">AES-256</span></div>
        <div class="metric-row"><span class="metric-label">LATENCY</span><span class="metric-value cyan" id="sys-latency">&mdash; ms</span></div>
        <div class="metric-row"><span class="metric-label">PACKETS</span><span class="metric-value yellow" id="sys-packets">0</span></div>
        <div class="metric-row"><span class="metric-label">SIGNAL</span><span class="metric-value">STABLE</span></div>
      </div>

      <div class="card">
        <div class="card-title"><span>Board Config</span><span class="card-badge">BF-4.9</span></div>
        <div class="metric-row"><span class="metric-label">BOARD</span><span class="metric-value" id="sb-board">Synthetic</span></div>
        <div class="metric-row"><span class="metric-label">CHANNELS</span><span class="metric-value cyan" id="sb-channels">16</span></div>
        <div class="metric-row"><span class="metric-label">SAMPLE RATE</span><span class="metric-value">250 Hz</span></div>
        <div class="metric-row"><span class="metric-label">WINDOW</span><span class="metric-value">4.0s</span></div>
        <div class="data-tags">
          <span class="data-tag" style="color:var(--cyan);border-color:rgba(0,240,255,0.2)">Fz</span>
          <span class="data-tag" style="color:var(--cyan);border-color:rgba(0,240,255,0.2)">Cz</span>
          <span class="data-tag" style="color:var(--cyan);border-color:rgba(0,240,255,0.2)">Pz</span>
          <span class="data-tag" style="color:var(--cyan);border-color:rgba(0,240,255,0.2)">Oz</span>
          <span class="data-tag" style="color:var(--yellow);border-color:rgba(252,238,10,0.2)">C3</span>
          <span class="data-tag" style="color:var(--yellow);border-color:rgba(252,238,10,0.2)">C4</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title"><span>Pipeline Stats</span><span class="card-badge" id="pipe-mode">MOCK</span></div>
        <div class="stat-block"><span class="stat-label">TURNS</span><span class="stat-val" id="stat-turns">0</span></div>
        <div class="stat-block"><span class="stat-label">AVG CONF</span><span class="stat-val cyan" id="stat-conf">&mdash;%</span></div>
        <div class="stat-block"><span class="stat-label">UPTIME</span><span class="stat-val" id="stat-uptime">00:00</span></div>
      </div>

      <div class="card history-card">
        <div class="card-title"><span>Interaction Log</span><span class="card-badge" id="history-count">0</span></div>
        <div class="history-list" id="history-list">
          <div class="empty">// No interactions recorded</div>
        </div>
      </div>

    </div>

  </div>

  <footer>
    <span>NeuroDecode Phase 1 // <span id="sb-mode">Mock Mode</span></span>
    <span id="uptime">SESSION 00:00:00</span>
  </footer>

</div>

<script>
const canvas = document.getElementById('eeg-canvas');
const ctx = canvas.getContext('2d');
let eegBuffer = [];
const MAX_BUFFER = 500;
const CHANNELS = 8;
const CH_COLORS = ['#00F0FF','#FCEE0A','#FF1F8F','#00F0FF','#FCEE0A','#FF1F8F','#00F0FF','#FCEE0A'];
const CH_NAMES = ['Fz','Cz','Pz','Oz','C3','C4','T7','T8'];
const MI_LABELS = ['Left Hand','Right Hand','Feet','Tongue'];
const MI_SHORT = ['LH','RH','FT','TG'];
const MI_COLORS = ['#00F0FF','#FF1F8F','#FCEE0A','#FF003C'];

document.getElementById('eeg-channels').innerHTML = CH_NAMES.map((n, i) =>
  '<span class="ch-tag" style="color:' + CH_COLORS[i] + ';border-color:' + CH_COLORS[i] + '20">' + n + '</span>'
).join('');
document.getElementById('prob-labels').innerHTML = MI_LABELS.map(l => '<span>' + l + '</span>').join('');

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  let w = rect.width || canvas.offsetWidth || (canvas.parentElement ? canvas.parentElement.offsetWidth : 0) || 800;
  let h = rect.height || 320;
  if (w < 10) w = 800;
  if (h < 10) h = 320;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
}

// Continuous render loop - always redraw even without new data
function renderLoop() {
  resizeCanvas();
  drawEEG(eegBuffer);
  requestAnimationFrame(renderLoop);
}
window.addEventListener('resize', resizeCanvas);
requestAnimationFrame(renderLoop);

function drawEEG(data) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  if (w < 10 || h < 10) return;
  ctx.clearRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= CHANNELS; i++) {
    const y = (h / CHANNELS) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  for (let i = 0; i <= 12; i++) {
    const x = (w / 12) * i;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }

  if (!data || data.length === 0) {
    // Draw channel labels even without data
    const chH = h / CHANNELS;
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    for (let ch = 0; ch < CHANNELS; ch++) {
      ctx.fillText(CH_NAMES[ch], 4, chH * ch + chH / 2 + 4);
    }
    return;
  }

  const chH = h / CHANNELS;
  const labelW = 28;
  const stepX = (w - labelW) / Math.max(data.length, 1);

  for (let ch = 0; ch < CHANNELS && ch < data[0].length; ch++) {
    const baseY = chH * ch + chH / 2;
    const amp = chH * 0.38;

    // Channel label
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = CH_COLORS[ch] + '99';
    ctx.fillText(CH_NAMES[ch], 4, baseY + 4);

    // Waveform
    ctx.strokeStyle = CH_COLORS[ch];
    ctx.lineWidth = 1.8;
    ctx.shadowColor = CH_COLORS[ch];
    ctx.shadowBlur = 10;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = labelW + i * stepX;
      const val = data[i][ch] || 0;
      const y = baseY - val * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

const stateClasses = { idle:'st-idle', detecting:'st-detecting', intent_locked:'st-intent_locked', awaiting_llm:'st-awaiting_llm', presenting_candidates:'st-presenting_candidates', selecting:'st-selecting', completed:'st-completed' };
const stateLabels = { idle:'IDLE', detecting:'DETECTING', intent_locked:'INTENT LOCKED', awaiting_llm:'GENERATING', presenting_candidates:'SELECT CANDIDATE', selecting:'SELECTING', completed:'COMPLETED' };
const pipeMap = { idle:[], detecting:[1], intent_locked:[1,2,3], awaiting_llm:[1,2,3], presenting_candidates:[1,2,3,4], selecting:[1,2,3,4], completed:[1,2,3,4,5] };

function updatePipeline(state) {
  const active = pipeMap[state] || [];
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('pipe-' + i);
    if (!el) continue;
    el.classList.remove('active', 'done');
    if (active.includes(i)) {
      if (i < active[active.length - 1]) el.classList.add('done');
      else el.classList.add('active');
    }
  }
}

function updateState(data) {
  const pill = document.getElementById('status-pill');
  const text = document.getElementById('status-text');
  pill.className = 'status-pill ' + (stateClasses[data.state] || 'st-idle');
  text.textContent = stateLabels[data.state] || data.state.toUpperCase();
  updatePipeline(data.state);

  const intentText = document.getElementById('intent-text');
  const intentIcon = document.getElementById('intent-icon');
  const intentDesc = document.getElementById('intent-desc');
  const fill = document.getElementById('conf-fill');
  const val = document.getElementById('conf-val');
  const pct = document.getElementById('conf-pct');

  if (data.current_intent) {
    intentText.textContent = data.current_intent;
    intentText.classList.add('active');
    intentText.style.opacity = '1';
    intentIcon.style.opacity = '0.8';
    intentIcon.textContent = '';
    intentDesc.textContent = data.current_mode || '';
    if (data.current_confidence !== null && data.current_confidence !== undefined) {
      const p = data.current_confidence * 100;
      fill.style.width = p + '%';
      val.textContent = p.toFixed(1) + '%';
      pct.textContent = p.toFixed(0) + '%';
    }
  } else if (data.state === 'idle') {
    intentText.textContent = '\u2014';
    intentText.classList.remove('active');
    intentText.style.opacity = '0.3';
    intentIcon.style.opacity = '0.2';
    intentIcon.textContent = '';
    intentDesc.textContent = 'Awaiting signal...';
    fill.style.width = '0%';
    val.textContent = '\u2014 %';
    pct.textContent = '\u2014';
  }

  const probBars = document.getElementById('prob-bars');
  if (data.current_probabilities && data.current_probabilities.length > 0 && data.state !== 'idle') {
    const probTotal = data.current_probabilities.reduce((s, p) => s + p, 0) || 1;
    probBars.innerHTML = MI_COLORS.map((c, i) => {
      const w = ((data.current_probabilities[i] || 0) / probTotal) * 100;
      return '<div class="prob-seg" style="width:' + w + '%;background:' + c + ';box-shadow:0 0 3px ' + c + '"></div>';
    }).join('');
  }

  const list = document.getElementById('cand-list');
  const countBadge = document.getElementById('cand-count');
  if (data.candidates && data.candidates.length > 0) {
    countBadge.textContent = data.candidates.length + ' options';
    list.innerHTML = data.candidates.map((c, i) => {
      const hint = MI_LABELS[i] || 'Option ' + (i+1);
      const short = MI_SHORT[i] || ('O' + (i+1));
      const color = MI_COLORS[i] || '#FCEE0A';
      return '<div class="candidate" data-index="' + i + '">' +
        '<div class="cand-header"><span class="cand-num">[' + (i + 1) + ']</span>' +
        '<span class="cand-source" style="color:' + color + '">BCI: ' + short + ' ' + hint + '</span></div>' +
        '<div class="cand-text">' + c + '</div></div>';
    }).join('');
  } else if (data.state === 'awaiting_llm') {
    countBadge.textContent = '\u2014';
    list.innerHTML = '<div class="empty">// Generating<span class="loading"></span></div>';
  } else if (data.state === 'idle') {
    countBadge.textContent = '\u2014';
    list.innerHTML = '<div class="empty">// Awaiting intent lock</div>';
  }
}

function updateHistory(turns) {
  const list = document.getElementById('history-list');
  const countEl = document.getElementById('history-count');
  if (!turns || turns.length === 0) {
    list.innerHTML = '<div class="empty">// No interactions recorded</div>';
    countEl.textContent = '0';
    return;
  }
  countEl.textContent = turns.length;
  document.getElementById('stat-turns').textContent = turns.length;
  const avgConf = turns.reduce((s, t) => s + (t.intent_confidence || 0), 0) / turns.length;
  document.getElementById('stat-conf').textContent = (avgConf * 100).toFixed(0) + '%';

  list.innerHTML = turns.map(t =>
    '<div class="history-item"><div class="history-meta">' +
    '<div class="history-mode">' + (t.intent_label || t.intent_mode || '\u2014') + '</div>' +
    '<div class="history-time">' + ((t.intent_confidence * 100).toFixed(0)) + '% | ' + ((t.duration || 0).toFixed(1)) + 's</div>' +
    '</div><div class="history-response">' + (t.final_response || '\u2014') + '</div></div>'
  ).join('');
  list.scrollTop = list.scrollHeight;
}

const evtSource = new EventSource('/events');
let lastEEGTime = 0, packetCount = 0;
const sessionStart = Date.now();
let lastStateSnapshot = '';

evtSource.onerror = function(e) {
  // EventSource auto-reconnects; just log silently
};

evtSource.onmessage = function(e) {
  const data = JSON.parse(e.data);
  if (data.type === 'eeg_batch') {
    eegBuffer = data.data;
    packetCount++;
    if (packetCount % 5 === 0) {
      const now = Date.now();
      if (lastEEGTime > 0) {
        const ms = ((now - lastEEGTime) / 5).toFixed(0);
        document.getElementById('latency').textContent = ms + ' ms';
        document.getElementById('sys-latency').textContent = ms + ' ms';
      }
      lastEEGTime = now;
    }
    if (packetCount % 10 === 0) document.getElementById('sys-packets').textContent = packetCount.toLocaleString();
  } else if (data.type === 'state') {
    const snapshot = JSON.stringify(data);
    if (snapshot !== lastStateSnapshot) {
      lastStateSnapshot = snapshot;
      updateState(data);
    }
  }
  else if (data.type === 'history') { updateHistory(data.turns); }
  else if (data.type === 'selection') {
    document.querySelectorAll('.candidate').forEach((card, i) => {
      if (i === data.index) card.classList.add(data.auto ? 'auto-selected' : 'selected');
    });
  }
};

setInterval(function() {
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
  const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  document.getElementById('uptime').textContent = 'SESSION ' + h + ':' + m + ':' + s;
  document.getElementById('stat-uptime').textContent = m + ':' + s;
}, 1000);
</script>
</body>
</html>"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        if Flask is None:
            raise ImportError(
                "Flask is required for visual feedback. "
                "Install with: pip install flask"
            )
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._eeg_queue: Queue = Queue(maxsize=50)
        self._state_queue: Queue = Queue(maxsize=50)
        self._latest_state: dict = {"type": "state", "state": "idle"}
        self._history: list = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template_string(self.HTML_TEMPLATE)

        @self.app.route("/events")
        def events():
            def event_stream():
                yield "retry: 5000\n\n"
                yield f"data: {json.dumps(self._latest_state)}\n\n"
                while self._running:
                    # State events have priority over EEG data
                    try:
                        data = self._state_queue.get(timeout=0.05)
                        yield f"data: {json.dumps(data)}\n\n"
                        continue
                    except Empty:
                        pass
                    try:
                        data = self._eeg_queue.get(timeout=0.95)
                        yield f"data: {json.dumps(data)}\n\n"
                    except Empty:
                        yield ": keepalive\n\n"
            return Response(
                event_stream(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    def update_eeg(self, eeg_batch: list) -> bool:
        """Push a batch of EEG samples to the stream queue.

        Uses a rolling-window model: each call delivers a complete window
        of recent EEG data (2D array of shape [n_samples][n_channels]).
        The frontend replaces its entire buffer with each batch, ensuring
        the waveform display is always full from the first push.

        Args:
            eeg_batch: 2D list of shape [n_samples][n_channels].

        Returns:
            True if successfully queued, False if dropped due to full queue.
        """
        try:
            self._eeg_queue.put_nowait({"type": "eeg_batch", "data": eeg_batch})
            return True
        except Full:
            return False

    def update_state(
        self,
        state: str,
        intent: Optional[dict] = None,
        candidates: Optional[list] = None,
    ):
        event = {
            "type": "state",
            "state": state,
            "current_intent": intent.get("mode_label") if intent else None,
            "current_mode": intent.get("description") if intent else None,
            "current_confidence": intent.get("confidence") if intent else None,
            "current_probabilities": intent.get("raw_probabilities") if intent else None,
            "candidates": candidates or [],
        }
        with self._lock:
            self._latest_state = event
        try:
            self._state_queue.put_nowait(event)
        except Full:
            pass

    def update_selection(self, index: int, auto: bool = False):
        event = {"type": "selection", "index": index, "auto": auto}
        try:
            self._state_queue.put_nowait(event)
        except Full:
            pass

    def update_history(self, turns: list):
        event = {"type": "history", "turns": turns}
        try:
            self._state_queue.put_nowait(event)
        except Full:
            pass

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=lambda: self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True,
            ),
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Visual feedback server started at http://{self.host}:{self.port}")

    def stop(self):
        self._running = False
        logger.info("Visual feedback server stopped")
