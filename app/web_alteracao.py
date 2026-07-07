from __future__ import annotations

import cgi
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import ollama

try:
    from local_forensics import (
        ForensicResult,
        analyze_forensics,
        build_calibration_samples,
        compare_with_original,
        comparison_feature_profile,
        feature_profile,
        file_sha256 as forensic_sha256,
    )
except ModuleNotFoundError:  # pragma: no cover - supports python -m app.web_alteracao
    from app.local_forensics import (
        ForensicResult,
        analyze_forensics,
        build_calibration_samples,
        compare_with_original,
        comparison_feature_profile,
        feature_profile,
        file_sha256 as forensic_sha256,
    )

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - fallback for minimal installs
    Image = None
    ImageOps = None


PORT = int(os.environ.get("WEB_PORT", "9090"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11435")
MODEL = os.environ.get("VISION_MODEL", "codex-dental:latest")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
ANALYSIS_DIR = RUNTIME_DIR / "analysis_cache"
HISTORY_FILE = RUNTIME_DIR / "analysis_history.json"
CALIBRATION_FILE = RUNTIME_DIR / "integrity_calibration.json"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ANALYSIS_MAX_SIDE = int(os.environ.get("ANALYSIS_MAX_SIDE", "896"))
TESTS_DIR = PROJECT_ROOT / "tests"

MODEL_OPTIONS = {
    "temperature": 0.1,
    "num_ctx": 896,
    "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "48")),
    "num_thread": int(os.environ.get("OLLAMA_NUM_THREAD", "0")) or None,
}
MODEL_OPTIONS = {key: value for key, value in MODEL_OPTIONS.items() if value is not None}
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
DEFAULT_ESTIMATED_SECONDS = float(os.environ.get("DEFAULT_ESTIMATED_SECONDS", "8"))
FAST_LOCAL_MODE = os.environ.get("FAST_LOCAL_MODE", "1").lower() not in {"0", "false", "nao", "não"}

HTML = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Perito Visual</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f6f8;
      color: #18202a;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    main {
      width: min(960px, 100%);
      background: #fff;
      border: 1px solid #d9e0e7;
      border-radius: 8px;
      box-shadow: 0 12px 34px rgba(24, 32, 42, 0.08);
      overflow: hidden;
    }
    header {
      padding: 24px;
      border-bottom: 1px solid #e4e9ee;
      background: #fbfcfd;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0;
    }
    p {
      margin: 0;
      color: #52606d;
      line-height: 1.45;
    }
    section {
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .upload {
      border: 1px dashed #9aa8b5;
      border-radius: 8px;
      padding: 22px;
      background: #f8fafc;
      display: grid;
      gap: 14px;
    }
    input[type="file"] {
      width: 100%;
      font-size: 15px;
    }
    .field {
      display: grid;
      gap: 7px;
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 9px;
      color: #334155;
      font-size: 14px;
    }
    .toggle input {
      width: 18px;
      height: 18px;
    }
    .field label {
      font-weight: 800;
      color: #26313d;
    }
    .hint {
      font-size: 13px;
      color: #64748b;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    button {
      width: fit-content;
      border: 0;
      border-radius: 6px;
      padding: 11px 16px;
      background: #1264a3;
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    .actions {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .feedback-button {
      background: #475569;
    }
    .feedback-button.modified {
      background: #b42318;
    }
    .feedback-button.ai {
      background: #7c3aed;
    }
    .feedback-button.real {
      background: #067647;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .preview {
      display: none;
      max-width: 100%;
      max-height: 360px;
      border-radius: 6px;
      border: 1px solid #d9e0e7;
      object-fit: contain;
      background: #101820;
    }
    .forensics {
      display: none;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      border: 1px solid #d9e0e7;
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .metric {
      display: grid;
      gap: 3px;
      min-width: 0;
    }
    .metric span {
      font-size: 12px;
      color: #64748b;
    }
    .metric strong {
      color: #26313d;
      word-break: break-word;
    }
    .result {
      display: none;
      border: 1px solid #d9e0e7;
      border-radius: 8px;
      padding: 18px;
      background: #ffffff;
    }
    .verdict {
      font-size: 22px;
      font-weight: 800;
      margin-bottom: 12px;
    }
    .verdict.sim { color: #b42318; }
    .verdict.nao { color: #067647; }
    .verdict.indeterminado { color: #b54708; }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-family: Consolas, Monaco, monospace;
      font-size: 14px;
      line-height: 1.45;
      color: #26313d;
    }
    .status {
      min-height: 22px;
      color: #52606d;
    }
    .thinking {
      display: none;
      border: 1px solid #d9e0e7;
      border-radius: 8px;
      padding: 14px;
      background: #ffffff;
      gap: 10px;
    }
    .thinking.active {
      display: grid;
    }
    .thinking-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      font-size: 14px;
      color: #334155;
    }
    .percent {
      font-weight: 800;
      color: #1264a3;
      min-width: 44px;
      text-align: right;
    }
    .bar {
      height: 10px;
      background: #e8eef4;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      width: 0%;
      height: 100%;
      background: #1264a3;
      border-radius: inherit;
      transition: width 360ms ease;
    }
    .steps {
      display: grid;
      gap: 6px;
      margin-top: 2px;
      color: #64748b;
      font-size: 13px;
    }
    .step.done {
      color: #067647;
      font-weight: 700;
    }
    .step.active {
      color: #1264a3;
      font-weight: 700;
    }
    .history {
      border-top: 1px solid #e4e9ee;
      background: #fbfcfd;
    }
    .history-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .history-head h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    .history-tabs {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    .history-tab {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: #334155;
      font-weight: 700;
      cursor: pointer;
    }
    .history-tab.active {
      border-color: #1264a3;
      background: #e8f2fb;
      color: #0f4f82;
    }
    .history-tab-count {
      min-width: 22px;
      border-radius: 999px;
      padding: 2px 7px;
      background: #e2e8f0;
      color: #334155;
      font-size: 12px;
      text-align: center;
    }
    .history-tab.active .history-tab-count {
      background: #1264a3;
      color: #fff;
    }
    .history-list {
      display: grid;
      gap: 12px;
    }
    .history-empty {
      color: #64748b;
      border: 1px dashed #cbd5e1;
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }
    .history-item {
      display: grid;
      grid-template-columns: 132px 1fr;
      gap: 14px;
      border: 1px solid #d9e0e7;
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .history-images {
      display: grid;
      gap: 8px;
    }
    .history-item img {
      width: 132px;
      height: 92px;
      object-fit: cover;
      border-radius: 6px;
      border: 1px solid #d9e0e7;
      background: #101820;
    }
    .history-item img.original-thumb {
      height: 64px;
      opacity: 0.86;
    }
    .history-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      color: #64748b;
      font-size: 13px;
      margin-bottom: 8px;
    }
    .history-verdict {
      font-weight: 800;
      margin-bottom: 6px;
    }
    .history-verdict.sim { color: #b42318; }
    .history-verdict.nao { color: #067647; }
    .history-verdict.indeterminado { color: #b54708; }
    .history-report {
      white-space: pre-wrap;
      word-break: break-word;
      color: #26313d;
      font-size: 13px;
      line-height: 1.4;
    }
    @media (max-width: 620px) {
      .history-item {
        grid-template-columns: 1fr;
      }
      .history-item img {
        width: 100%;
        height: auto;
        max-height: 260px;
        object-fit: contain;
      }
      .preview-grid,
      .forensics {
        grid-template-columns: 1fr;
      }
      .history-tabs {
        align-items: stretch;
      }
      .history-tab {
        flex: 1 1 calc(50% - 8px);
        justify-content: center;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Perito Visual</h1>
      <p>Envie uma foto, print ou imagem para verificar sinais de edicao manual, alteracao digital ou geracao/edicao por IA.</p>
    </header>
    <section>
      <form class="upload" id="form">
        <div class="field">
          <label for="image">Imagem suspeita</label>
          <input id="image" name="image" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" required>
          <span class="hint">Foto, print, raio-X ou imagem odontologica que sera analisada.</span>
        </div>
        <div class="field">
          <label for="originalImage">Imagem original opcional</label>
          <input id="originalImage" name="original_image" type="file" accept="image/png,image/jpeg,image/webp,image/bmp">
          <span class="hint">Use quando tiver a foto original para comparar contra a suspeita.</span>
        </div>
        <label class="toggle" for="useLlm">
          <input id="useLlm" name="use_llm" type="checkbox">
          Usar LLM detalhada (mais lento)
        </label>
        <div class="preview-grid">
          <img id="preview" class="preview" alt="Previa da imagem suspeita">
          <img id="originalPreview" class="preview" alt="Previa da imagem original">
        </div>
        <div class="actions">
          <button id="submit" type="submit">Analisar imagem</button>
          <button id="markReal" class="feedback-button real" type="button">Real</button>
          <button id="markModified" class="feedback-button modified" type="button">Modificada</button>
          <button id="markAi" class="feedback-button ai" type="button">IA</button>
        </div>
        <div id="thinking" class="thinking" aria-live="polite">
          <div class="thinking-head">
            <span id="phase">Aguardando envio</span>
            <span id="percent" class="percent">0%</span>
          </div>
          <div class="bar" aria-hidden="true">
            <div id="barFill" class="bar-fill"></div>
          </div>
          <div class="steps">
            <div id="stepUpload" class="step">1. Enviando imagem</div>
            <div id="stepVision" class="step">2. Pericia local e LLM analisando evidencias</div>
            <div id="stepReport" class="step">3. Preparando veredito</div>
          </div>
        </div>
        <div id="status" class="status"></div>
      </form>
      <div id="result" class="result">
        <div id="verdict" class="verdict"></div>
        <div id="forensics" class="forensics">
          <div class="metric"><span>Score local</span><strong id="forensicScore">-</strong></div>
          <div class="metric"><span>Fonte</span><strong id="forensicSource">-</strong></div>
          <div class="metric"><span>Evidencias locais</span><strong id="forensicEvidence">-</strong></div>
        </div>
        <pre id="report"></pre>
      </div>
    </section>
    <section class="history">
      <div class="history-head">
        <h2>Historico de analises</h2>
        <p id="historyCount">0 registros</p>
      </div>
      <div class="history-tabs" id="historyTabs" role="tablist" aria-label="Filtros do historico">
        <button class="history-tab active" type="button" data-tab="all">Todos <span class="history-tab-count">0</span></button>
        <button class="history-tab" type="button" data-tab="modified">Modificados/IA <span class="history-tab-count">0</span></button>
        <button class="history-tab" type="button" data-tab="real">Reais <span class="history-tab-count">0</span></button>
        <button class="history-tab" type="button" data-tab="inconclusive">Inconclusivos <span class="history-tab-count">0</span></button>
        <button class="history-tab" type="button" data-tab="calibration">Calibracao <span class="history-tab-count">0</span></button>
      </div>
      <div id="historyList" class="history-list">
        <div class="history-empty">Nenhuma analise armazenada ainda.</div>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById('form');
    const input = document.getElementById('image');
    const originalInput = document.getElementById('originalImage');
    const useLlmInput = document.getElementById('useLlm');
    const preview = document.getElementById('preview');
    const originalPreview = document.getElementById('originalPreview');
    const submit = document.getElementById('submit');
    const markReal = document.getElementById('markReal');
    const markModified = document.getElementById('markModified');
    const markAi = document.getElementById('markAi');
    const statusBox = document.getElementById('status');
    const thinking = document.getElementById('thinking');
    const phase = document.getElementById('phase');
    const percent = document.getElementById('percent');
    const barFill = document.getElementById('barFill');
    const steps = [
      document.getElementById('stepUpload'),
      document.getElementById('stepVision'),
      document.getElementById('stepReport')
    ];
    const result = document.getElementById('result');
    const verdict = document.getElementById('verdict');
    const report = document.getElementById('report');
    const forensics = document.getElementById('forensics');
    const forensicScore = document.getElementById('forensicScore');
    const forensicSource = document.getElementById('forensicSource');
    const forensicEvidence = document.getElementById('forensicEvidence');
    const historyTabs = document.getElementById('historyTabs');
    const historyList = document.getElementById('historyList');
    const historyCount = document.getElementById('historyCount');
    let progressTimer = null;
    let progressValue = 0;
    let estimatedSeconds = 75;
    let progressStartedAt = 0;
    let historyItems = [];
    let activeHistoryTab = 'all';

    function setProgress(value, label) {
      progressValue = Math.max(progressValue, Math.min(value, 100));
      percent.textContent = `${Math.round(progressValue)}%`;
      barFill.style.width = `${progressValue}%`;
      phase.textContent = label;

      steps.forEach((step) => {
        step.classList.remove('active', 'done');
      });
      if (progressValue < 25) {
        steps[0].classList.add('active');
      } else if (progressValue < 86) {
        steps[0].classList.add('done');
        steps[1].classList.add('active');
      } else if (progressValue < 100) {
        steps[0].classList.add('done');
        steps[1].classList.add('done');
        steps[2].classList.add('active');
      } else {
        steps.forEach((step) => step.classList.add('done'));
      }
    }

    function startThinking() {
      thinking.classList.add('active');
      progressValue = 0;
      progressStartedAt = performance.now();
      setProgress(4, 'Enviando imagem para analise...');
      clearInterval(progressTimer);
      progressTimer = setInterval(() => {
        const elapsedSeconds = (performance.now() - progressStartedAt) / 1000;
        const ratio = Math.min(elapsedSeconds / Math.max(estimatedSeconds, 8), 1);
        let next = 8 + (ratio * 86);
        let label = 'Pericia local analisando evidencias visuais...';

        if (elapsedSeconds < 2) {
          next = Math.max(progressValue, 8);
          label = 'Preparando imagem otimizada...';
        } else if (ratio < 0.72) {
          label = `Pericia local e LLM analisando (${Math.round(elapsedSeconds)}s de ~${Math.round(estimatedSeconds)}s)...`;
        } else if (ratio < 0.96) {
          label = 'Normalizando score, confianca e justificativa...';
        } else {
          next = Math.min(97, next);
          label = 'Finalizando veredito...';
        }

        setProgress(Math.min(next, 97), label);
      }, 1000);
    }

    function finishThinking(label) {
      clearInterval(progressTimer);
      progressTimer = null;
      setProgress(100, label);
    }

    function resetThinking() {
      clearInterval(progressTimer);
      progressTimer = null;
      progressValue = 0;
      thinking.classList.remove('active');
      setProgress(0, 'Aguardando envio');
    }

    function verdictClass(value) {
      const normalized = String(value || '').toLowerCase();
      if (normalized.includes('ia') || normalized.includes('alterada') || normalized.includes('modificado')) return 'sim';
      if (normalized.includes('real')) return 'nao';
      if (normalized.includes('sim')) return 'sim';
      if (normalized.includes('nao') || normalized.includes('não')) return 'nao';
      return 'indeterminado';
    }

    function isModifiedHistory(item) {
      const verdict = String(item.verdict || '').toUpperCase();
      return verdict.includes('MODIFICADO') || verdict.includes('ALTERADA') || verdict.includes('IA');
    }

    function isCalibrationHistory(item) {
      const source = String(item.source || '').toLowerCase();
      const model = String(item.model || '').toLowerCase();
      return source.includes('calibracao') || source.includes('feedback') || model.includes('calibracao');
    }

    function historyTabCounts(items) {
      return {
        all: items.length,
        modified: items.filter(isModifiedHistory).length,
        real: items.filter((item) => String(item.verdict || '').toUpperCase() === 'REAL').length,
        inconclusive: items.filter((item) => String(item.verdict || '').toUpperCase() === 'INDETERMINADO').length,
        calibration: items.filter(isCalibrationHistory).length
      };
    }

    function filteredHistory(items) {
      if (activeHistoryTab === 'modified') return items.filter(isModifiedHistory);
      if (activeHistoryTab === 'real') return items.filter((item) => String(item.verdict || '').toUpperCase() === 'REAL');
      if (activeHistoryTab === 'inconclusive') return items.filter((item) => String(item.verdict || '').toUpperCase() === 'INDETERMINADO');
      if (activeHistoryTab === 'calibration') return items.filter(isCalibrationHistory);
      return items;
    }

    function updateHistoryTabs(items) {
      const counts = historyTabCounts(items);
      historyTabs.querySelectorAll('.history-tab').forEach((tab) => {
        const tabName = tab.dataset.tab;
        tab.classList.toggle('active', tabName === activeHistoryTab);
        const counter = tab.querySelector('.history-tab-count');
        if (counter) counter.textContent = counts[tabName] || 0;
      });
    }

    function emptyHistoryMessage() {
      const labels = {
        all: 'Nenhuma analise armazenada ainda.',
        modified: 'Nenhuma analise modificada ou IA nesta aba.',
        real: 'Nenhuma analise real nesta aba.',
        inconclusive: 'Nenhuma analise inconclusiva nesta aba.',
        calibration: 'Nenhuma calibracao salva nesta aba.'
      };
      return labels[activeHistoryTab] || labels.all;
    }

    function renderHistory(items) {
      historyItems = items;
      updateHistoryTabs(items);
      const visibleItems = filteredHistory(items);
      historyCount.textContent = `${visibleItems.length} de ${items.length} registro${items.length === 1 ? '' : 's'}`;
      if (!visibleItems.length) {
        historyList.innerHTML = `<div class="history-empty">${emptyHistoryMessage()}</div>`;
        return;
      }

      historyList.innerHTML = '';
      for (const item of visibleItems) {
        const card = document.createElement('article');
        card.className = 'history-item';

        const imageBox = document.createElement('div');
        imageBox.className = 'history-images';

        const image = document.createElement('img');
        image.src = item.image_url;
        image.alt = item.filename || 'Imagem analisada';
        image.loading = 'lazy';
        imageBox.appendChild(image);

        if (item.original_image_url) {
          const originalImage = document.createElement('img');
          originalImage.className = 'original-thumb';
          originalImage.src = item.original_image_url;
          originalImage.alt = item.original_filename || 'Imagem original';
          originalImage.loading = 'lazy';
          imageBox.appendChild(originalImage);
        }

        const body = document.createElement('div');
        const meta = document.createElement('div');
        meta.className = 'history-meta';
        const duration = item.duration_seconds ? ` · ${item.duration_seconds}s` : '';
        meta.textContent = `${item.created_at || ''} · ${item.filename || ''}${duration}`;

        const score = item.forensic_score !== undefined && item.forensic_score !== null ? ` - score ${item.forensic_score}` : '';
        const source = item.source ? ` - ${item.source}` : '';
        meta.textContent = `${meta.textContent}${score}${source}`;

        const itemVerdict = document.createElement('div');
        itemVerdict.className = `history-verdict ${verdictClass(item.verdict)}`;
        itemVerdict.textContent = `Veredito: ${item.verdict || 'INDETERMINADO'}`;

        const itemReport = document.createElement('div');
        itemReport.className = 'history-report';
        itemReport.textContent = item.report || '';

        body.append(meta, itemVerdict, itemReport);
        card.append(imageBox, body);
        historyList.appendChild(card);
      }
    }

    historyTabs.addEventListener('click', (event) => {
      const tab = event.target.closest('.history-tab');
      if (!tab) return;
      activeHistoryTab = tab.dataset.tab || 'all';
      renderHistory(historyItems);
    });

    async function calibrateImage(label) {
      const file = input.files[0];
      if (!file) {
        statusBox.textContent = 'Selecione uma imagem antes de marcar como exemplo.';
        return;
      }

      submit.disabled = true;
      markReal.disabled = true;
      markModified.disabled = true;
      markAi.disabled = true;
      result.style.display = 'none';
      forensics.style.display = 'none';
      resetThinking();
      const statusByLabel = {
        REAL: 'Salvando exemplo local como real...',
        MODIFICADO: 'Salvando exemplo local como modificado...',
        IA_GERADA_EDITADA: 'Salvando exemplo local como IA/editada...'
      };
      statusBox.textContent = statusByLabel[label] || 'Salvando exemplo local...';

      const data = new FormData();
      data.append('image', file);
      data.append('label', label);
      if (originalInput.files[0]) {
        data.append('original_image', originalInput.files[0]);
      }

      try {
        const response = await fetch('/calibrate', { method: 'POST', body: data });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Falha ao salvar exemplo.');

        verdict.textContent = `Veredito: ${payload.verdict}`;
        verdict.className = `verdict ${verdictClass(payload.verdict)}`;
        report.textContent = payload.report;
        forensicScore.textContent = payload.forensic_score !== undefined ? `${payload.forensic_score}%` : '-';
        forensicSource.textContent = payload.source || '-';
        forensicEvidence.textContent = (payload.forensic_evidence || []).join('; ') || '-';
        forensics.style.display = 'grid';
        result.style.display = 'block';
        statusBox.textContent = originalInput.files[0]
          ? 'Par salvo. O projeto local vai usar o padrao de diferenca entre suspeita e original nas proximas analises.'
          : 'Exemplo salvo. O projeto local vai usar esse perfil visual para reconhecer padroes parecidos nas proximas analises.';
        await loadHistory();
      } catch (error) {
        statusBox.textContent = error.message;
      } finally {
        submit.disabled = false;
        markReal.disabled = false;
        markModified.disabled = false;
        markAi.disabled = false;
      }
    }

    async function loadHistory() {
      try {
        const response = await fetch('/history');
        if (!response.ok) return;
        const payload = await response.json();
        renderHistory(payload.history || []);
      } catch (_error) {
        historyList.innerHTML = '<div class="history-empty">Nao foi possivel carregar o historico.</div>';
      }
    }

    async function loadMetrics() {
      try {
        const response = await fetch('/metrics');
        if (!response.ok) return;
        const payload = await response.json();
        if (payload.estimated_seconds) {
          estimatedSeconds = Math.max(8, Number(payload.estimated_seconds));
        }
      } catch (_error) {
        estimatedSeconds = 75;
      }
    }

    input.addEventListener('change', () => {
      const file = input.files[0];
      result.style.display = 'none';
      forensics.style.display = 'none';
      resetThinking();
      if (!file) {
        preview.style.display = 'none';
        return;
      }
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
    });

    originalInput.addEventListener('change', () => {
      const file = originalInput.files[0];
      result.style.display = 'none';
      forensics.style.display = 'none';
      resetThinking();
      if (!file) {
        originalPreview.style.display = 'none';
        return;
      }
      originalPreview.src = URL.createObjectURL(file);
      originalPreview.style.display = 'block';
    });

    markReal.addEventListener('click', () => calibrateImage('REAL'));
    markModified.addEventListener('click', () => calibrateImage('MODIFICADO'));
    markAi.addEventListener('click', () => calibrateImage('IA_GERADA_EDITADA'));

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const file = input.files[0];
      if (!file) return;

      submit.disabled = true;
      markReal.disabled = true;
      markModified.disabled = true;
      markAi.disabled = true;
      statusBox.textContent = useLlmInput.checked
        ? 'Analisando com pericia local e LLM local...'
        : 'Analisando em modo rapido local...';
      result.style.display = 'none';
      forensics.style.display = 'none';
      await loadMetrics();
      startThinking();

      const data = new FormData();
      data.append('image', file);
      if (originalInput.files[0]) {
        data.append('original_image', originalInput.files[0]);
      }
      if (useLlmInput.checked) {
        data.append('use_llm', '1');
      }

      try {
        const response = await fetch('/analyze', { method: 'POST', body: data });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Falha na analise.');

        verdict.textContent = `Veredito: ${payload.verdict}`;
        verdict.className = `verdict ${verdictClass(payload.verdict)}`;
        report.textContent = payload.report;
        forensicScore.textContent = payload.forensic_score !== undefined ? `${payload.forensic_score}%` : '-';
        forensicSource.textContent = payload.source || '-';
        forensicEvidence.textContent = (payload.forensic_evidence || []).join('; ') || '-';
        forensics.style.display = 'grid';
        result.style.display = 'block';
        finishThinking('Resposta pronta.');
        statusBox.textContent = payload.duration_seconds
          ? `Analise concluida em ${payload.duration_seconds}s.`
          : 'Analise concluida.';
        await loadHistory();
      } catch (error) {
        clearInterval(progressTimer);
        statusBox.textContent = error.message;
      } finally {
        submit.disabled = false;
        markReal.disabled = false;
        markModified.disabled = false;
        markAi.disabled = false;
      }
    });

    loadHistory();
    loadMetrics();
  </script>
</body>
</html>
"""


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def clean_model_response(text: object) -> str:
    value = str(text)
    if "<unused95>" in value:
        value = value.rsplit("<unused95>", 1)[-1]
    value = re.sub(r"<unused94>\s*thought.*?<unused95>", "", value, flags=re.DOTALL)
    value = value.replace("<unused94>", "").replace("<unused95>", "")
    value = re.sub(r"(?is)\n?\s*thought\s*\n.*$", "", value)
    value = re.sub(r"(?is)^.*?(VEREDITO\s*:)", r"\1", value)
    allowed_lines = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(
            r"^(VEREDITO|TIPO|CONFIANCA|SCORE|SCORE_ALTERACAO|JUSTIFICATIVA|EVIDENCIAS)\s*:",
            stripped,
            re.I,
        ):
            allowed_lines.append(stripped)
    return "\n".join(allowed_lines).strip() or value.strip()


def infer_verdict(report: str) -> str:
    normalized = report.upper()
    label_match = re.search(
        r"\bVEREDITO\s*:\s*(REAL|MODIFICADO|ALTERADA[_ ]MANUALMENTE|ALTERADA[_ ]DIGITALMENTE|IA(?:[_ ]GERADA[_/]?EDITADA)?|INDETERMINADO)\b",
        normalized,
    )
    if label_match:
        value = label_match.group(1).replace(" ", "_")
        return "IA_GERADA_EDITADA" if value == "IA" else value
    match = re.search(r"\bVEREDITO\s*:\s*(SIM|NAO|NÃO|INDETERMINADO)\b", normalized)
    if match:
        value = match.group(1)
        return "NAO" if value == "NÃO" else value
    if re.search(r"\bSIM\b", normalized[:160]):
        return "SIM"
    if re.search(r"\bNAO\b|\bNÃO\b", normalized[:160]):
        return "NAO"
    return "INDETERMINADO"


def canonical_integrity_label(value: str) -> str:
    normalized = value.upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "NAO": "REAL",
        "NÃO": "REAL",
        "SIM": "MODIFICADO",
        "ALTERADA": "MODIFICADO",
        "ALTERADA_MANUALMENTE": "MODIFICADO",
        "ALTERADA_DIGITALMENTE": "MODIFICADO",
        "IA": "IA_GERADA_EDITADA",
        "IA_GERADA": "IA_GERADA_EDITADA",
        "IA_EDITADA": "IA_GERADA_EDITADA",
        "IA_GERADA/EDITADA": "IA_GERADA_EDITADA",
        "ALTERADA_MANUAL": "MODIFICADO",
        "ALTERADA_DIGITAL": "MODIFICADO",
    }
    normalized = aliases.get(normalized, normalized)
    valid = {
        "REAL",
        "MODIFICADO",
        "ALTERADA_MANUALMENTE",
        "ALTERADA_DIGITALMENTE",
        "IA_GERADA_EDITADA",
        "INDETERMINADO",
    }
    return normalized if normalized in valid else "INDETERMINADO"


def infer_score(report: str) -> int | None:
    match = re.search(r"\bSCORE(?:_ALTERACAO)?\s*:\s*(\d{1,3})\b", report, re.I)
    if not match:
        return None
    return max(0, min(int(match.group(1)), 100))


def verdict_from_score(score: int | None, fallback: str) -> str:
    fallback = canonical_integrity_label(fallback)
    if score is None:
        return fallback
    if score <= 29:
        return "REAL"
    if score <= 59:
        return "INDETERMINADO"
    if score <= 79:
        return fallback if fallback in {"MODIFICADO", "ALTERADA_MANUALMENTE", "ALTERADA_DIGITALMENTE"} else "MODIFICADO"
    return fallback if fallback not in {"REAL", "INDETERMINADO"} else "IA_GERADA_EDITADA"


def confidence_from_score(score: int | None, verdict: str) -> str:
    if score is None:
        return "media" if verdict != "INDETERMINADO" else "baixa"
    if verdict == "INDETERMINADO":
        return "media" if 40 <= score <= 60 else "baixa"
    distance = min(abs(score - 29), abs(score - 70))
    if distance >= 25:
        return "alta"
    if distance >= 10:
        return "media"
    return "baixa"


def get_field(report: str, field: str) -> str:
    match = re.search(rf"^{field}\s*:\s*(.+)$", report, re.I | re.M)
    return match.group(1).strip() if match else ""


def normalize_report(report: str) -> dict[str, str]:
    score = infer_score(report)
    verdict = verdict_from_score(score, infer_verdict(report))
    confidence = get_field(report, "CONFIANCA") or confidence_from_score(score, verdict)
    justification = get_field(report, "JUSTIFICATIVA")
    if not justification:
        if verdict == "REAL":
            justification = "A imagem apresenta anatomia coerente, reflexos compatíveis e ausencia de bordas artificiais."
        elif verdict == "MODIFICADO":
            justification = "A imagem apresenta sinais de modificacao visual sobreposta ou edicao."
        elif verdict == "ALTERADA_MANUALMENTE":
            justification = "A imagem apresenta marcacoes, esbocos, textos ou anotacoes sobrepostas."
        elif verdict == "ALTERADA_DIGITALMENTE":
            justification = "A imagem apresenta sinais de borroes, recortes, colagens ou edicao de software."
        elif verdict == "IA_GERADA_EDITADA":
            justification = "A imagem apresenta sinais de geracao ou edicao por IA, como anatomia ou textura nao biologica."
        else:
            justification = "A imagem apresenta evidencias insuficientes para classificar a integridade com seguranca."
    tipo = get_field(report, "TIPO") or verdict
    evidencias = get_field(report, "EVIDENCIAS") or "sem evidencias adicionais informadas"
    score_text = str(score if score is not None else (15 if verdict == "REAL" else 50 if verdict == "INDETERMINADO" else 80))
    normalized_report = "\n".join(
        [
            f"VEREDITO: {verdict}",
            f"CONFIANCA: {confidence}",
            f"JUSTIFICATIVA: {justification}",
            f"TIPO: {tipo}",
            f"SCORE_ALTERACAO: {score_text}",
            f"EVIDENCIAS: {evidencias}",
        ]
    )
    return {"verdict": verdict, "report": normalized_report}


def report_from_forensics(forensic: ForensicResult) -> dict[str, str]:
    verdict = canonical_integrity_label(forensic.verdict_hint)
    evidence = "; ".join(forensic.evidence[:5]) or "sem evidencias locais fortes"
    if forensic.calibration_entry:
        justification = str(forensic.calibration_entry.get("justification", "")).strip()
    else:
        justification = ""
    if not justification:
        if verdict == "REAL":
            justification = "A imagem apresenta baixa pontuacao local de alteracao e sinais visuais coerentes."
        elif verdict == "IA_GERADA_EDITADA":
            justification = "A imagem apresenta sinais locais fortes compativeis com edicao ou geracao por IA."
        elif verdict == "MODIFICADO":
            justification = "A imagem apresenta modificacao visual detectada, como linha, seta, desenho ou anotacao sobreposta."
        elif verdict == "ALTERADA_DIGITALMENTE":
            justification = "A imagem apresenta sinais locais compativeis com edicao digital."
        elif verdict == "ALTERADA_MANUALMENTE":
            justification = "A imagem apresenta sinais locais compativeis com marcacao ou sobreposicao manual."
        else:
            justification = "A imagem apresenta sinais locais intermediarios ou conflitantes."
    report = "\n".join(
        [
            f"VEREDITO: {verdict}",
            f"CONFIANCA: {forensic.confidence}",
            f"JUSTIFICATIVA: {justification}",
            f"TIPO: {verdict if verdict != 'IA_GERADA_EDITADA' else 'IA'}",
            f"SCORE_ALTERACAO: {forensic.score}",
            f"EVIDENCIAS: {evidence}",
        ]
    )
    return {"verdict": verdict, "report": report}


def merge_llm_and_forensics(llm_report: str, forensic: ForensicResult) -> dict[str, str]:
    llm_normalized = normalize_report(llm_report)
    llm_score = infer_score(llm_normalized["report"])
    local_score = forensic.score

    if llm_score is None:
        final_score = local_score
    elif forensic.original_used:
        if local_score >= 80 or local_score <= 20:
            final_score = local_score
        else:
            final_score = round((local_score * 0.7) + (llm_score * 0.3))
    else:
        final_score = round((local_score * 0.55) + (llm_score * 0.45))

    local_preferred = forensic.verdict_hint
    llm_preferred = infer_verdict(llm_normalized["report"])
    preferred = local_preferred if local_score >= 60 or local_score <= 29 else llm_preferred
    verdict = verdict_from_score(final_score, preferred)
    confidence = confidence_from_score(final_score, verdict)

    llm_justification = get_field(llm_normalized["report"], "JUSTIFICATIVA")
    local_evidence = "; ".join(forensic.evidence[:4])
    llm_evidence = get_field(llm_normalized["report"], "EVIDENCIAS")
    if verdict == "REAL":
        justification = "A imagem apresenta baixa pontuacao local de alteracao e nao ha evidencias fortes de manipulacao."
    elif forensic.original_used and final_score >= 60:
        justification = "A comparacao com a imagem original mostra diferencas relevantes nas regioes analisadas."
    else:
        justification = llm_justification or "A classificacao combina sinais locais e leitura visual da LLM."

    report = "\n".join(
        [
            f"VEREDITO: {verdict}",
            f"CONFIANCA: {confidence}",
            f"JUSTIFICATIVA: {justification}",
            f"TIPO: {verdict if verdict != 'IA_GERADA_EDITADA' else 'IA'}",
            f"SCORE_ALTERACAO: {final_score}",
            f"EVIDENCIAS: {local_evidence}; LLM: {llm_evidence or 'sem evidencia adicional'}",
        ]
    )
    return {"verdict": verdict, "report": report}


def normalize_history_item(item: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(item)
    report = str(cleaned.get("report") or "").strip()
    if not report or report.lower().startswith("thought") or "<unused" in report:
        fallback = cleaned.get("verdict", "INDETERMINADO")
        report = f"VEREDITO: {fallback}\nSCORE_ALTERACAO: {cleaned.get('forensic_score', 50)}"
    normalized = normalize_report(clean_model_response(report))
    cleaned["verdict"] = normalized["verdict"]
    cleaned["report"] = normalized["report"]
    if cleaned.get("forensic_score") is None:
        cleaned["forensic_score"] = infer_score(normalized["report"])
    if cleaned.get("source") is None:
        cleaned["source"] = "historico_normalizado"
    if cleaned.get("forensic_evidence") is None:
        evidence = get_field(normalized["report"], "EVIDENCIAS")
        cleaned["forensic_evidence"] = [evidence] if evidence else []
    return cleaned


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [normalize_history_item(item) for item in data if isinstance(item, dict)]


def file_sha256(path: Path) -> str:
    return forensic_sha256(path)


def load_calibration() -> dict[str, Any]:
    if not CALIBRATION_FILE.exists():
        return {}
    try:
        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_calibration(calibration: dict[str, Any]) -> None:
    CALIBRATION_FILE.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_calibration_samples():
    return build_calibration_samples(
        load_calibration(),
        [UPLOAD_DIR, TESTS_DIR],
    )


def calibrated_result(image_path: Path) -> dict[str, str] | None:
    calibration = load_calibration()
    entry = calibration.get(file_sha256(image_path))
    if not isinstance(entry, dict):
        return None
    report = "\n".join(
        [
            f"VEREDITO: {entry.get('label', 'INDETERMINADO')}",
            f"CONFIANCA: {entry.get('confidence', 'alta')}",
            f"JUSTIFICATIVA: {entry.get('justification', '')}",
            f"TIPO: {entry.get('type', 'INDETERMINADO')}",
            f"SCORE_ALTERACAO: {entry.get('score', 50)}",
            f"EVIDENCIAS: {entry.get('evidence', '')}",
        ]
    )
    normalized = normalize_report(report)
    return {
        "verdict": normalized["verdict"],
        "report": normalized["report"],
        "duration_seconds": "0.01",
        "source": "calibracao_local",
    }


def estimated_analysis_seconds() -> float:
    durations: list[float] = []
    for item in load_history()[:20]:
        try:
            value = float(item.get("duration_seconds", 0))
        except (TypeError, ValueError):
            continue
        if 3 <= value <= 240:
            durations.append(value)
    if not durations:
        return DEFAULT_ESTIMATED_SECONDS
    durations.sort()
    midpoint = len(durations) // 2
    if len(durations) % 2:
        return durations[midpoint]
    return (durations[midpoint - 1] + durations[midpoint]) / 2


def save_history(history: list[dict[str, Any]]) -> None:
    HISTORY_FILE.write_text(
        json.dumps(history[:100], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_history(entry: dict[str, Any]) -> None:
    history = load_history()
    history.insert(0, entry)
    save_history(history)


def calibration_entry_for_label(label: str) -> dict[str, Any]:
    normalized = canonical_integrity_label(label)
    if normalized == "REAL":
        return {
            "label": "REAL",
            "confidence": "alta",
            "score": 8,
            "type": "REAL",
            "evidence": "perfil visual informado pelo usuario como imagem real/original",
            "justification": "A imagem foi marcada pelo usuario como exemplo real para aprendizado local de padrao.",
        }
    if normalized == "IA_GERADA_EDITADA":
        return {
            "label": "IA_GERADA_EDITADA",
            "confidence": "alta",
            "score": 92,
            "type": "IA",
            "evidence": "rotulo informado pelo usuario como imagem alterada/gerada por IA",
            "justification": "A imagem foi marcada pelo usuario como exemplo de edicao ou geracao por IA.",
        }
    return {
        "label": "MODIFICADO",
        "confidence": "alta",
        "score": 90,
        "type": "MODIFICADO",
        "evidence": "perfil visual informado pelo usuario como imagem modificada/falsa",
        "justification": "A imagem foi marcada pelo usuario como exemplo modificado para aprendizado local de padrao.",
    }


def calibrated_feedback_result(label: str) -> dict[str, Any]:
    entry = calibration_entry_for_label(label)
    report = "\n".join(
        [
            f"VEREDITO: {entry['label']}",
            f"CONFIANCA: {entry['confidence']}",
            f"JUSTIFICATIVA: {entry['justification']}",
            f"TIPO: {entry['type']}",
            f"SCORE_ALTERACAO: {entry['score']}",
            f"EVIDENCIAS: {entry['evidence']}",
        ]
    )
    normalized = normalize_report(report)
    return {
        "verdict": normalized["verdict"],
        "report": normalized["report"],
        "duration_seconds": "0.01",
        "source": "feedback_usuario",
        "forensic_score": entry["score"],
        "forensic_evidence": [entry["evidence"]],
        "forensic_metrics": {},
        "calibration_entry": entry,
    }


def prepare_analysis_image(image_path: Path) -> Path:
    if Image is None or ImageOps is None:
        return image_path

    ANALYSIS_DIR.mkdir(exist_ok=True)
    optimized_path = ANALYSIS_DIR / f"{image_path.stem}_fast.jpg"
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE))
            image.save(
                optimized_path,
                format="JPEG",
                quality=82,
                optimize=True,
                progressive=False,
            )
    except Exception:
        return image_path
    return optimized_path


def analyze_image(image_path: Path, original_path: Path | None = None, use_llm: bool = False) -> dict[str, Any]:
    total_started = time.perf_counter()
    forensic = analyze_forensics(
        image_path,
        original_path=original_path,
        calibration=load_calibration(),
        calibration_samples=load_calibration_samples(),
    )
    if forensic.skip_llm or (FAST_LOCAL_MODE and not use_llm):
        local_result = report_from_forensics(forensic)
        return {
            "verdict": local_result["verdict"],
            "report": local_result["report"],
            "duration_seconds": f"{time.perf_counter() - total_started:.2f}",
            "source": forensic.source if forensic.skip_llm else "modo_rapido_local",
            "forensic_score": forensic.score,
            "forensic_evidence": forensic.evidence,
            "forensic_metrics": forensic.metrics,
        }

    local_summary = "\n".join(
        [
            f"SCORE_LOCAL: {forensic.score}",
            f"VEREDITO_LOCAL: {forensic.verdict_hint}",
            f"FONTE_LOCAL: {forensic.source}",
            f"ORIGINAL_USADA: {'SIM' if forensic.original_used else 'NAO'}",
            f"EVIDENCIAS_LOCAIS: {'; '.join(forensic.evidence)}",
        ]
    )

    prompt = """
Atue como perito visual forense odontologico. Antes de qualquer diagnostico, classifique a integridade da imagem.

Resumo da pericia local ja calculada:
{local_summary}

Use o resumo local como evidencia objetiva. Se discordar dele, aponte uma evidencia visual concreta.

Use esta hierarquia de rotulos:
REAL: anatomia coerente, reflexos compativeis e ausencia de bordas artificiais.
MODIFICADO: esbocos, marcacoes, linhas, setas, textos, rabiscos, borroes, recortes, colagens, clonagem, halos, bordas artificiais ou edicao de software.
IA_GERADA_EDITADA: dentes fundidos, dentes artificialmente lisos/reconstruidos, texturas gengivais artificiais, padroes nao biologicos, objetos derretidos ou geometria impossivel.
INDETERMINADO: imagem ruim ou evidencias conflitantes/insuficientes.

Cheque evidencias objetivas: anatomia, textura, reflexos, sombras, nitidez, compressao, bordas, repeticoes, instrumentos e perspectiva.
Atencao especial em fotos intraorais: dentes naturais costumam ter manchas, translucidez, desgaste e sulcos irregulares. Dentes substituidos por IA costumam ficar excessivamente limpos, uniformes, plastificados ou com sulcos/contornos reconstruidos.

Pontue SCORE_ALTERACAO de 0 a 100:
0-29 = sem sinais relevantes; use VEREDITO REAL.
30-69 = sinais fracos/conflitantes; use VEREDITO INDETERMINADO.
70-100 = sinais fortes; use MODIFICADO ou IA_GERADA_EDITADA.

Nao use INDETERMINADO por cautela generica; so use quando houver sinais suspeitos insuficientes ou imagem ruim.
Nao escreva pensamento interno.
Responda exatamente:
VEREDITO: REAL, MODIFICADO, IA_GERADA_EDITADA ou INDETERMINADO
CONFIANCA: baixa, media ou alta
JUSTIFICATIVA: conclusao tecnica em uma frase, por exemplo "A imagem parece ser uma foto real de uma dentadura."
TIPO: REAL, MODIFICADO, IA ou INDETERMINADO
SCORE_ALTERACAO: numero de 0 a 100
EVIDENCIAS: ate 3 evidencias curtas separadas por ponto e virgula
""".strip().format(local_summary=local_summary)

    client = ollama.Client(host=OLLAMA_HOST)
    analysis_path = prepare_analysis_image(image_path)
    response = client.generate(
        model=MODEL,
        prompt=prompt,
        images=[str(analysis_path)],
        options=MODEL_OPTIONS,
        keep_alive=KEEP_ALIVE,
    )
    report = clean_model_response(response["response"])
    normalized = merge_llm_and_forensics(report, forensic)
    return {
        "verdict": normalized["verdict"],
        "report": normalized["report"],
        "duration_seconds": f"{time.perf_counter() - total_started:.2f}",
        "source": "hibrido_local_llm",
        "forensic_score": forensic.score,
        "forensic_evidence": forensic.evidence,
        "forensic_metrics": forensic.metrics,
    }


def json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def save_uploaded_image(field: Any) -> Path:
    extension = Path(field.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Formato nao suportado. Use PNG, JPG, WEBP ou BMP.")
    UPLOAD_DIR.mkdir(exist_ok=True)
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    with image_path.open("wb") as output:
        output.write(field.file.read())
    return image_path


class PeritoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/history":
            json_response(self, HTTPStatus.OK, {"history": load_history()})
            return

        if self.path == "/metrics":
            json_response(
                self,
                HTTPStatus.OK,
                {"estimated_seconds": round(estimated_analysis_seconds(), 2)},
            )
            return

        if self.path.startswith("/uploads/"):
            filename = Path(unquote(self.path.removeprefix("/uploads/"))).name
            image_path = UPLOAD_DIR / filename
            if not image_path.exists() or image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                self.send_error(HTTPStatus.NOT_FOUND, "Imagem nao encontrada")
                return

            content_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
            }.get(image_path.suffix.lower(), "application/octet-stream")
            body = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "private, max-age=86400")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Pagina nao encontrada")
            return

        body = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path not in {"/analyze", "/calibrate"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint nao encontrado")
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Envie um formulario multipart com o campo image."})
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )
        field = form["image"] if "image" in form else None
        if field is None or not getattr(field, "filename", ""):
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Nenhuma imagem foi enviada."})
            return

        try:
            image_path = save_uploaded_image(field)
        except ValueError as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        original_field = form["original_image"] if "original_image" in form else None
        original_path: Path | None = None
        if original_field is not None and getattr(original_field, "filename", ""):
            try:
                original_path = save_uploaded_image(original_field)
            except ValueError as exc:
                json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"Imagem original: {exc}"})
                return

        if self.path == "/calibrate":
            label_field = form["label"] if "label" in form else None
            label = str(getattr(label_field, "value", "MODIFICADO")).upper()
            label = canonical_integrity_label(label)
            if label not in {"REAL", "MODIFICADO", "IA_GERADA_EDITADA"}:
                json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Rotulo invalido. Use REAL, MODIFICADO ou IA_GERADA_EDITADA."},
                )
                return

            result = calibrated_feedback_result(label)
            calibration = load_calibration()
            calibration_entry = dict(result["calibration_entry"])
            calibration_entry["features"] = feature_profile(image_path)
            calibration_entry["stored_filename"] = image_path.name
            calibration_entry["learned_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            forensic_metrics: dict[str, Any] = {}
            if original_path is not None:
                comparison = compare_with_original(image_path, original_path)
                comparison_features = comparison_feature_profile(comparison)
                forensic_metrics["comparison"] = comparison
                calibration_entry["original_sha256"] = file_sha256(original_path)
                calibration_entry["original_stored_filename"] = original_path.name
                calibration_entry["comparison"] = comparison
                calibration_entry["comparison_features"] = comparison_features
                calibration_entry["evidence"] = (
                    f"{calibration_entry['evidence']}; padrao comparativo salvo contra imagem original"
                )
                if label in {"MODIFICADO", "IA_GERADA_EDITADA"}:
                    if label == "IA_GERADA_EDITADA":
                        calibration_entry["justification"] = (
                            "A imagem foi marcada pelo usuario como edicao ou geracao por IA em comparacao com a original enviada."
                        )
                    else:
                        calibration_entry["justification"] = (
                            "A imagem foi marcada pelo usuario como modificada em comparacao com a original enviada."
                        )
                    original_entry = calibration_entry_for_label("REAL")
                    original_entry["features"] = feature_profile(original_path)
                    original_entry["stored_filename"] = original_path.name
                    original_entry["learned_at"] = calibration_entry["learned_at"]
                    original_entry["paired_modified_sha256"] = file_sha256(image_path)
                    original_entry["evidence"] = "imagem original associada a par comparativo marcado pelo usuario"
                    original_entry["justification"] = (
                        "A imagem foi marcada como original de referencia para aprendizado local comparativo."
                    )
                    calibration[file_sha256(original_path)] = original_entry
                result["source"] = "feedback_usuario_comparacao"
                result["forensic_evidence"] = [calibration_entry["evidence"]]
                result["forensic_metrics"] = forensic_metrics
                refreshed_report = "\n".join(
                    [
                        f"VEREDITO: {calibration_entry['label']}",
                        f"CONFIANCA: {calibration_entry['confidence']}",
                        f"JUSTIFICATIVA: {calibration_entry['justification']}",
                        f"TIPO: {calibration_entry['type']}",
                        f"SCORE_ALTERACAO: {calibration_entry['score']}",
                        f"EVIDENCIAS: {calibration_entry['evidence']}",
                    ]
                )
                normalized = normalize_report(refreshed_report)
                result["verdict"] = normalized["verdict"]
                result["report"] = normalized["report"]
            calibration[file_sha256(image_path)] = calibration_entry
            save_calibration(calibration)

            entry = {
                "id": image_path.stem,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": Path(field.filename).name,
                "stored_filename": image_path.name,
                "image_url": f"/uploads/{image_path.name}",
                "original_filename": Path(original_field.filename).name if original_path is not None else None,
                "original_stored_filename": original_path.name if original_path is not None else None,
                "original_image_url": f"/uploads/{original_path.name}" if original_path is not None else None,
                "model": "calibracao_local",
                "verdict": result["verdict"],
                "report": result["report"],
                "duration_seconds": result["duration_seconds"],
                "source": result["source"],
                "forensic_score": result["forensic_score"],
                "forensic_evidence": result["forensic_evidence"],
                "forensic_metrics": result.get("forensic_metrics", {}),
            }
            append_history(entry)
            payload = {key: value for key, value in result.items() if key != "calibration_entry"}
            json_response(self, HTTPStatus.OK, {**payload, "history_item": entry})
            return

        use_llm = "use_llm" in form and str(getattr(form["use_llm"], "value", "")).lower() in {"1", "true", "sim", "on"}

        try:
            result = analyze_image(image_path, original_path, use_llm=use_llm)
        except ollama.ResponseError as exc:
            json_response(self, HTTPStatus.BAD_GATEWAY, {"error": f"Erro do Ollama: {exc}"})
            return
        except Exception as exc:
            json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Erro inesperado: {exc}"})
            return

        entry = {
            "id": image_path.stem,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": Path(field.filename).name,
            "stored_filename": image_path.name,
            "image_url": f"/uploads/{image_path.name}",
            "original_filename": Path(original_field.filename).name if original_path is not None else None,
            "original_stored_filename": original_path.name if original_path is not None else None,
            "original_image_url": f"/uploads/{original_path.name}" if original_path is not None else None,
            "model": MODEL,
            "verdict": result["verdict"],
            "report": result["report"],
            "duration_seconds": result.get("duration_seconds"),
            "source": result.get("source", "llm"),
            "forensic_score": result.get("forensic_score"),
            "forensic_evidence": result.get("forensic_evidence", []),
            "forensic_metrics": result.get("forensic_metrics", {}),
        }
        append_history(entry)
        json_response(self, HTTPStatus.OK, {**result, "history_item": entry})

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> int:
    configure_stdio()
    UPLOAD_DIR.mkdir(exist_ok=True)
    ANALYSIS_DIR.mkdir(exist_ok=True)
    if HISTORY_FILE.exists():
        save_history(load_history())
    server = ThreadingHTTPServer(("127.0.0.1", PORT), PeritoHandler)
    print(f"Pagina web: http://127.0.0.1:{PORT}")
    print(f"Ollama: {OLLAMA_HOST} | Modelo: {MODEL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Servidor encerrado.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
