const HELMET_API_BASE = "http://127.0.0.1:5001";

const state = {
  currentSample: null,
  progressTimer: null,
  progressValue: 0,
};

const qs = (selector) => document.querySelector(selector);

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function apiUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${HELMET_API_BASE}${path}`;
}

function setLoading(isLoading) {
  const btn = qs("#helmetPredictBtn");
  if (!btn) return;
  btn.disabled = isLoading;
  btn.textContent = isLoading ? "분석 중..." : "헬멧 탐지 시작";
}

function showResultPanel() {
  const panel = qs("#helmetResultPanel");
  if (panel) panel.classList.remove("hidden");
}

function scrollToResultPanel() {
  const panel = qs("#helmetResultPanel");
  if (!panel) return;
  window.setTimeout(() => {
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 80);
}

function updateProgressBar(value, message) {
  const safeValue = Math.max(0, Math.min(100, Math.round(value)));
  const fill = qs("#helmetProgressFill");
  const percent = qs("#helmetProgressPercent");
  const text = qs("#helmetProgressText");

  if (fill) fill.style.width = `${safeValue}%`;
  if (percent) percent.textContent = `${safeValue}%`;
  if (text && message) text.textContent = message;
}

function stopProgress() {
  if (state.progressTimer) {
    window.clearInterval(state.progressTimer);
    state.progressTimer = null;
  }
}

function renderProgress(message = "분석 요청을 처리하고 있습니다.") {
  showResultPanel();
  const target = qs("#helmetResultContent");
  if (!target) return;

  target.innerHTML = `
    <div class="helmet-progress-card">
      <div class="helmet-progress-head">
        <div>
          <strong>분석 중...</strong>
          <p id="helmetProgressText">${escapeHtml(message)}</p>
        </div>
        <span id="helmetProgressPercent">0%</span>
      </div>
      <div class="helmet-progress-track" aria-label="분석 진행률">
        <div class="helmet-progress-fill" id="helmetProgressFill" style="width: 0%"></div>
      </div>
      <p class="helmet-progress-note">파일 크기와 영상 길이에 따라 처리 시간이 달라질 수 있습니다.</p>
    </div>
  `;
}

function startProgress(type = "file") {
  stopProgress();
  state.progressValue = 4;

  const message = type === "video"
    ? "영상 프레임을 분석하고 GIF 미리보기를 준비하고 있습니다."
    : "이미지를 분석하고 결과를 준비하고 있습니다.";

  renderProgress(message);
  updateProgressBar(state.progressValue, message);
  scrollToResultPanel();

  state.progressTimer = window.setInterval(() => {
    if (state.progressValue < 45) {
      state.progressValue += Math.random() * 7 + 3;
    } else if (state.progressValue < 78) {
      state.progressValue += Math.random() * 4 + 1.5;
    } else if (state.progressValue < 93) {
      state.progressValue += Math.random() * 1.2 + 0.3;
    }

    state.progressValue = Math.min(state.progressValue, 93);
    const nextMessage = state.progressValue > 72
      ? "결과 화면을 준비하고 있습니다."
      : message;
    updateProgressBar(state.progressValue, nextMessage);
  }, 420);
}

function finishProgress() {
  stopProgress();
  state.progressValue = 100;
  updateProgressBar(100, "분석이 완료되었습니다.");
}

function renderError(message) {
  showResultPanel();
  const target = qs("#helmetResultContent");
  if (!target) return;
  target.innerHTML = `
    <div class="helmet-status-card helmet-status-error">
      <strong>분석을 완료하지 못했습니다.</strong>
      <p>${escapeHtml(message || "잠시 후 다시 시도해 주세요.")}</p>
    </div>
  `;
}

function buildSummaryCards(summary = {}, type = "image") {
  const cards = [];

  if (type === "video") {
    cards.push(["분석 프레임", summary.processed_frames ?? "-"]);
    cards.push(["탐지 프레임", summary.detection_frames ?? "-"]);
    cards.push(["헬멧 착용 감지", summary.rider_helmet_detections ?? 0]);
    cards.push(["헬멧 미착용 감지", summary.rider_no_helmet_detections ?? 0]);
    cards.push(["주요 탐지 신뢰도", formatPercent(summary.max_confidence)]);
    cards.push(["평균 탐지 신뢰도", formatPercent(summary.avg_confidence)]);
  } else {
    cards.push(["전체 라이더", summary.total_riders ?? 0]);
    cards.push(["헬멧 착용", summary.rider_helmet ?? 0]);
    cards.push(["헬멧 미착용", summary.rider_no_helmet ?? 0]);
    cards.push(["주요 탐지 신뢰도", formatPercent(summary.max_confidence)]);
    cards.push(["평균 탐지 신뢰도", formatPercent(summary.avg_confidence)]);
    cards.push(["처리 방식", "이미지 분석"]);
  }

  return cards.map(([label, value]) => `
    <div class="helmet-info-item">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `).join("");
}

function buildModelCards(modelInfo = {}, type = "image") {
  const confValue = Number(modelInfo.conf_threshold ?? 0.4);
  const confLabel = Number.isFinite(confValue)
    ? `${Math.round(confValue * 100)}% 이상`
    : "40% 이상";

  const cards = [
    ["모델", modelInfo.model_name || "YOLO11s"],
    ["출력 클래스", "헬멧 착용 / 미착용"],
    ["탐지 표시 기준", confLabel],
    ["처리 방식", type === "video" ? "영상 프레임 분석" : "이미지 분석"],
  ];

  if (type === "video") {
    cards.push(["GIF 미리보기", "중간 구간 자동 생성"]);
    cards.push(["전체 결과", "파일 저장 제공"]);
  } else {
    cards.push(["입력 크기", modelInfo.image_size || "640"]);
  }

  return cards.map(([label, value]) => `
    <div class="helmet-info-item">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `).join("");
}

function buildReasons(reasons = []) {
  if (!Array.isArray(reasons) || reasons.length === 0) {
    return `<p>분석 사유가 별도로 제공되지 않았습니다.</p>`;
  }
  return `<ul>${reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}</ul>`;
}

function renderAnalysisDetails(data) {
  const summary = data.summary || {};
  const modelInfo = data.model_info || {};
  const type = data.type || "image";
  const reasons = data.analysis_reasons || summary.analysis_reasons || [];

  return `
    <details class="helmet-analysis-details">
      <summary>분석 정보 보기</summary>
      <div class="helmet-info-block">
        <h3>분석 요약</h3>
        <div class="helmet-info-grid">
          ${buildSummaryCards(summary, type)}
        </div>
      </div>
      <div class="helmet-info-block">
        <h3>분석 사유</h3>
        <div class="helmet-reason-box">
          ${buildReasons(reasons)}
        </div>
      </div>
      <div class="helmet-info-block">
        <h3>모델 정보</h3>
        <div class="helmet-info-grid">
          ${buildModelCards(modelInfo, type)}
        </div>
      </div>
    </details>
  `;
}

function renderDetectionList(detections = []) {
  if (!Array.isArray(detections) || detections.length === 0) {
    return `<p class="helmet-muted">표시할 탐지 박스가 없습니다.</p>`;
  }
  return `
    <div class="helmet-detection-list">
      ${detections.map((det, idx) => `
        <div class="helmet-detection-row">
          <span>#${idx + 1}</span>
          <strong>${escapeHtml(det.class_name)}</strong>
          <em>${formatPercent((det.confidence || 0) * 100)}</em>
        </div>
      `).join("")}
    </div>
  `;
}

function renderResult(data) {
  showResultPanel();
  const target = qs("#helmetResultContent");
  if (!target) return;

  if (!data.ok) {
    renderError(data.message || "분석 실패");
    return;
  }

  if (data.type === "video") {
    const gifUrl = apiUrl(data.preview_gif_url);
    const frameUrl = apiUrl(data.representative_image_url);
    const downloadUrl = apiUrl(data.download_url || data.result_video_url);
    const preview = gifUrl
      ? `<img class="helmet-result-media helmet-gif-preview" src="${gifUrl}" alt="프레임별 탐지 GIF 미리보기">`
      : frameUrl
        ? `<img class="helmet-result-media" src="${frameUrl}" alt="대표 프레임">`
        : `<div class="helmet-placeholder">표시할 미리보기가 없습니다.</div>`;

    target.innerHTML = `
      <div class="helmet-result-layout">
        <div class="helmet-media-wrap">
          ${preview}
          <p class="helmet-media-caption">영상 중간 구간의 탐지 결과를 GIF 미리보기로 표시합니다.</p>
        </div>
        <div class="helmet-result-side">
          <div class="helmet-status-card">
            <strong>영상 분석 완료</strong>
            <p>프레임별 탐지 결과를 미리보기로 확인할 수 있습니다.</p>
          </div>
          ${downloadUrl ? `<a class="btn ghost helmet-download-btn" href="${downloadUrl}" target="_blank" rel="noopener">전체 결과 영상 저장</a>` : ""}
          ${renderAnalysisDetails(data)}
        </div>
      </div>
    `;
    return;
  }

  const imageUrl = apiUrl(data.result_url);
  target.innerHTML = `
    <div class="helmet-result-layout">
      <div class="helmet-media-wrap">
        <img class="helmet-result-media" src="${imageUrl}" alt="헬멧 탐지 결과 이미지">
      </div>
      <div class="helmet-result-side">
        <div class="helmet-status-card">
          <strong>이미지 분석 완료</strong>
          <p>탐지된 라이더의 헬멧 착용 여부를 결과 이미지에 표시했습니다.</p>
        </div>
        ${renderAnalysisDetails(data)}
        <div class="helmet-box-list-card">
          <h3>탐지 박스</h3>
          ${renderDetectionList(data.detections || [])}
        </div>
      </div>
    </div>
  `;
}

async function postToHelmetApi(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${HELMET_API_BASE}/api/helmet/predict`, {
    method: "POST",
    body: formData,
  });

  let data;
  try {
    data = await response.json();
  } catch (error) {
    throw new Error("분석 서버 응답을 읽을 수 없습니다.");
  }

  if (!response.ok || !data.ok) {
    throw new Error(data.message || "분석 요청에 실패했습니다.");
  }

  return data;
}

async function handleUpload(event) {
  event.preventDefault();
  const input = qs("#helmetFileInput");
  const file = input?.files?.[0];
  if (!file) {
    renderError("분석할 파일을 선택해 주세요.");
    return;
  }

  const type = file.type?.startsWith("video/") ? "video" : "image";
  setLoading(true);
  startProgress(type);
  try {
    const data = await postToHelmetApi(file);
    finishProgress();
    window.setTimeout(() => renderResult(data), 250);
  } catch (error) {
    stopProgress();
    renderError(error.message);
    scrollToResultPanel();
  } finally {
    setLoading(false);
  }
}

async function analyzeSample(sample) {
  setLoading(true);
  startProgress(sample.type || "file");
  try {
    const response = await fetch(sample.url);
    if (!response.ok) throw new Error("예시 파일을 불러오지 못했습니다.");
    const blob = await response.blob();
    const extension = sample.type === "video" ? "mp4" : "jpg";
    const file = new File([blob], `sample.${extension}`, { type: blob.type || (sample.type === "video" ? "video/mp4" : "image/jpeg") });
    const data = await postToHelmetApi(file);
    finishProgress();
    window.setTimeout(() => renderResult(data), 250);
  } catch (error) {
    stopProgress();
    renderError(error.message);
    scrollToResultPanel();
  } finally {
    setLoading(false);
  }
}

function renderSamples(samples = []) {
  const grid = qs("#helmetSampleGrid");
  if (!grid) return;
  if (!samples.length) {
    grid.innerHTML = `<div class="helmet-empty-message">표시할 예시 파일이 없습니다.</div>`;
    return;
  }

  grid.innerHTML = samples.map((sample, index) => {
    const thumb = sample.thumbnail_url || sample.url;
    const media = thumb
      ? `<img src="${thumb}" alt="${escapeHtml(sample.label)} 썸네일">`
      : `<div class="helmet-sample-placeholder">${sample.type === "video" ? "예시 영상" : "예시 이미지"}</div>`;
    return `
      <article class="helmet-sample-card" data-sample-index="${index}">
        <div class="helmet-sample-thumb">
          ${media}
          <span>${sample.type === "video" ? "VIDEO" : "IMAGE"}</span>
        </div>
        <div class="helmet-sample-body">
          <strong>${sample.type === "video" ? "예시 영상" : "예시 이미지"}</strong>
          <p>${sample.type === "video" ? "중간 구간 GIF 미리보기와 분석 요약을 확인합니다." : "이미지 탐지 결과를 바로 확인합니다."}</p>
          <button class="btn ghost helmet-sample-btn" type="button">이 예시 분석해보기</button>
        </div>
      </article>
    `;
  }).join("");

  grid.querySelectorAll(".helmet-sample-card").forEach((card) => {
    card.addEventListener("click", (event) => {
      const index = Number(card.dataset.sampleIndex);
      if (!Number.isNaN(index)) analyzeSample(samples[index]);
    });
  });
}

async function loadSamples() {
  try {
    const response = await fetch("/api/helmet/samples");
    const data = await response.json();
    renderSamples(data.samples || []);
  } catch (error) {
    const grid = qs("#helmetSampleGrid");
    if (grid) grid.innerHTML = `<div class="helmet-empty-message">예시 파일을 불러오지 못했습니다.</div>`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = qs("#helmetUploadForm");
  if (form) form.addEventListener("submit", handleUpload);
  loadSamples();
});
