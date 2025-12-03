// Results page script for Ethical Eye
console.log("Ethical Eye: Results page script loading...");

const visionEndpoint = "http://127.0.0.1:5000/vision/analyze";
let selectedScreenshot = null;
let detectionsCache = [];
let screenshotMeta = null;

document.addEventListener("DOMContentLoaded", function () {
  console.log("Ethical Eye: DOM ready, loading results...");

  loadResults();
  setupVisionUploader();
  loadStoredVisionResults();
  window.addEventListener("resize", () => {
    if (detectionsCache.length) {
      drawVisionDetections(detectionsCache);
    } else {
      resetOverlayCanvas();
    }
  });
});

function loadResults() {
  const loading = document.getElementById("loading");
  const content = document.getElementById("content");
  const noResults = document.getElementById("no-results");

  if (
    typeof chrome === "undefined" ||
    !chrome.storage ||
    !chrome.storage.local
  ) {
    loading.style.display = "none";
    noResults.style.display = "block";
    noResults.textContent = "Chrome storage unavailable in this context.";
    return;
  }

  chrome.storage.local.get(
    ["ethicalEyeFullResults", "ethicalEyeResultsTimestamp"],
    function (data) {
      loading.style.display = "none";

      if (
        !data.ethicalEyeFullResults ||
        data.ethicalEyeFullResults.length === 0
      ) {
        noResults.style.display = "block";
        return;
      }

      const results = data.ethicalEyeFullResults;
      const timestamp = data.ethicalEyeResultsTimestamp;

      console.log("Ethical Eye: Loaded", results.length, "results");

      displaySummary(results);
      displayPatterns(results);

      if (timestamp) {
        const timestampEl = document.getElementById("timestamp");
        const date = new Date(timestamp);
        timestampEl.textContent = `Analysis completed: ${date.toLocaleString()}`;
      }

      content.style.display = "block";
    }
  );
}

// ---------- Multimodal screenshot analysis ----------
function setupVisionUploader() {
  const input = document.getElementById("visionScreenshotInput");
  const analyzeBtn = document.getElementById("visionAnalyzeBtn");
  const preview = document.getElementById("visionPreview");

  if (!input || !analyzeBtn || !preview) {
    return;
  }

  input.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) {
      selectedScreenshot = null;
      analyzeBtn.disabled = true;
      setVisionStatus("No screenshot selected.", "info");
      preview.style.display = "none";
      return;
    }

    selectedScreenshot = file;
    analyzeBtn.disabled = false;
    setVisionStatus("Screenshot ready. Click analyze to continue.", "info");
    renderVisionPreview(file);
  });

  analyzeBtn.addEventListener("click", () => {
    if (!selectedScreenshot) {
      setVisionStatus("Please choose a screenshot first.", "error");
      return;
    }
    runVisionAnalysis(selectedScreenshot);
  });
}

function renderVisionPreview(file) {
  const reader = new FileReader();
  reader.onload = function (e) {
    const img = document.getElementById("visionPreviewImage");
    const preview = document.getElementById("visionPreview");
    img.onload = () => {
      resetOverlayCanvas();
      screenshotMeta = {
        width: img.naturalWidth,
        height: img.naturalHeight,
      };
      if (detectionsCache.length) {
        drawVisionDetections(detectionsCache);
      }
    };
    img.src = e.target.result;
    preview.style.display = "block";
  };
  reader.readAsDataURL(file);
}

function resetOverlayCanvas() {
  const canvas = document.getElementById("visionOverlayCanvas");
  const img = document.getElementById("visionPreviewImage");
  if (!canvas || !img || !img.naturalWidth) {
    return;
  }

  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function runVisionAnalysis(file) {
  const analyzeBtn = document.getElementById("visionAnalyzeBtn");
  const formData = new FormData();
  formData.append("file", file, file.name || "screenshot.png");

  analyzeBtn.disabled = true;
  setVisionStatus("Analyzing screenshot...", "working");

  fetch(visionEndpoint, {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Vision service error: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      detectionsCache = data.detections || [];
      drawVisionDetections(detectionsCache);
      renderVisionDetections(detectionsCache, data.metadata || {});

      if (detectionsCache.length === 0) {
        setVisionStatus("No visual dark patterns detected.", "success");
      } else {
        setVisionStatus(
          `Found ${detectionsCache.length} potential dark patterns.`,
          "success"
        );
      }
    })
    .catch((error) => {
      console.error("Vision analysis failed", error);
      setVisionStatus(
        "Vision service unavailable. Start the API gateway and vision service.",
        "error"
      );
    })
    .finally(() => {
      analyzeBtn.disabled = false;
    });
}

function drawVisionDetections(detections) {
  const canvas = document.getElementById("visionOverlayCanvas");
  const img = document.getElementById("visionPreviewImage");
  if (!canvas || !img || !img.naturalWidth) {
    return;
  }

  const ctx = canvas.getContext("2d");
  const baseWidth = screenshotMeta?.width || img.naturalWidth;
  const baseHeight = screenshotMeta?.height || img.naturalHeight;

  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;

  const scaleX = canvas.width / baseWidth;
  const scaleY = canvas.height / baseHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "12px Segoe UI";

  detections.forEach((det, index) => {
    const [x, y, w, h] = det.bbox || [0, 0, 0, 0];
    const drawX = x * scaleX;
    const drawY = y * scaleY;
    const drawW = w * scaleX;
    const drawH = h * scaleY;

    ctx.strokeStyle = "#ff6b6b";
    ctx.fillStyle = "rgba(255, 107, 107, 0.15)";
    ctx.strokeRect(drawX, drawY, drawW, drawH);
    ctx.fillRect(drawX, drawY, drawW, drawH);

    const label = `${index + 1}. ${det.label || "Suspicious Region"}`;
    const textWidth = ctx.measureText(label).width + 10;
    const boxY = Math.max(0, drawY - 18);

    ctx.fillStyle = "#ff6b6b";
    ctx.fillRect(drawX, boxY, textWidth, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, drawX + 4, boxY + 13);
  });
}

function renderVisionDetections(detections, metadata) {
  const list = document.getElementById("visionDetectionList");
  if (!list) {
    return;
  }

  if (!detections.length) {
    list.innerHTML = `
            <div class="vision-empty">
                <p>No visual dark patterns detected yet. Upload a screenshot to get started.</p>
            </div>
        `;
    return;
  }

  const cards = detections
    .map((det, index) => {
      const reason = det.reason || "Visual anomaly detected.";
      const score = det.score ? (det.score * 100).toFixed(1) : "0.0";
      const text = det.text
        ? `<p class="vision-detection-text">${escapeHtml(det.text)}</p>`
        : "";
      const textModel = det.text_category
        ? `
                <p class="vision-text-classifier">
                    Text model: ${escapeHtml(det.text_category)} (${(
            (det.text_confidence || 0) * 100
          ).toFixed(1)}%)
                </p>`
        : "";

      return `
                <div class="vision-detection-card">
                    <div class="vision-detection-header">
                        <span class="vision-badge">${index + 1}</span>
                        <div>
                            <h4>${escapeHtml(
                              det.label || "Suspicious Region"
                            )}</h4>
                            <p class="vision-score">Confidence: ${score}%</p>
                        </div>
                    </div>
                    <p class="vision-reason">${escapeHtml(reason)}</p>
                    ${text}
                    ${textModel}
                </div>
            `;
    })
    .join("");

  const info = `
        <div class="vision-metadata">
            <p>Model: ${escapeHtml(
              metadata.model || "Unknown"
            )} • Device: ${escapeHtml(
    metadata.device || "cpu"
  )} • Regions scanned: ${metadata.regions_evaluated || 0}</p>
        </div>
    `;

  list.innerHTML = info + cards;
}

function setVisionStatus(message, state) {
  const status = document.getElementById("visionStatus");
  if (!status) {
    return;
  }

  status.textContent = message;
  status.className = `vision-status ${state || ""}`;
}

function loadStoredVisionResults() {
  if (
    typeof chrome === "undefined" ||
    !chrome.storage ||
    !chrome.storage.local
  ) {
    return;
  }

  chrome.storage.local.get(
    [
      "ethicalEyeVisionScreenshot",
      "ethicalEyeVisionDetections",
      "ethicalEyeVisionTimestamp",
      "ethicalEyeFullResults",
      "ethicalEyeVisionScreenshotMeta",
    ],
    (data) => {
      if (
        !data ||
        !data.ethicalEyeVisionScreenshot ||
        !data.ethicalEyeFullResults
      ) {
        return;
      }

      const screenshot = data.ethicalEyeVisionScreenshot;
      const fullResults = data.ethicalEyeFullResults || [];
      screenshotMeta = data.ethicalEyeVisionScreenshotMeta || null;

      // Build detections from text-based DOM boxes for high accuracy
      detectionsCache = fullResults
        .filter((r) => r.box && typeof r.box.x === "number")
        .map((r) => ({
          label: r.category,
          score: r.confidence,
          reason: r.explanation,
          text: r.text,
          bbox: [
            r.box.x,
            r.box.y,
            r.box.width,
            r.box.height,
          ],
        }));

      const img = document.getElementById("visionPreviewImage");
      const preview = document.getElementById("visionPreview");
      if (img && preview) {
        img.onload = () => {
          resetOverlayCanvas();
          drawVisionDetections(detectionsCache);
        };
        img.src = screenshot;
        preview.style.display = "block";
      }

      renderVisionDetections(detectionsCache, {
        model: "text+DOM-based detector",
        device: "browser",
        regions_evaluated: detectionsCache.length,
      });

      const timestamp =
        data.ethicalEyeVisionTimestamp &&
        new Date(data.ethicalEyeVisionTimestamp).toLocaleString();
      if (timestamp) {
        setVisionStatus(
          `Loaded screenshot with ${detectionsCache.length} text-based dark patterns from ${timestamp}`,
          "success"
        );
      } else {
        setVisionStatus(
          `Loaded screenshot with ${detectionsCache.length} text-based dark patterns`,
          "success"
        );
      }
    }
  );
}

function displaySummary(results) {
  const summaryContainer = document.getElementById("summary");

  // Count patterns by category
  const categoryCount = {};
  let totalConfidence = 0;

  results.forEach((result) => {
    categoryCount[result.category] = (categoryCount[result.category] || 0) + 1;
    totalConfidence += result.confidence;
  });

  const avgConfidence = ((totalConfidence / results.length) * 100).toFixed(1);
  const categories = Object.keys(categoryCount).length;

  summaryContainer.innerHTML = `
        <div class="summary-card">
            <h3>${results.length}</h3>
            <p>Dark Patterns Detected</p>
        </div>
        <div class="summary-card">
            <h3>${categories}</h3>
            <p>Different Categories</p>
        </div>
        <div class="summary-card">
            <h3>${avgConfidence}%</h3>
            <p>Average Confidence</p>
        </div>
    `;
}

function displayPatterns(results) {
  const container = document.getElementById("patterns-container");

  container.innerHTML = "";

  results.forEach((result, index) => {
    const card = createPatternCard(result, index + 1);
    container.appendChild(card);
  });
}

function createPatternCard(result, index) {
  const card = document.createElement("div");
  card.className = "pattern-card";

  // Create top words badges
  const topWordsHTML =
    result.top_words && result.top_words.length > 0
      ? result.top_words
          .map(
            ([word, score]) =>
              `<div class="word-badge">
                <span>${word}</span>
                <span class="word-score">${score.toFixed(3)}</span>
            </div>`
          )
          .join("")
      : '<p style="color: #999; font-style: italic;">No SHAP data available</p>';

  // Create SHAP visualization
  const shapVisualization = createSHAPVisualization(result);

  card.innerHTML = `
        <div class="pattern-header">
            <div class="pattern-category">${index}. ${result.category}</div>
            <div class="pattern-confidence">${(result.confidence * 100).toFixed(
              1
            )}%</div>
        </div>
        
        <div class="pattern-text">${escapeHtml(result.text)}</div>
        
        ${
          result.pattern_description
            ? `
            <div class="pattern-description">
                <strong>Description:</strong> ${escapeHtml(
                  result.pattern_description
                )}
            </div>
        `
            : ""
        }
        
        ${
          result.explanation
            ? `
            <div class="pattern-explanation">
                <strong>AI Explanation:</strong> ${escapeHtml(
                  result.explanation
                )}
            </div>
        `
            : ""
        }
        
        <div class="shap-section">
            <h3>Top Contributing Words (SHAP Importance)</h3>
            <div class="top-words">
                ${topWordsHTML}
            </div>
            
            ${
              shapVisualization
                ? `
                <div class="shap-visualization">
                    <h4 style="margin-bottom: 10px; color: #666;">Token-level SHAP Values</h4>
                    ${shapVisualization}
                </div>
            `
                : ""
            }
            
            ${
              !result.shap_computed &&
              result.top_words &&
              result.top_words.length > 0
                ? `
                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin-top: 10px; border-radius: 5px; font-size: 0.9em; color: #856404;">
                    <strong>Note:</strong> SHAP computation unavailable. Showing keyword-based indicators instead.
                </div>
            `
                : ""
            }
        </div>
    `;

  return card;
}

function createSHAPVisualization(result) {
  if (
    !result.tokens ||
    !result.shap_values ||
    result.tokens.length === 0 ||
    result.shap_values.length === 0
  ) {
    return null;
  }

  // Find max absolute value for normalization
  const maxAbsValue = Math.max(...result.shap_values.map((v) => Math.abs(v)));

  if (maxAbsValue === 0) {
    return '<p style="color: #999; font-style: italic;">No SHAP values available</p>';
  }

  // Only show visualization if SHAP was actually computed (not keyword-based)
  if (!result.shap_computed) {
    return '<p style="color: #999; font-style: italic;">SHAP values not computed (using keyword-based fallback)</p>';
  }

  // Create visualization rows
  const rows = [];
  const minLength = Math.min(result.tokens.length, result.shap_values.length);

  for (let i = 0; i < minLength; i++) {
    const token = result.tokens[i];
    const value = result.shap_values[i];

    // Skip special tokens
    if (token.match(/^\[(CLS|SEP|PAD|UNK)\]$/) || token.trim() === "") {
      continue;
    }

    // Normalize value for display (0-100%)
    const normalizedValue = (Math.abs(value) / maxAbsValue) * 100;
    const isPositive = value >= 0;

    rows.push(`
            <div class="token-row">
                <div class="token-text">${escapeHtml(token)}</div>
                <div class="shap-bar">
                    <div class="shap-bar-fill ${
                      isPositive ? "positive" : "negative"
                    }" 
                         style="width: ${normalizedValue}%"></div>
                </div>
                <div class="shap-value">${value >= 0 ? "+" : ""}${value.toFixed(
      4
    )}</div>
            </div>
        `);
  }

  return rows.length > 0
    ? rows.join("")
    : '<p style="color: #999; font-style: italic;">No valid tokens to display</p>';
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

console.log("Ethical Eye: Results page script loaded!");
