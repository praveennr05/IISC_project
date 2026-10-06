/**
 * CardioGuard AI - Frontend Controller
 * Connects clinical user form to FastAPI /predict and hand-written SQL /stats
 */

// Clinical Presets for Demonstration
const PRESETS = {
  low: {
    age: 38,
    sex: 0,        // Female
    cp: 2,         // Atypical angina
    trestbps: 115, // Normal blood pressure
    chol: 180,     // Optimal cholesterol
    fbs: 0,        // Normal glucose
    restecg: 0,    // Normal ECG
    thalach: 172,  // Excellent exercise heart rate
    exang: 0,      // No exercise angina
    oldpeak: 0.0,  // No ST depression
    slope: 1,      // Upsloping
    ca: 0,         // 0 vessels blocked
    thal: 3        // Normal thalassemia
  },
  moderate: {
    age: 52,
    sex: 1,        // Male
    cp: 3,         // Non-anginal pain
    trestbps: 135, // Pre-hypertension
    chol: 235,     // Borderline high cholesterol
    fbs: 0,
    restecg: 1,    // ST-T wave changes
    thalach: 142,
    exang: 0,
    oldpeak: 1.2,  // Mild ST depression
    slope: 2,      // Flat
    ca: 0,
    thal: 3
  },
  high: {
    age: 67,
    sex: 1,        // Male
    cp: 4,         // Asymptomatic / ischemic
    trestbps: 160, // Stage 2 Hypertension
    chol: 286,     // High cholesterol
    fbs: 0,
    restecg: 2,    // Left ventricular hypertrophy
    thalach: 108,  // Chronotropic incompetence
    exang: 1,      // Exercise induced angina
    oldpeak: 2.6,  // Severe ST depression
    slope: 2,      // Flat
    ca: 2,         // 2 major vessels blocked
    thal: 7        // Reversible defect
  }
};

document.addEventListener("DOMContentLoaded", () => {
  checkApiHealth();
  fetchStats();

  const form = document.getElementById("prediction-form");
  form.addEventListener("submit", handleFormSubmit);
});

// Load preset values into form
function loadPreset(profileType) {
  const profile = PRESETS[profileType];
  if (!profile) return;

  for (const [key, value] of Object.entries(profile)) {
    const input = document.getElementById(key);
    if (input) {
      input.value = value;
    }
  }

  // Clear existing errors
  hideError();
  // Automatically trigger prediction for instant demonstration
  submitPrediction();
}

// Health check liveness poll
async function checkApiHealth() {
  const badge = document.getElementById("api-status-badge");
  try {
    const res = await fetch("/health");
    if (res.ok) {
      const data = await res.json();
      badge.textContent = `Online (S = ${data.candidate_seed})`;
      badge.className = "badge status-badge status-online";
    } else {
      badge.textContent = "Degraded (503)";
      badge.className = "badge status-badge status-offline";
    }
  } catch (err) {
    badge.textContent = "Offline";
    badge.className = "badge status-badge status-offline";
  }
}

// Form Submission Handler
function handleFormSubmit(e) {
  e.preventDefault();
  submitPrediction();
}

async function submitPrediction() {
  const submitBtn = document.getElementById("submit-btn");
  const form = document.getElementById("prediction-form");
  hideError();

  // Assemble payload from form inputs
  const formData = new FormData(form);
  const payload = {
    age: parseInt(formData.get("age"), 10),
    sex: parseInt(formData.get("sex"), 10),
    cp: parseInt(formData.get("cp"), 10),
    trestbps: parseInt(formData.get("trestbps"), 10),
    chol: parseInt(formData.get("chol"), 10),
    fbs: parseInt(formData.get("fbs"), 10),
    restecg: parseInt(formData.get("restecg"), 10),
    thalach: parseInt(formData.get("thalach"), 10),
    exang: parseInt(formData.get("exang"), 10),
    oldpeak: parseFloat(formData.get("oldpeak")),
    slope: parseInt(formData.get("slope"), 10),
    ca: parseInt(formData.get("ca"), 10),
    thal: parseInt(formData.get("thal"), 10)
  };

  submitBtn.disabled = true;
  submitBtn.querySelector(".btn-text").textContent = "Analyzing Diagnostics...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      if (response.status === 422 && data.details) {
        const errorList = data.details.map(d => `• <strong>${d.field}:</strong> ${d.message}`).join("<br>");
        showError(`<strong>Input Validation Error:</strong><br>${errorList}`);
      } else {
        showError(data.detail || "An unexpected error occurred while predicting.");
      }
      return;
    }

    // Render results
    renderResults(data);
    // Refresh stats from database
    fetchStats();
  } catch (err) {
    showError(`Network / Server Connection Error: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector(".btn-text").textContent = "Run Cardiovascular Risk Prediction";
  }
}

function renderResults(data) {
  const placeholder = document.getElementById("result-placeholder");
  const resultContent = document.getElementById("result-content");
  const categoryBadge = document.getElementById("risk-category-badge");
  const gaugeBar = document.getElementById("gauge-bar");
  const riskPercentText = document.getElementById("risk-percentage-text");
  const plainWordsText = document.getElementById("plain-words-text");
  const guidanceText = document.getElementById("clinical-guidance-text");
  const lifestyleList = document.getElementById("lifestyle-list");

  placeholder.style.display = "none";
  resultContent.style.display = "block";

  const percent = Math.round(data.risk_score * 100);
  riskPercentText.textContent = `${percent}% Risk`;
  gaugeBar.style.width = `${percent}%`;

  // Apply contextual category styling
  gaugeBar.className = "gauge-bar";
  categoryBadge.className = "badge";

  if (data.risk_category === "Low Risk") {
    gaugeBar.classList.add("risk-low");
    categoryBadge.classList.add("badge-low");
    categoryBadge.textContent = "🟢 Low Risk";
  } else if (data.risk_category === "Moderate Risk") {
    gaugeBar.classList.add("risk-moderate");
    categoryBadge.classList.add("badge-moderate");
    categoryBadge.textContent = "🟡 Moderate Risk";
  } else {
    gaugeBar.classList.add("risk-high");
    categoryBadge.classList.add("badge-high");
    categoryBadge.textContent = "🔴 High Risk";
  }

  plainWordsText.textContent = data.plain_words_summary;
  guidanceText.textContent = data.clinical_guidance;

  lifestyleList.innerHTML = "";
  data.lifestyle_recommendations.forEach(rec => {
    const li = document.createElement("li");
    li.textContent = rec;
    lifestyleList.appendChild(li);
  });

  // Smooth scroll into view on mobile
  if (window.innerWidth < 992) {
    resultContent.scrollIntoView({ behavior: "smooth" });
  }
}

// Fetch stats computed via pure hand-written SQL
async function fetchStats() {
  try {
    const res = await fetch("/stats");
    if (!res.ok) return;
    const stats = await res.json();

    document.getElementById("stat-total-requests").textContent = stats.total_requests;
    document.getElementById("stat-avg-risk").textContent = stats.average_predicted_risk_percent;
    document.getElementById("stat-high-risk-share").textContent = stats.share_of_high_risk_percent;
  } catch (err) {
    console.warn("Failed fetching SQL stats:", err);
  }
}

function showError(htmlMessage) {
  const alert = document.getElementById("form-error-alert");
  alert.innerHTML = htmlMessage;
  alert.style.display = "block";
}

function hideError() {
  const alert = document.getElementById("form-error-alert");
  alert.style.display = "none";
  alert.innerHTML = "";
}
