const state = {
  subjects: [],
  runId: null,
  examples: [],
  riskStudents: [],
};

const byId = (id) => document.getElementById(id);
const subjectSelect = byId("subject-select");
const analyzeButton = byId("analyze-button");
const pageStatus = byId("page-status");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function number(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function humanize(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function setStatus(message, isError = false) {
  pageStatus.textContent = message;
  pageStatus.classList.toggle("status-error", isError);
}

function showError(message) {
  const box = byId("error-box");
  box.textContent = message;
  box.classList.remove("hidden");
}

function clearError() {
  byId("error-box").classList.add("hidden");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function loadSubjects() {
  try {
    const payload = await requestJson("/api/subjects");
    state.subjects = payload.subjects;
    subjectSelect.replaceChildren();
    for (const subject of state.subjects) {
      const option = document.createElement("option");
      option.value = subject.column;
      option.textContent = subject.label;
      subjectSelect.append(option);
    }
    subjectSelect.value = "applied_data_science_marks";
    byId("subject-note").textContent = `${payload.students} fictional project records across five subjects.`;
    byId("dataset-warning").textContent = payload.note;
    setStatus("Choose a subject and run the analysis.");
  } catch (error) {
    setStatus("Could not load the project data.", true);
    showError(error.message);
  }
}

async function loadBigDataSummary() {
  const body = byId("semester-summary");
  try {
    const summary = await requestJson("/api/big-data");
    byId("bigdata-records").textContent = summary.records.toLocaleString();
    byId("bigdata-batches").textContent = summary.map_batches.toLocaleString();
    body.replaceChildren();
    for (const item of summary.semesters) {
      const row = document.createElement("tr");
      for (const value of [
        `Semester ${item.semester}`, item.students, item.at_risk,
        percent(item.at_risk_rate), number(item.mean_average, 1), `${number(item.mean_attendance, 1)}%`,
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      body.append(row);
    }
    byId("bigdata-note").textContent = `${summary.execution} ${summary.batch_size} records per map batch; risk cutoff ${summary.risk_cutoff}/100.`;
  } catch (error) {
    byId("bigdata-records").textContent = "Unavailable";
    byId("bigdata-note").textContent = error.message;
  }
}

function renderModelTable(models) {
  const body = byId("model-results");
  body.replaceChildren();
  for (const model of models) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(model.model)}</td>
      <td>${number(model.Accuracy)}</td><td>${number(model.Precision)}</td>
      <td>${number(model.Recall)}</td><td>${number(model.F1)}</td>
      <td>${number(model["ROC AUC"])}</td>
      <td>${number(model.demographic_parity_difference)}</td>
      <td>${number(model.equal_opportunity_difference)}</td>
      <td>${number(model.train_seconds, 3)}</td>
      <td>${number(model.prediction_ms_per_student, 4)}</td>`;
    body.append(row);
  }
}

function renderRiskStudents() {
  const body = byId("risk-list-body");
  const query = byId("risk-search").value.trim().toLowerCase();
  const filtered = state.riskStudents.filter((student) =>
    `${student.student_name} ${student.student_id}`.toLowerCase().includes(query)
  );
  body.replaceChildren();

  const count = byId("risk-count");
  count.textContent = query
    ? `Showing ${filtered.length} of ${state.riskStudents.length} flagged students.`
    : `${state.riskStudents.length} students flagged in the test group.`;

  if (!filtered.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.className = "empty-cell";
    cell.textContent = state.riskStudents.length
      ? "No matching student names. Try a shorter search."
      : "No students crossed this alert threshold in the test group.";
    row.append(cell);
    body.append(row);
    return;
  }

  for (const student of filtered) {
    const row = document.createElement("tr");
    const cells = [
      [student.student_name, "student-name-cell"],
      [student.student_id, "student-id-cell"],
      [student.gender, ""],
      [`Semester ${student.semester}`, ""],
      [`${number(student.subject_mark, 1)} / 100`, "mark-cell"],
      [`${number(student.overall_average, 1)} / 100`, ""],
      [percent(student.risk_probability), "risk-cell"],
    ];
    for (const [value, className] of cells) {
      const cell = document.createElement("td");
      cell.textContent = value ?? "—";
      if (className) cell.className = className;
      row.append(cell);
    }
    body.append(row);
  }
}

function renderFairness(payload) {
  const cards = byId("fairness-summary");
  cards.replaceChildren();
  const rf = payload.models.find((model) => model.model === "Random Forest");
  const fair = payload.fairness_aware;
  const items = [
    ["Random Forest · demographic parity gap", rf?.demographic_parity_difference],
    ["Random Forest · equal opportunity gap", rf?.equal_opportunity_difference],
  ];
  if (fair && !fair.error) {
    items.push(["Adjusted · demographic parity gap", fair.fairness.demographic_parity_difference]);
    items.push(["Adjusted · equal opportunity gap", fair.fairness.equal_opportunity_difference]);
  }
  for (const [label, value] of items) {
    const card = document.createElement("div");
    card.className = "fairness-card";
    card.innerHTML = `<span>${escapeHtml(label)}</span><strong>${number(value)}</strong>`;
    cards.append(card);
  }

  const body = byId("group-rates");
  body.replaceChildren();
  const baseline = payload.baseline_group_metrics.map((item) => ({ ...item, Rule: "Baseline" }));
  const adjusted = fair && !fair.error
    ? fair.group_metrics.map((item) => ({ ...item, Rule: "Equal opportunity" }))
    : [];
  for (const values of [...baseline, ...adjusted]) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(values.Group)}</td><td>${escapeHtml(values.Rule)}</td>
      <td>${percent(values["Selection rate"])}</td>
      <td>${percent(values["True positive rate"])}</td>
      <td>${escapeHtml(values["Test examples"] ?? "—")}</td>
      <td>${escapeHtml(values["At-risk examples"] ?? "—")}</td>`;
    body.append(row);
  }
  if (fair?.error) showError(`The fairness adjustment was unavailable for this subject split: ${fair.error}`);
}

function renderExamples(examples) {
  state.examples = examples;
  const select = byId("example-select");
  select.replaceChildren();
  examples.forEach((example) => {
    const option = document.createElement("option");
    option.value = example.position;
    option.textContent = `${example.label} · actual ${example.actual_at_risk ? "at risk" : "not at risk"}`;
    select.append(option);
  });
}

function renderSummary(summary) {
  byId("summary-students").textContent = summary.students.toLocaleString();
  byId("summary-risk-rate").textContent = `${percent(summary.at_risk_rate)} · ${summary.observed_at_risk_count} students`;
  byId("summary-predictors").textContent = summary.predictors;
  byId("summary-group").textContent = humanize(summary.audit_column);
}

async function runAnalysis(event) {
  event.preventDefault();
  clearError();
  analyzeButton.disabled = true;
  setStatus("Training the three models for this subject…");
  try {
    const payload = await requestJson("/api/analyze", {
      method: "POST",
      body: JSON.stringify({
        subject_column: subjectSelect.value,
        grade_cutoff: Number(byId("grade-cutoff").value),
        alert_threshold: Number(byId("alert-threshold").value),
      }),
    });
    state.runId = payload.run_id;
    state.riskStudents = payload.risk_students;
    byId("risk-heading").textContent = `Students to review · ${payload.subject.label}`;
    byId("risk-subtitle").textContent = `Random Forest predictions for ${payload.subject.label}, using the ${payload.summary.grade_cutoff}/100 at-risk cutoff.`;
    renderSummary(payload.summary);
    renderRiskStudents();
    renderModelTable(payload.models);
    renderFairness(payload);
    renderExamples(payload.examples);
    byId("download-results").href = `/api/results/${payload.run_id}.csv`;
    byId("download-results").classList.remove("hidden");
    byId("download-risk-list").href = `/api/at-risk/${payload.run_id}.csv`;
    byId("download-risk-list").classList.remove("hidden");
    byId("results").classList.remove("hidden");
    setStatus("Subject comparison complete.");
    await explainExample();
  } catch (error) {
    setStatus("Subject comparison could not be completed.", true);
    showError(error.message);
  } finally {
    analyzeButton.disabled = false;
  }
}

async function explainExample() {
  if (!state.runId || !state.examples.length) return;
  const button = byId("explain-button");
  button.disabled = true;
  byId("shap-chart").textContent = "Creating the explanation…";
  try {
    const payload = await requestJson("/api/explain", {
      method: "POST",
      body: JSON.stringify({
        run_id: state.runId,
        model: byId("explain-model").value,
        sample_position: Number(byId("example-select").value),
      }),
    });
    byId("prediction-summary").innerHTML = `
      <span class="prediction-pill"><strong>${escapeHtml(payload.student_name)}</strong></span>
      <span class="prediction-pill">Risk chance: <strong>${percent(payload.risk_probability)}</strong></span>
      <span class="prediction-pill">Observed subject mark: <strong>${number(payload.actual_score, 1)} / 100</strong></span>
      <span class="prediction-pill">Alert: <strong>${payload.alert ? "Yes" : "No"}</strong></span>
      <span class="prediction-pill">Gender group: <strong>${escapeHtml(payload.gender)}</strong></span>`;
    const profile = byId("student-profile");
    profile.replaceChildren();
    for (const [key, value] of Object.entries(payload.profile)) {
      const item = document.createElement("div");
      item.innerHTML = `<dt>${escapeHtml(humanize(key))}</dt><dd>${escapeHtml(value)}</dd>`;
      profile.append(item);
    }
    renderShap(payload.contributions);
  } catch (error) {
    byId("shap-chart").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function renderShap(contributions) {
  const chart = byId("shap-chart");
  chart.replaceChildren();
  if (!contributions.length) {
    chart.textContent = "No explanation values were returned.";
    return;
  }
  const maxValue = Math.max(...contributions.map((item) => Math.abs(item["SHAP contribution"])), 1e-9);
  for (const item of [...contributions].reverse()) {
    const value = Number(item["SHAP contribution"]);
    const row = document.createElement("div");
    row.className = "shap-row";
    const width = `${Math.max(2, Math.abs(value) / maxValue * 48)}%`;
    const bar = value >= 0
      ? `<span class="shap-bar positive" style="width:${width}"></span>`
      : `<span class="shap-bar negative" style="width:${width}"></span>`;
    row.innerHTML = `<span class="shap-name" title="${escapeHtml(item.Feature)}">${escapeHtml(humanize(item.Feature))}</span><span class="shap-track">${bar}</span><span class="shap-value">${number(value, 3)}</span>`;
    chart.append(row);
  }
}

byId("analysis-form").addEventListener("submit", runAnalysis);
byId("explain-button").addEventListener("click", explainExample);
byId("explain-model").addEventListener("change", explainExample);
byId("example-select").addEventListener("change", explainExample);
byId("risk-search").addEventListener("input", renderRiskStudents);

loadSubjects();
loadBigDataSummary();
