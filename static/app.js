const byId = (id) => document.getElementById(id);
const state = { role: null, students: [] };

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function showOnly(screen) {
  for (const id of ["login-screen", "admin-dashboard", "dashboard-app"]) {
    byId(id).classList.toggle("hidden", id !== screen);
  }
}

function showLoginMessage(id, message) {
  byId(id).textContent = message;
}

function setHeader(role, label = "") {
  byId("student-session").classList.toggle("hidden", !role);
  byId("signed-in-label").textContent = label;
  byId("return-admin-button").classList.toggle("hidden", role !== "admin-viewing-student");
}

async function handleStudentLogin(event) {
  event.preventDefault();
  const button = byId("login-button");
  button.disabled = true;
  showLoginMessage("login-status", "Signing in…");
  try {
    const payload = await requestJson("/api/login", {
      method: "POST",
      body: JSON.stringify({
        student_id: byId("login-student-id").value,
        password: byId("login-password").value,
      }),
    });
    state.role = "student";
    setHeader("student", `${payload.student.student_name} · ${payload.student.student_id}`);
    await loadPerformance(false);
  } catch (error) {
    showLoginMessage("login-status", error.message);
  } finally {
    button.disabled = false;
  }
}

async function handleAdminLogin(event) {
  event.preventDefault();
  const button = byId("admin-login-button");
  button.disabled = true;
  showLoginMessage("admin-login-status", "Signing in…");
  try {
    await requestJson("/api/admin-login", {
      method: "POST",
      body: JSON.stringify({
        username: byId("admin-username").value,
        password: byId("admin-password").value,
      }),
    });
    state.role = "admin";
    setHeader("admin", "Administrator");
    await loadAdminOverview();
  } catch (error) {
    showLoginMessage("admin-login-status", error.message);
  } finally {
    button.disabled = false;
  }
}

function formatScore(value) {
  return value === null || value === undefined ? "—" : `${Number(value).toFixed(1)}%`;
}

async function loadAdminOverview() {
  showOnly("admin-dashboard");
  byId("admin-error").classList.add("hidden");
  try {
    const data = await requestJson("/api/admin-overview");
    state.students = data.students;
    byId("admin-total").textContent = data.total_students;
    byId("admin-average").textContent = formatScore(data.class_average);
    byId("admin-topper").textContent = data.topper ? data.topper.student_name : "—";
    renderStudentRows(state.students);
  } catch (error) {
    byId("admin-error").textContent = error.message;
    byId("admin-error").classList.remove("hidden");
  }
}

function renderStudentRows(students) {
  const tbody = byId("admin-student-list");
  tbody.replaceChildren();
  for (const student of students) {
    const row = document.createElement("tr");
    for (const value of [student.student_name, student.student_id, `Semester ${student.semester}`, formatScore(student.overall_average)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    const actionCell = document.createElement("td");
    const button = document.createElement("button");
    button.className = "button button-secondary button-small";
    button.type = "button";
    button.textContent = "Open tracking";
    button.addEventListener("click", () => openStudent(student));
    actionCell.append(button);
    row.append(actionCell);
    tbody.append(row);
  }
}

async function openStudent(student) {
  try {
    await requestJson("/api/admin/view-student", {
      method: "POST", body: JSON.stringify({ student_id: student.student_id }),
    });
    setHeader("admin-viewing-student", `Admin · viewing ${student.student_name} (${student.student_id})`);
    byId("student-view-kicker").textContent = "ADMIN · INDIVIDUAL STUDENT VIEW";
    byId("student-view-title").textContent = `${student.student_name}'s performance`;
    byId("student-view-description").textContent = "You are viewing this student's record. Use Admin overview to return to the class list.";
    await loadPerformance(true);
  } catch (error) {
    byId("admin-error").textContent = error.message;
    byId("admin-error").classList.remove("hidden");
  }
}

async function loadPerformance(adminView) {
  showOnly("dashboard-app");
  byId("performance-error").classList.add("hidden");
  try {
    const payload = await requestJson("/api/my-performance");
    const student = payload.student;
    byId("personal-student-id").textContent = `${student.student_name} · ${student.student_id}`;
    byId("personal-average").textContent = formatScore(student.overall_average);
    byId("personal-semester").textContent = student.semester;
    byId("personal-attendance").textContent = formatScore(student.attendance_percent);
    byId("personal-study-hours").textContent = `${student.weekly_study_hours} hours`;
    byId("personal-assignments").textContent = formatScore(student.assignment_completion_percent);
    byId("personal-labs").textContent = formatScore(student.lab_completion_percent);
    byId("personal-participation").textContent = `${student.class_participation_score} / 5`;
    const marks = byId("personal-subject-marks");
    marks.replaceChildren();
    for (const subject of student.subject_marks) {
      const item = document.createElement("div");
      item.className = "subject-mark-row";
      const title = document.createElement("span");
      title.textContent = subject.subject;
      const track = document.createElement("div");
      track.className = "subject-mark-track";
      const fill = document.createElement("span");
      fill.style.width = `${Math.max(0, Math.min(100, Number(subject.mark) || 0))}%`;
      track.append(fill);
      const score = document.createElement("strong");
      score.textContent = `${Number(subject.mark).toFixed(1)} / 100`;
      item.append(title, track, score);
      marks.append(item);
    }
    byId("personal-record-note").textContent = payload.note;
    if (!adminView) {
      byId("student-view-kicker").textContent = "YOUR STUDENT RECORD";
      byId("student-view-title").textContent = "Your performance";
      byId("student-view-description").textContent = "This view shows only the record linked to your sign-in.";
    }
  } catch (error) {
    byId("performance-error").textContent = error.message;
    byId("performance-error").classList.remove("hidden");
  }
}

async function returnToAdmin() {
  try {
    await requestJson("/api/admin/return", { method: "POST", body: "{}" });
    setHeader("admin", "Administrator");
    await loadAdminOverview();
  } catch (error) {
    byId("performance-error").textContent = error.message;
    byId("performance-error").classList.remove("hidden");
  }
}

async function signOut() {
  try { await requestJson("/api/logout", { method: "POST", body: "{}" }); } catch (_) { /* Reset the visible session either way. */ }
  state.role = null;
  setHeader(null);
  byId("login-form").reset();
  byId("admin-login-form").reset();
  showLoginMessage("login-status", "");
  showLoginMessage("admin-login-status", "");
  showOnly("login-screen");
}

async function restoreSession() {
  try {
    const data = await requestJson("/api/session");
    if (!data.authenticated) return;
    if (data.role === "admin") {
      state.role = "admin";
      if (data.view_student_id) {
        const student = state.students.find((item) => item.student_id === data.view_student_id);
        setHeader("admin-viewing-student", `Admin · viewing ${data.view_student_id}`);
        byId("student-view-kicker").textContent = "ADMIN · INDIVIDUAL STUDENT VIEW";
        byId("student-view-title").textContent = student ? `${student.student_name}'s performance` : "Student performance";
        byId("student-view-description").textContent = "You are viewing this student's record. Use Admin overview to return to the class list.";
        await loadPerformance(true);
      } else {
        setHeader("admin", "Administrator");
        await loadAdminOverview();
      }
    } else if (data.role === "student") {
      state.role = "student";
      setHeader("student", `${data.student.student_name} · ${data.student.student_id}`);
      await loadPerformance(false);
    }
  } catch (_) { /* The login page remains available if session restore fails. */ }
}

byId("login-form").addEventListener("submit", handleStudentLogin);
byId("admin-login-form").addEventListener("submit", handleAdminLogin);
byId("logout-button").addEventListener("click", signOut);
byId("return-admin-button").addEventListener("click", returnToAdmin);
byId("student-search").addEventListener("input", (event) => {
  const term = event.target.value.trim().toLowerCase();
  renderStudentRows(state.students.filter((student) =>
    student.student_name.toLowerCase().includes(term) || student.student_id.toLowerCase().includes(term)
  ));
});
restoreSession();
