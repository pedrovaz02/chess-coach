/* Chess Coach frontend
   Submits a Lichess username to /recommend/{username} and renders the result. */

const form = document.getElementById("lookup-form");
const usernameInput = document.getElementById("username");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const username = usernameInput.value.trim();
  if (!username) return;

  setLoading(true);
  resultsEl.classList.add("hidden");

  try {
    const res = await fetch(`/recommend/${encodeURIComponent(username)}`);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      const detail = data.detail || `Server responded ${res.status}`;
      throw new Error(detail);
    }
    const data = await res.json();
    render(data);
    showStatus(null);
    resultsEl.classList.remove("hidden");
  } catch (err) {
    showStatus(err.message || String(err), "error");
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? "Fetching games…" : "Recommend";
  if (isLoading) {
    showStatus("Fetching games and computing your playstyle… this takes 5–10 seconds.", "loading");
  }
}

function showStatus(message, kind = null) {
  if (!message) {
    statusEl.classList.add("hidden");
    statusEl.textContent = "";
    return;
  }
  statusEl.classList.remove("hidden", "loading", "error");
  if (kind) statusEl.classList.add(kind);
  statusEl.textContent = message;
}

function render(data) {
  document.getElementById("cluster-id").textContent = `C${data.cluster.id}`;
  document.getElementById("cluster-name").textContent = data.cluster.name;
  document.getElementById("cluster-blurb").textContent = data.cluster.blurb;
  document.getElementById("user-rating").textContent = Math.round(data.user_rating);
  document.getElementById("cluster-rating").textContent = Math.round(data.cluster.avg_rating);
  document.getElementById("cluster-size").textContent = `${data.cluster.size.toLocaleString()} players`;
  document.getElementById("n-games").textContent = data.n_games_used;

  const acplEl = document.getElementById("cluster-acpl");
  if (data.cluster.accuracy) {
    const acpl = Math.round(data.cluster.accuracy.avg_acpl);
    const blunder = (data.cluster.accuracy.avg_blunder_rate * 100).toFixed(1);
    acplEl.textContent = `${acpl} cp loss · ${blunder}% blunders`;
  } else {
    acplEl.textContent = "—";
  }

  renderOpenings(document.getElementById("white-openings"), data.top_openings.white);
  renderOpenings(document.getElementById("black-openings"), data.top_openings.black);
  renderComparison(data.feature_comparison);
}

function renderOpenings(listEl, openings) {
  listEl.innerHTML = "";
  if (!openings.length) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="empty">No openings cleared the sample-size threshold for this cluster.</span>`;
    listEl.appendChild(li);
    return;
  }
  openings.forEach((op, idx) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="rank">${idx + 1}</span>
      <span class="name">${escapeHtml(op.name)}</span>
      <span class="sample">${op.n.toLocaleString()} games</span>
    `;
    listEl.appendChild(li);
  });
}

function renderComparison(rows) {
  const tbody = document.querySelector("#comparison-table tbody");
  tbody.innerHTML = "";

  // Sort by magnitude of delta so the most distinctive features come first
  const sorted = [...rows].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  sorted.forEach((row) => {
    const tr = document.createElement("tr");
    const deltaClass = row.delta > 0.001 ? "delta-pos" : row.delta < -0.001 ? "delta-neg" : "";
    const sign = row.delta > 0 ? "+" : "";
    tr.innerHTML = `
      <td>${row.feature}</td>
      <td>${fmt(row.user)}</td>
      <td>${fmt(row.cluster_mean)}</td>
      <td class="${deltaClass}">${sign}${fmt(row.delta)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function fmt(x) {
  if (Math.abs(x) >= 10) return x.toFixed(1);
  return x.toFixed(3);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

/* ── Tab switching ─────────────────────────────────────────────────── */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.tab;
    document.getElementById("panel-recommend").classList.toggle("hidden", target !== "recommend");
    document.getElementById("panel-versus").classList.toggle("hidden", target !== "versus");
  });
});

/* ── Versus: compare two players ───────────────────────────────────── */

const vsForm = document.getElementById("versus-form");
const vsBtn = document.getElementById("vs-btn");
const vsStatus = document.getElementById("vs-status");
const vsResults = document.getElementById("vs-results");

vsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const a = document.getElementById("vs-a").value.trim();
  const b = document.getElementById("vs-b").value.trim();
  if (!a || !b) return;

  vsBtn.disabled = true;
  vsBtn.textContent = "Fetching…";
  vsResults.classList.add("hidden");
  vsShowStatus(`Fetching games for ${a} and ${b}… this takes 10–20 seconds.`, "loading");

  try {
    const res = await fetch(`/versus/${encodeURIComponent(a)}/${encodeURIComponent(b)}`);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `Server responded ${res.status}`);
    }
    renderVersus(await res.json());
    vsShowStatus(null);
    vsResults.classList.remove("hidden");
  } catch (err) {
    vsShowStatus(err.message || String(err), "error");
  } finally {
    vsBtn.disabled = false;
    vsBtn.textContent = "Compare";
  }
});

function vsShowStatus(message, kind) {
  if (!message) { vsStatus.classList.add("hidden"); vsStatus.textContent = ""; return; }
  vsStatus.classList.remove("hidden", "loading", "error");
  if (kind) vsStatus.classList.add(kind);
  vsStatus.textContent = message;
}

function renderVersus(d) {
  const a = d.player_a, b = d.player_b;

  document.getElementById("vs-a-id").textContent = `C${a.cluster.id}`;
  document.getElementById("vs-a-name").textContent = `${a.username} (${Math.round(a.rating)}) — ${a.cluster.name}`;
  document.getElementById("vs-a-cluster").textContent = a.cluster.blurb;
  document.getElementById("vs-b-id").textContent = `C${b.cluster.id}`;
  document.getElementById("vs-b-name").textContent = `${b.username} (${Math.round(b.rating)}) — ${b.cluster.name}`;
  document.getElementById("vs-b-cluster").textContent = b.cluster.blurb;

  renderMatchup("vs-match1-title", "vs-match1",
    `${a.username} (White) vs ${b.username} (Black)`, d.matchups.a_white_b_black);
  renderMatchup("vs-match2-title", "vs-match2",
    `${b.username} (White) vs ${a.username} (Black)`, d.matchups.b_white_a_black);

  // Feature comparison table
  document.getElementById("vs-th-a").textContent = a.username;
  document.getElementById("vs-th-b").textContent = b.username;
  const tbody = document.querySelector("#vs-table tbody");
  tbody.innerHTML = "";
  d.feature_comparison.forEach((row) => {
    const tr = document.createElement("tr");
    const dCls = row.delta > 0.001 ? "delta-pos" : row.delta < -0.001 ? "delta-neg" : "";
    tr.innerHTML = `
      <td>${row.feature}</td>
      <td>${fmt(row.a)}</td>
      <td>${fmt(row.b)}</td>
      <td class="${dCls}">${row.delta > 0 ? "+" : ""}${fmt(row.delta)}</td>`;
    tbody.appendChild(tr);
  });
}

function renderMatchup(titleId, listId, title, openings) {
  document.getElementById(titleId).textContent = title;
  const listEl = document.getElementById(listId);
  listEl.innerHTML = "";
  if (!openings || !openings.length) {
    listEl.innerHTML = `<li><span class="empty">no shared opening data</span></li>`;
    return;
  }
  openings.forEach((op) => {
    const li = document.createElement("li");
    const mark = op.both_positive ? ' <span class="mutual">✓ both</span>' : "";
    li.innerHTML = `
      <span class="name">${escapeHtml(op.family)}${mark}</span>
      <span class="sample">${op.white_rel >= 0 ? "+" : ""}${op.white_rel.toFixed(3)} / ${op.black_rel >= 0 ? "+" : ""}${op.black_rel.toFixed(3)}</span>`;
    listEl.appendChild(li);
  });
}
