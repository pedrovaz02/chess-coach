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
  document.getElementById("cluster-size").textContent = `${data.cluster.size} players`;
  document.getElementById("n-games").textContent = data.n_games_used;

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
      <span class="eco">${op.eco}</span>
      <span class="name">${escapeHtml(op.name)}</span>
      <span class="sample">${op.n} games</span>
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
