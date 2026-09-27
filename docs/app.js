document.addEventListener("DOMContentLoaded", () => {
  // --- 1. Pooling Playground ---
  const poolingSelect = document.getElementById("pooling-select");
  const paddingSelect = document.getElementById("padding-style");
  const tokensMatrix = document.getElementById("tokens-matrix");
  const maskDisplay = document.getElementById("mask-display");
  const pooledVector = document.getElementById("pooled-vector");

  const sampleTokens = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [99.0, 99.0, 99.0, 99.0],
  ];

  function getMask(pattern) {
    if (pattern === "right") return [1, 1, 1, 0];
    if (pattern === "left") return [0, 0, 1, 1];
    return [1, 1, 1, 1];
  }

  function computePooled(tokens, mask, mode) {
    const dim = 4;
    const res = [0, 0, 0, 0];

    if (mode === "mean") {
      let sumMask = 0;
      for (let s = 0; s < 4; s++) sumMask += mask[s];
      const denom = Math.max(sumMask, 1e-9);

      for (let d = 0; d < dim; d++) {
        let sum = 0;
        for (let s = 0; s < 4; s++) {
          sum += tokens[s][d] * mask[s];
        }
        res[d] = sum / denom;
      }
    } else if (mode === "cls") {
      let first = 0;
      for (let s = 0; s < 4; s++) {
        if (mask[s] > 0) {
          first = s;
          break;
        }
      }
      for (let d = 0; d < dim; d++) res[d] = tokens[first][d];
    } else if (mode === "max") {
      for (let d = 0; d < dim; d++) {
        let maxVal = -Infinity;
        for (let s = 0; s < 4; s++) {
          if (mask[s] > 0 && tokens[s][d] > maxVal) {
            maxVal = tokens[s][d];
          }
        }
        res[d] = isFinite(maxVal) ? maxVal : 0;
      }
    } else if (mode === "lasttoken") {
      let last = 0;
      for (let s = 3; s >= 0; s--) {
        if (mask[s] > 0) {
          last = s;
          break;
        }
      }
      for (let d = 0; d < dim; d++) res[d] = tokens[last][d];
    }

    return res;
  }

  function updatePooling() {
    const mode = poolingSelect.value;
    const pattern = paddingSelect.value;
    const mask = getMask(pattern);
    const pooled = computePooled(sampleTokens, mask, mode);

    // Render tokens
    tokensMatrix.innerHTML = "";
    sampleTokens.forEach((row, i) => {
      const rowDiv = document.createElement("div");
      const isAttended = mask[i] > 0;
      rowDiv.className = `matrix-row ${isAttended ? 'attended' : 'masked'}`;

      let label = `t${i}: `;
      rowDiv.innerHTML = `<span style="color: #64748b; width: 35px;">${label}</span>` +
        row.map(v => `<span class="token-cell">${v.toFixed(1)}</span>`).join("");
      tokensMatrix.appendChild(rowDiv);
    });

    // Render mask
    maskDisplay.innerHTML = `<div class="mask-chips">` +
      mask.map((m, i) => `<span class="mask-chip ${m ? 'valid' : 'pad'}">t${i}: ${m ? 'Attended (1)' : 'Pad (0)'}</span>`).join("") +
      `</div>`;

    // Render pooled
    pooledVector.innerHTML = `<strong>Pooled [1×4]:</strong> [${pooled.map(v => v.toFixed(2)).join(", ")}]`;
  }

  poolingSelect.addEventListener("change", updatePooling);
  paddingSelect.addEventListener("change", updatePooling);
  updatePooling();

  // --- 2. ColBERT MaxSim Visualizer ---
  const queryTokens = ["quantum", "computing", "silicon"];
  const docTokens = ["apple", "silicon", "m4", "quantum", "accelerator"];

  // Synthetic similarity dot products
  const simMatrix = [
    [0.12, 0.25, 0.18, 0.94, 0.38], // "quantum"
    [0.15, 0.42, 0.35, 0.52, 0.88], // "computing"
    [0.65, 0.98, 0.58, 0.22, 0.45], // "silicon"
  ];

  function renderMaxSim() {
    const table = document.getElementById("maxsim-table");
    const breakdown = document.getElementById("maxsim-breakdown");
    const totalVal = document.getElementById("maxsim-total");

    let tableHtml = "<thead><tr><th>Query \\ Doc</th>";
    docTokens.forEach(d => {
      tableHtml += `<th>"${d}"</th>`;
    });
    tableHtml += "<th>Max Sim</th></tr></thead><tbody>";

    let totalScore = 0;
    breakdown.innerHTML = "";

    queryTokens.forEach((q, qi) => {
      tableHtml += `<tr><th>"${q}"</th>`;
      let rowMax = -Infinity;
      let maxDocIdx = 0;

      simMatrix[qi].forEach((val, di) => {
        if (val > rowMax) {
          rowMax = val;
          maxDocIdx = di;
        }
      });

      simMatrix[qi].forEach((val, di) => {
        const isMax = di === maxDocIdx;
        tableHtml += `<td class="${isMax ? 'max-cell' : ''}">${val.toFixed(2)}</td>`;
      });

      tableHtml += `<td style="font-weight: 700; color: #34d399;">${rowMax.toFixed(2)}</td></tr>`;
      totalScore += rowMax;

      const li = document.createElement("li");
      li.innerHTML = `max("${q}" &middot; Doc) = <strong>${rowMax.toFixed(2)}</strong> (matched on <em>"${docTokens[maxDocIdx]}"</em>)`;
      breakdown.appendChild(li);
    });

    tableHtml += "</tbody>";
    table.innerHTML = tableHtml;
    totalVal.textContent = totalScore.toFixed(3);
  }

  renderMaxSim();

  // --- 3. Semantic Search & Cosine Similarity ---
  const searchInput = document.getElementById("search-query");
  const searchResults = document.getElementById("search-results");

  const sampleCorpus = [
    "Apple M-series chips feature high-bandwidth unified memory architecture.",
    "BGE and ModernBERT provide state-of-the-art sentence embedding representations.",
    "ColBERT performs token-level late interaction scoring via MaxSim.",
    "Making artisan sourdough bread with whole wheat sourdough starter.",
    "Metal Performance Shaders accelerated GPU matrix multiplication."
  ];

  function computeSemanticScore(query, doc) {
    const qWords = query.toLowerCase().split(/\W+/).filter(Boolean);
    const dWords = doc.toLowerCase().split(/\W+/).filter(Boolean);

    let matchCount = 0;
    qWords.forEach(qw => {
      if (dWords.some(dw => dw.includes(qw) || qw.includes(dw))) {
        matchCount += 1;
      }
    });

    // Base score with cosine-like characteristic
    let score = (matchCount / Math.max(qWords.length, 1)) * 0.7 + 0.25;
    return Math.min(0.98, Math.max(0.12, score));
  }

  function updateSearch() {
    const query = searchInput.value;
    const scored = sampleCorpus.map((doc, idx) => ({
      doc,
      score: computeSemanticScore(query, doc)
    }));

    scored.sort((a, b) => b.score - a.score);

    searchResults.innerHTML = "";
    scored.forEach((item, rank) => {
      const row = document.createElement("div");
      row.className = "corpus-item";
      const pct = Math.round(item.score * 100);

      row.innerHTML = `
        <span class="rank-badge">#${rank + 1}</span>
        <span class="corpus-text">${item.doc}</span>
        <div class="sim-score-bar-wrap">
          <span class="sim-val">${item.score.toFixed(3)}</span>
          <div class="sim-bar-bg">
            <div class="sim-bar-fill" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
      searchResults.appendChild(row);
    });
  }

  searchInput.addEventListener("input", updateSearch);
  updateSearch();
});
