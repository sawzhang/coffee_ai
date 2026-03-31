/**
 * Coffee Attribution AutoResearch — Main Application (API-backed with local fallback)
 */

(async function () {
  'use strict';

  // ── Initialize i18n ───────────────────────────────────
  I18n.init();

  CoffeeCharts.defaults();

  // ── Load Static Data ───────────────────────────────────
  const [model, results, beansSummary] = await Promise.all([
    DataLoader.getModel(),
    DataLoader.getResults(),
    DataLoader.getBeansSummary(),
  ]);

  // ── Header Badges ──────────────────────────────────────
  if (model) {
    document.getElementById('badge-mae').textContent =
      `val_mae: ${model.val_mae}` + (model.cv_mae ? ` (cv: ${model.cv_mae})` : '');
  }
  if (beansSummary) {
    document.getElementById('badge-beans').textContent = `beans: ${beansSummary.total_beans}`;
  }
  if (results) {
    document.getElementById('badge-experiments').textContent = `experiments: ${results.length}`;
  }

  // ── Radar + Features Charts ────────────────────────────
  if (model) {
    CoffeeCharts.renderRadar('chart-radar', model);
    CoffeeCharts.renderFeatures('chart-features', model);
  }

  // ── Experiment Progress ────────────────────────────────
  if (results && results.length > 0) {
    document.getElementById('no-experiments').style.display = 'none';
    CoffeeCharts.renderProgress('chart-progress', results);
  } else {
    document.getElementById('chart-progress').style.display = 'none';
  }

  // ── Top Beans ──────────────────────────────────────────
  if (beansSummary && beansSummary.top_beans) {
    renderBeanGrid('bean-grid', beansSummary.top_beans);
  }

  // ── Score Distribution ─────────────────────────────────
  if (beansSummary) {
    CoffeeCharts.renderDistribution('chart-distribution', beansSummary);
  }

  // ── Interactive Scorer (API-backed with local fallback) ─
  const rangeBindings = [
    ['input-altitude', 'val-altitude'],
    ['input-shade', 'val-shade'],
    ['input-delta-t', 'val-delta-t'],
    ['input-ferm', 'val-ferm'],
    ['input-dtr', 'val-dtr'],
    ['input-drop-temp', 'val-drop-temp'],
    ['input-water-temp', 'val-water-temp'],
    ['input-ratio', 'val-ratio'],
    ['input-grind', 'val-grind'],
    ['pref-acidity', 'val-pref-acidity'],
    ['pref-sweetness', 'val-pref-sweetness'],
    ['pref-complexity', 'val-pref-complexity'],
    ['pref-fermentation', 'val-pref-fermentation'],
    ['pref-body', 'val-pref-body'],
  ];

  for (const [inputId, outputId] of rangeBindings) {
    const input = document.getElementById(inputId);
    const output = document.getElementById(outputId);
    if (input && output) {
      input.addEventListener('input', () => {
        output.textContent = input.value;
        debouncedUpdateScore();
      });
    }
  }

  const selects = [
    'input-variety', 'input-country', 'input-soil', 'input-process', 'input-anaerobic',
    'input-drying', 'input-roast', 'input-brew'
  ];
  for (const id of selects) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', debouncedUpdateScore);
  }

  // Debounce API calls
  let scoreTimeout = null;
  function debouncedUpdateScore() {
    clearTimeout(scoreTimeout);
    scoreTimeout = setTimeout(updateScore, 200);
  }

  const COUNTRY_LATITUDES = {
    'Ethiopia': 8, 'Kenya': 0, 'Panama': 9, 'Colombia': 5,
    'Guatemala': 15, 'Costa Rica': 10, 'Brazil': -18,
    'Indonesia': -2, 'Yemen': 15, 'China': 22,
  };

  function getInputs() {
    return {
      variety: val('input-variety'),
      altitude_m: num('input-altitude'),
      soil_type: val('input-soil'),
      shade_pct: num('input-shade'),
      latitude: COUNTRY_LATITUDES[val('input-country')] || 8,
      delta_t_c: num('input-delta-t'),
      method_p: val('input-process'),
      anaerobic: val('input-anaerobic') === 'true',
      fermentation_hours: num('input-ferm'),
      drying_method: val('input-drying'),
      drying_days: 14,
      roast_level: val('input-roast'),
      first_crack_temp_c: 198,
      drop_temp_c: num('input-drop-temp'),
      dtr_pct: num('input-dtr'),
      total_time_s: 600,
      method_b: val('input-brew'),
      grind_microns: num('input-grind'),
      water_temp_c: num('input-water-temp'),
      ratio: num('input-ratio'),
      brew_time_s: 180,
      water_tds_ppm: 120,
    };
  }

  async function updateScore() {
    const inputs = getInputs();
    const result = await CoffeeScorer.predict(inputs);

    if (result) {
      document.getElementById('predicted-score').textContent = result.score.toFixed(1);
      const gradeText = (typeof I18n !== 'undefined')
        ? I18n.grade(result.score)
        : result.grade;
      document.getElementById('predicted-grade').textContent = gradeText;

      // Show prediction interval if available
      const intervalEl = document.getElementById('predicted-interval');
      if (intervalEl && result.score_low && result.score_high) {
        intervalEl.textContent = `${result.score_low.toFixed(1)} – ${result.score_high.toFixed(1)}`;
        intervalEl.style.display = '';
      } else if (intervalEl) {
        intervalEl.style.display = 'none';
      }

      if (result.attribution) {
        CoffeeCharts.renderAttribution('chart-attribution', result.attribution);
      }
    }
  }

  // Initial score
  updateScore();

  // ── Factor Recombination (Explore) ─────────────────────
  const exploreBtn = document.getElementById('btn-explore');
  if (exploreBtn) {
    exploreBtn.addEventListener('click', async () => {
      const inputs = getInputs();
      const gFactors = {
        variety: inputs.variety,
        altitude_m: inputs.altitude_m,
        country: val('input-country'),
        region: '',
        soil_type: inputs.soil_type,
        shade_pct: inputs.shade_pct,
        latitude: inputs.latitude,
        delta_t_c: inputs.delta_t_c,
      };

      exploreBtn.textContent = I18n.t('btn-explore-loading');
      exploreBtn.disabled = true;

      const result = await CoffeeScorer.explore(gFactors);

      exploreBtn.textContent = I18n.t('btn-explore');
      exploreBtn.disabled = false;

      if (result && result.results) {
        renderExploreResults(result);
      }
    });
  }

  function renderExploreResults(data) {
    const container = document.getElementById('explore-results');
    if (!container) return;

    container.innerHTML = `
      <div class="explore-header">
        <strong>${data.base_variety}</strong> from ${data.base_country}
        @ ${data.base_altitude}m — ${data.combinations} combinations tested
      </div>
      <div class="explore-grid">
        ${data.results.slice(0, 12).map((r, i) => `
          <div class="explore-card ${i === 0 ? 'explore-best' : ''}">
            <span class="explore-method">${r.method.replace(/_/g, ' ')}${r.anaerobic ? ' (anaerobic)' : ''}</span>
            <span class="explore-ferm">${r.fermentation_hours}h fermentation</span>
            <span class="explore-score">${r.predicted_score}</span>
            <span class="explore-grade">${r.grade}</span>
          </div>
        `).join('')}
      </div>
    `;
  }

  // ── User Preferences + Recommendations ─────────────────
  loadPrefs();

  const saveBtn = document.getElementById('btn-save-prefs');
  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      const prefs = {
        acidity: num('pref-acidity'),
        sweetness: num('pref-sweetness'),
        complexity: num('pref-complexity'),
        fermentation: num('pref-fermentation'),
        body: num('pref-body'),
      };
      localStorage.setItem('coffee_user_prefs', JSON.stringify(prefs));

      // Show status
      const status = document.getElementById('pref-status');
      status.textContent = 'Loading recommendations...';

      // Fetch recommendations
      const result = await CoffeeScorer.recommend(prefs, 10);
      if (result && result.beans) {
        renderRecommendations(result.beans);
        status.textContent = `Found ${result.beans.length} matches!`;
      } else {
        status.textContent = 'Saved!';
      }
      setTimeout(() => { status.textContent = ''; }, 3000);
    });
  }

  function renderRecommendations(beans) {
    const container = document.getElementById('recommend-results');
    if (!container) return;

    container.innerHTML = beans.map((b, i) => `
      <div class="bean-card">
        <div class="bean-name">${i + 1}. ${escapeHtml(b.name)}</div>
        <div class="bean-meta">
          ${escapeHtml(b.country)} · ${escapeHtml(b.variety)}
          · ${escapeHtml(b.process.replace(/_/g, ' '))}
          · ${b.altitude}m
        </div>
        <span class="bean-score">${b.predicted_score}</span>
        <span class="bean-match" style="margin-left:0.5rem;font-size:0.8rem;color:var(--green)">
          match: ${(b.pref_match * 100).toFixed(0)}%
        </span>
      </div>
    `).join('');
  }

  function loadPrefs() {
    const saved = localStorage.getItem('coffee_user_prefs');
    if (!saved) return;
    try {
      const prefs = JSON.parse(saved);
      setVal('pref-acidity', prefs.acidity);
      setVal('pref-sweetness', prefs.sweetness);
      setVal('pref-complexity', prefs.complexity);
      setVal('pref-fermentation', prefs.fermentation);
      setVal('pref-body', prefs.body);
    } catch (e) { /* ignore */ }
  }

  // ── Bean Grid Renderer ─────────────────────────────────
  function renderBeanGrid(containerId, beans) {
    const grid = document.getElementById(containerId);
    if (!grid) return;
    grid.innerHTML = beans.map(b => `
      <div class="bean-card">
        <div class="bean-name">${escapeHtml(b.name)}</div>
        <div class="bean-meta">
          ${escapeHtml(b.country)} · ${escapeHtml(b.variety)}<br>
          ${escapeHtml(b.process.replace(/_/g, ' '))} · ${escapeHtml(b.roast.replace(/_/g, ' '))}
        </div>
        <span class="bean-score">${b.score}</span>
      </div>
    `).join('');
  }

  // ── Helpers ────────────────────────────────────────────
  function val(id) { const el = document.getElementById(id); return el ? el.value : ''; }
  function num(id) { return parseFloat(val(id)) || 0; }
  function setVal(id, v) {
    const el = document.getElementById(id);
    if (el && v !== undefined) {
      el.value = v;
      const out = document.getElementById('val-' + id);
      if (out) out.textContent = v;
    }
  }
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
})();
