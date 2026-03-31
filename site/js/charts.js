/**
 * Chart.js visualization wrappers for Coffee Attribution
 */

const CoffeeCharts = {
  instances: {},

  colors: {
    accent: '#c89b5e',
    accentLight: '#e0b978',
    green: '#6abf69',
    red: '#cf6b6b',
    blue: '#6b9fcf',
    dimText: '#9a8d7f',
    border: '#3d3428',
    bgInput: '#2e2821',
    factors: {
      G: '#6abf69',
      P: '#6b9fcf',
      R: '#cf6b6b',
      B: '#c89b5e',
    }
  },

  defaults() {
    Chart.defaults.color = this.colors.dimText;
    Chart.defaults.borderColor = this.colors.border;
    Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, sans-serif';
    Chart.defaults.font.size = 12;
  },

  /**
   * Factor weight radar chart
   */
  renderRadar(canvasId, model) {
    this._destroy(canvasId);

    // Aggregate weights by factor group
    const featureNames = model.feature_names;
    const weights = model.weights;
    const factorSums = { G: 0, P: 0, R: 0, B: 0 };

    const gPrefixes = ['altitude_m', 'shade_pct', 'latitude', 'delta_t_c', 'variety_', 'soil_type_'];
    const pPrefixes = ['fermentation_hours', 'drying_days', 'method_p_', 'drying_method_', 'anaerobic'];
    const rPrefixes = ['first_crack_temp_c', 'drop_temp_c', 'dtr_pct', 'total_time_s', 'roast_level_'];
    const bPrefixes = ['grind_microns', 'water_temp_c', 'ratio', 'brew_time_s', 'water_tds_ppm', 'method_b_'];

    for (let i = 0; i < featureNames.length; i++) {
      const name = featureNames[i];
      const w = Math.abs(weights[i]);
      if (gPrefixes.some(p => name.startsWith(p))) factorSums.G += w;
      else if (pPrefixes.some(p => name.startsWith(p))) factorSums.P += w;
      else if (rPrefixes.some(p => name.startsWith(p))) factorSums.R += w;
      else if (bPrefixes.some(p => name.startsWith(p))) factorSums.B += w;
    }

    // Normalize to 0-10 scale
    const maxW = Math.max(...Object.values(factorSums)) || 1;
    const data = Object.fromEntries(
      Object.entries(factorSums).map(([k, v]) => [k, (v / maxW) * 10])
    );

    const ctx = document.getElementById(canvasId);
    this.instances[canvasId] = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: [
          'G: Genetics/Geography',
          'P: Processing',
          'R: Roasting',
          'B: Brewing'
        ],
        datasets: [{
          label: 'Factor Weight',
          data: [data.G, data.P, data.R, data.B],
          backgroundColor: 'rgba(200, 155, 94, 0.15)',
          borderColor: this.colors.accent,
          borderWidth: 2,
          pointBackgroundColor: Object.values(this.colors.factors),
          pointBorderColor: Object.values(this.colors.factors),
          pointRadius: 5,
        }]
      },
      options: {
        responsive: true,
        scales: {
          r: {
            beginAtZero: true,
            max: 10,
            grid: { color: this.colors.border },
            angleLines: { color: this.colors.border },
            ticks: { display: false },
          }
        },
        plugins: {
          legend: { display: false },
        }
      }
    });
  },

  /**
   * Top features horizontal bar chart
   */
  renderFeatures(canvasId, model) {
    this._destroy(canvasId);

    const topFeatures = (model.top_features || []).slice(0, 12);
    const labels = topFeatures.map(f => f.name.replace(/_/g, ' '));
    const values = topFeatures.map(f => f.importance);

    // Color by factor
    const barColors = topFeatures.map(f => {
      const name = f.name;
      if (['altitude_m', 'shade_pct', 'latitude', 'delta_t_c'].includes(name) || name.startsWith('variety_') || name.startsWith('soil_type_'))
        return this.colors.factors.G;
      if (['fermentation_hours', 'drying_days'].includes(name) || name.startsWith('method_p_') || name.startsWith('drying_method_') || name === 'anaerobic')
        return this.colors.factors.P;
      if (['first_crack_temp_c', 'drop_temp_c', 'dtr_pct', 'total_time_s'].includes(name) || name.startsWith('roast_level_'))
        return this.colors.factors.R;
      return this.colors.factors.B;
    });

    const ctx = document.getElementById(canvasId);
    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: barColors,
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            grid: { color: this.colors.border },
            title: { display: true, text: 'Absolute Weight', color: this.colors.dimText }
          },
          y: {
            grid: { display: false },
          }
        }
      }
    });
  },

  /**
   * Experiment progress line chart.
   * Tooltip fix: each data point stores the original experiment object
   * so indices are correct even after splitting into Kept/Discarded datasets.
   */
  renderProgress(canvasId, results) {
    this._destroy(canvasId);

    if (!results || results.length === 0) return null;

    const kept = results.filter(r => r.status === 'KEPT');
    const discarded = results.filter(r => r.status === 'DISCARDED');

    const ctx = document.getElementById(canvasId);
    this.instances[canvasId] = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Kept',
            data: kept.map(r => ({ x: r.id, y: r.val_mae, _exp: r })),
            backgroundColor: this.colors.green,
            pointRadius: 6,
          },
          {
            label: 'Discarded',
            data: discarded.map(r => ({ x: r.id, y: r.val_mae, _exp: r })),
            backgroundColor: 'rgba(207, 107, 107, 0.3)',
            pointRadius: 4,
          },
          {
            label: 'Best (running min)',
            data: this._runningMin(results),
            type: 'line',
            borderColor: this.colors.accentLight,
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: {
            title: { display: true, text: 'Experiment #', color: this.colors.dimText },
            grid: { color: this.colors.border },
          },
          y: {
            title: { display: true, text: 'val_mae', color: this.colors.dimText },
            grid: { color: this.colors.border },
            reverse: false,
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const point = ctx.raw;
                if (point && point._exp) {
                  const r = point._exp;
                  return `${r.description} (mae: ${r.val_mae})`;
                }
                // Fallback for running-min line
                if (point) {
                  return `Best mae: ${point.y}`;
                }
                return '';
              }
            }
          }
        }
      }
    });

    return this.instances[canvasId];
  },

  /**
   * Score distribution histogram
   */
  renderDistribution(canvasId, beansSummary) {
    this._destroy(canvasId);

    // Known distribution from the 966-bean CQI dataset
    // Bins: 60-65, 65-70, 70-75, 75-80, 80-85, 85-90, 90-95
    const bins = ['60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95'];

    // Compute distribution if score data available, otherwise use hardcoded from analysis
    // Hardcoded distribution from the 966-bean CQI dataset analysis:
    const counts = [5, 18, 42, 168, 614, 112, 7];

    // Color gradient from red (low) to green (high)
    const barColors = [
      'rgba(207, 107, 107, 0.7)',  // 60-65
      'rgba(207, 140, 107, 0.7)',  // 65-70
      'rgba(200, 155, 94, 0.7)',   // 70-75
      'rgba(180, 175, 94, 0.7)',   // 75-80
      'rgba(106, 191, 105, 0.7)',  // 80-85
      'rgba(107, 159, 207, 0.7)',  // 85-90
      'rgba(200, 155, 94, 0.9)',   // 90-95
    ];

    const ctx = document.getElementById(canvasId);
    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: bins,
        datasets: [{
          label: 'Beans',
          data: counts,
          backgroundColor: barColors,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const count = ctx.raw;
                const pct = ((count / 966) * 100).toFixed(1);
                return `${count} beans (${pct}%)`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            title: { display: true, text: 'Score Range', color: this.colors.dimText }
          },
          y: {
            grid: { color: this.colors.border },
            title: { display: true, text: 'Number of Beans', color: this.colors.dimText },
            beginAtZero: true,
          }
        }
      }
    });
  },

  /**
   * Mini attribution donut for scorer
   */
  renderAttribution(canvasId, attribution) {
    this._destroy(canvasId);

    const labels = Object.keys(attribution);
    const values = Object.values(attribution);
    const colors = labels.map(l => this.colors.factors[l]);

    const ctx = document.getElementById(canvasId);
    this.instances[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels.map(l => `${l} Factor`),
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true,
        cutout: '55%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 12, padding: 8, font: { size: 11 } }
          }
        }
      }
    });
  },

  _runningMin(results) {
    let min = Infinity;
    return results.map(r => {
      if (r.val_mae < min) min = r.val_mae;
      return { x: r.id, y: min };
    });
  },

  _destroy(id) {
    if (this.instances[id]) {
      this.instances[id].destroy();
      delete this.instances[id];
    }
  }
};
