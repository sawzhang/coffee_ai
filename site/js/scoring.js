/**
 * Coffee Scorer — API-backed prediction engine.
 * Calls /api/predict, /api/recommend, /api/explore on the backend.
 */

const CoffeeScorer = {
  API_BASE: '/api',

  async predict(inputs) {
    const body = this._buildRequest(inputs);
    try {
      const resp = await fetch(`${this.API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      console.warn('Predict API error:', e);
      return null;
    }
  },

  async recommend(prefs, topK = 10) {
    try {
      const resp = await fetch(`${this.API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prefs, top_k: topK }),
      });
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      console.warn('Recommend API error:', e);
      return null;
    }
  },

  async explore(gFactors, options = {}) {
    const body = {
      G: gFactors,
      vary_methods: options.methods || ['washed', 'natural', 'honey_yellow', 'honey_red'],
      vary_anaerobic: options.anaerobic || [false, true],
      vary_fermentation: options.fermentation || [24, 48, 72, 120],
    };
    try {
      const resp = await fetch(`${this.API_BASE}/explore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      console.warn('Explore API error:', e);
      return null;
    }
  },

  _buildRequest(inputs) {
    return {
      G: {
        variety: inputs.variety || 'Bourbon',
        altitude_m: inputs.altitude_m || 1600,
        country: inputs.country || 'Colombia',
        region: inputs.region || '',
        soil_type: inputs.soil_type || 'volcanic',
        shade_pct: inputs.shade_pct || 30,
        latitude: inputs.latitude || 8,
        delta_t_c: inputs.delta_t_c || 12,
      },
      P: {
        method: inputs.method_p || 'washed',
        anaerobic: inputs.anaerobic || false,
        fermentation_hours: inputs.fermentation_hours || 24,
        drying_method: inputs.drying_method || 'raised_bed',
        drying_days: inputs.drying_days || 14,
      },
      R: {
        roast_level: inputs.roast_level || 'medium_light',
        first_crack_temp_c: inputs.first_crack_temp_c || 198,
        drop_temp_c: inputs.drop_temp_c || 205,
        dtr_pct: inputs.dtr_pct || 22,
        total_time_s: inputs.total_time_s || 600,
      },
      B: {
        method: inputs.method_b || 'v60',
        grind_microns: inputs.grind_microns || 600,
        water_temp_c: inputs.water_temp_c || 93,
        ratio: inputs.ratio || 15,
        brew_time_s: inputs.brew_time_s || 240,
        water_tds_ppm: inputs.water_tds_ppm || 120,
      },
    };
  },

  getGrade(score) {
    if (score >= 90) return 'Outstanding';
    if (score >= 85) return 'Excellent';
    if (score >= 80) return 'Very Good';
    if (score >= 75) return 'Good';
    if (score >= 70) return 'Fair';
    return 'Below Specialty';
  },
};
