/**
 * Coffee Scorer — API-backed prediction engine with local fallback.
 * Calls /api/predict, /api/recommend, /api/explore on the backend.
 * Falls back to client-side linear approximation when API is unavailable.
 */

const CoffeeScorer = {
  // Auto-detect API: use local if available, otherwise try remote Render deployment
  API_BASE: '/api',
  _remoteBase: 'https://coffee-ai-api.onrender.com/api',
  _apiChecked: false,
  _useRemote: false,

  async _ensureApi() {
    if (this._apiChecked) return;
    this._apiChecked = true;
    try {
      const resp = await fetch('/api/health', { signal: AbortSignal.timeout(2000) });
      if (resp.ok) return; // local API works
    } catch (e) { /* local not available */ }
    // Try remote
    try {
      const resp = await fetch(this._remoteBase.replace('/api', '/api/health'), { signal: AbortSignal.timeout(5000) });
      if (resp.ok) {
        this._useRemote = true;
        this.API_BASE = this._remoteBase;
        console.log('Using remote API:', this._remoteBase);
      }
    } catch (e) {
      console.log('No API available, using local fallback');
    }
  },

  // V2 normalization ranges (from prepare_v2.py)
  _numRanges: {
    altitude_m: [800, 2400],
    shade_pct: [0, 80],
    latitude: [-25, 25],
    delta_t_c: [5, 20],
    fermentation_hours: [0, 200],
    drying_days: [1, 30],
  },

  // Categorical categories (from prepare_v2.py)
  _varieties: [
    "Gesha", "Bourbon", "Typica", "SL28", "SL34", "Caturra",
    "Catuai", "Pacamara", "Ethiopian Heirloom", "74158", "Castillo", "Catimor"
  ],
  _processMethods: ["washed", "natural", "honey_yellow", "honey_red", "honey_black", "wet_hulled"],
  _soilTypes: ["volcanic", "clay", "loam", "sandy", "laterite"],
  _dryingMethods: ["raised_bed", "patio", "mechanical"],

  async predict(inputs) {
    await this._ensureApi();
    const body = this._buildRequest(inputs);
    try {
      const resp = await fetch(`${this.API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error('API returned ' + resp.status);
      return await resp.json();
    } catch (e) {
      console.warn('Predict API error, using local fallback:', e.message);
      return this.predictLocal(inputs);
    }
  },

  /**
   * Client-side scoring fallback using model weights from model.json.
   * Implements: score = bias + sum(weight_i * feature_i)
   * where features are normalized the same way prepare_v2.py does.
   */
  predictLocal(inputs) {
    const model = DataLoader._cache['model.json'];
    if (!model || !model.weights || !model.feature_names || model.bias === undefined) {
      console.warn('Model data not loaded, cannot predict locally');
      return null;
    }

    const features = this._encodeFeatures(inputs);
    const weights = model.weights;
    const bias = model.bias;

    // Linear approximation: score = bias + sum(weight_i * scaled_feature_i)
    // Scale weights to produce reasonable score deltas (GBR importances are relative,
    // we scale them so the total range is approximately +/- 8 points around bias)
    const totalWeight = weights.reduce((s, w) => s + Math.abs(w), 0) || 1;
    const scaleFactor = 8.0 / totalWeight;

    let score = bias;
    for (let i = 0; i < features.length && i < weights.length; i++) {
      // For numerical features (first 6), feature is already 0-1 normalized,
      // center it around 0.5 so average inputs produce ~bias
      if (i < 6) {
        score += weights[i] * (features[i] - 0.5) * scaleFactor * 2;
      } else {
        // For categorical/boolean: weight contributes when feature is active
        score += weights[i] * features[i] * scaleFactor;
      }
    }

    // Clamp to reasonable range
    score = Math.max(60, Math.min(95, score));

    const grade = this.getGrade(score);

    // Compute approximate factor attribution from model factor_weights
    const attribution = model.factor_weights
      ? { ...model.factor_weights }
      : { G: 0.35, P: 0.25, R: 0.0, B: 0.0 };

    // Normalize attribution to sum to 1 for display
    const attrSum = Object.values(attribution).reduce((s, v) => s + v, 0) || 1;
    for (const k of Object.keys(attribution)) {
      attribution[k] = attribution[k] / attrSum;
    }

    return { score, grade, attribution, local: true };
  },

  /**
   * Encode inputs into feature vector matching prepare_v2.py order.
   */
  _encodeFeatures(inputs) {
    const features = [];

    // Numerical (min-max normalized)
    const numSources = {
      altitude_m: inputs.altitude_m || 1600,
      shade_pct: inputs.shade_pct || 30,
      latitude: Math.abs(inputs.latitude || 8),
      delta_t_c: inputs.delta_t_c || 12,
      fermentation_hours: inputs.fermentation_hours || 36,
      drying_days: inputs.drying_days || 14,
    };
    for (const key of Object.keys(this._numRanges)) {
      const [lo, hi] = this._numRanges[key];
      const val = numSources[key];
      features.push(hi > lo ? (val - lo) / (hi - lo) : 0);
    }

    // Categorical (one-hot)
    const catSources = {
      variety: inputs.variety || 'Bourbon',
      method_p: inputs.method_p || 'washed',
      soil_type: inputs.soil_type || 'volcanic',
      drying_method: inputs.drying_method || 'raised_bed',
    };
    const catFields = {
      variety: this._varieties,
      method_p: this._processMethods,
      soil_type: this._soilTypes,
      drying_method: this._dryingMethods,
    };
    for (const [field, categories] of Object.entries(catFields)) {
      const val = catSources[field];
      for (const cat of categories) {
        features.push(val === cat ? 1.0 : 0.0);
      }
    }

    // Boolean
    features.push(inputs.anaerobic ? 1.0 : 0.0);

    // Engineered features (from prepare_v2.py)
    const fermH = inputs.fermentation_hours || 36;
    const variety = inputs.variety || 'Bourbon';
    const isAnaerobic = inputs.anaerobic || false;
    const premiumVarieties = ['Gesha', '74158', 'Ethiopian Heirloom', 'SL28', 'SL34'];
    features.push(Math.min(fermH / 120.0, 1.0));  // ferm_capped
    features.push((fermH > 150 && !isAnaerobic) ? 1.0 : 0.0);  // ferm_risk
    features.push((premiumVarieties.includes(variety) && isAnaerobic) ? 1.0 : 0.0);  // premium_anaerobic

    // Interaction features (from prepare_v2.py extended)
    // Must match NUM_RANGES_V2 normalization exactly
    const altNorm = (numSources.altitude_m - 800) / (2400 - 800);
    const latNorm = (Math.abs(inputs.latitude || 8) - (-25)) / (25 - (-25));
    const dtNorm = ((inputs.delta_t_c || 12) - 5) / (20 - 5);
    const interactionVarieties = ['Typica', 'Bourbon', 'Caturra', 'Gesha', 'SL28'];
    for (const v of interactionVarieties) {
      features.push(altNorm * (variety === v ? 1.0 : 0.0));
    }
    features.push(altNorm * latNorm);    // altitude_x_latitude
    features.push(altNorm * dtNorm);     // altitude_x_delta_t
    features.push(latNorm * dtNorm);     // latitude_x_delta_t

    return features;
  },

  async recommend(prefs, topK = 10) {
    await this._ensureApi();
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
    await this._ensureApi();
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
