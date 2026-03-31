/**
 * Bilingual support (Chinese / English) for Coffee Attribution
 */

const I18n = {
  _lang: 'en',

  translations: {
    // Header
    'site-title': { en: 'Coffee Attribution AutoResearch', zh: '咖啡归因自动研究' },
    'site-subtitle': { en: 'Specialty Coffee Multi-Factor Attribution System', zh: '精品咖啡多因子归因系统' },

    // Section headers
    'section-radar-title': { en: 'Factor Weight Radar', zh: '因子权重雷达图' },
    'section-radar-desc': { en: 'Model-learned importance of each factor dimension (G/P/R/B)', zh: '模型学习到的各因子维度重要性 (G/P/R/B)' },
    'section-features-title': { en: 'Top Feature Importance', zh: '重要特征排名' },
    'section-features-desc': { en: 'Most influential features in predicting coffee quality', zh: '预测咖啡品质最具影响力的特征' },
    'section-progress-title': { en: 'Experiment Progress', zh: '实验进展' },
    'section-progress-desc': { en: 'AutoResearch experiment history \u2014 val_mae over iterations', zh: 'AutoResearch 实验历史 \u2014 验证 MAE 随迭代变化' },
    'section-beans-title': { en: 'Top Scoring Beans', zh: '最高评分豆子' },
    'section-beans-desc': { en: 'Highest-rated beans in the dataset', zh: '数据集中评分最高的豆子' },
    'section-distribution-title': { en: 'Score Distribution', zh: '评分分布' },
    'section-distribution-desc': { en: 'Distribution of quality scores across the dataset', zh: '数据集中品质评分的分布情况' },
    'section-scorer-title': { en: 'Interactive Coffee Scorer', zh: '交互式咖啡评分器' },
    'section-scorer-desc': { en: 'Adjust factors to predict coffee quality score', zh: '调整因子预测咖啡品质评分' },
    'section-explore-title': { en: 'Factor Recombination Engine', zh: '因子重组引擎' },
    'section-explore-desc': { en: 'Fix the G factor above, explore all P combinations to find the best match', zh: '固定上方 G 因子，探索所有 P 组合以找到最佳匹配' },
    'section-user-title': { en: 'U: Your Taste Profile', zh: 'U: 你的口味档案' },
    'section-user-desc': { en: 'Set your preferences \u2014 save to get personalized bean recommendations', zh: '设置你的偏好 \u2014 保存后获取个性化豆子推荐' },

    // Scorer factor groups
    'factor-g': { en: 'G: Genetics / Geography', zh: 'G: 基因 / 地理' },
    'factor-p': { en: 'P: Processing', zh: 'P: 处理方式' },
    'factor-r': { en: 'R: Roasting', zh: 'R: 烘焙' },
    'factor-b': { en: 'B: Brewing', zh: 'B: 冲煮' },

    // Input labels
    'label-variety': { en: 'Variety', zh: '品种' },
    'label-altitude': { en: 'Altitude (m)', zh: '海拔 (m)' },
    'label-soil': { en: 'Soil Type', zh: '土壤类型' },
    'label-shade': { en: 'Shade %', zh: '遮荫 %' },
    'label-delta-t': { en: 'Day-Night Temp Diff (C)', zh: '昼夜温差 (C)' },
    'label-process': { en: 'Method', zh: '处理法' },
    'label-anaerobic': { en: 'Anaerobic', zh: '厌氧处理' },
    'label-ferm': { en: 'Fermentation Hours', zh: '发酵时长' },
    'label-drying': { en: 'Drying Method', zh: '干燥方式' },
    'label-roast': { en: 'Roast Level', zh: '烘焙度' },
    'label-dtr': { en: 'DTR %', zh: 'DTR %' },
    'label-drop-temp': { en: 'Drop Temp (C)', zh: '下豆温度 (C)' },
    'label-brew-method': { en: 'Method', zh: '冲煮方式' },
    'label-water-temp': { en: 'Water Temp (C)', zh: '水温 (C)' },
    'label-ratio': { en: 'Ratio (1:X)', zh: '粉水比 (1:X)' },
    'label-grind': { en: 'Grind (microns)', zh: '研磨度 (微米)' },

    // User prefs
    'label-pref-acidity': { en: 'Acidity Preference', zh: '酸度偏好' },
    'label-pref-sweetness': { en: 'Sweetness Weight', zh: '甜度权重' },
    'label-pref-complexity': { en: 'Complexity Tolerance', zh: '复杂度接受度' },
    'label-pref-fermentation': { en: 'Fermentation Acceptance', zh: '发酵接受度' },
    'label-pref-body': { en: 'Body Preference', zh: '醇厚度偏好' },

    // Buttons
    'btn-explore': { en: 'Explore Combinations', zh: '探索组合' },
    'btn-explore-loading': { en: 'Exploring...', zh: '探索中...' },
    'btn-save-prefs': { en: 'Save & Get Recommendations', zh: '保存并获取推荐' },

    // Score display
    'predicted-score-label': { en: 'Predicted Score', zh: '预测评分' },
    'no-experiments': { en: 'No experiments yet. Run the research agent to start optimizing.', zh: '暂无实验。运行研究代理开始优化。' },

    // Grades
    'grade-outstanding': { en: 'Outstanding', zh: '卓越' },
    'grade-excellent': { en: 'Excellent', zh: '优秀' },
    'grade-very-good': { en: 'Very Good', zh: '非常好' },
    'grade-good': { en: 'Good', zh: '良好' },
    'grade-fair': { en: 'Fair', zh: '一般' },
    'grade-below': { en: 'Below Specialty', zh: '低于精品级' },

    // Footer
    'footer-text': { en: 'Coffee Attribution AutoResearch \u2014 Powered by', zh: '咖啡归因自动研究 \u2014 基于' },
  },

  init() {
    const saved = localStorage.getItem('lang');
    this._lang = (saved === 'zh' || saved === 'en') ? saved : 'en';
    this.apply();
    this._updateToggle();
  },

  toggle() {
    this._lang = this._lang === 'en' ? 'zh' : 'en';
    localStorage.setItem('lang', this._lang);
    this.apply();
    this._updateToggle();
  },

  get lang() {
    return this._lang;
  },

  t(key) {
    const entry = this.translations[key];
    if (!entry) return key;
    return entry[this._lang] || entry.en || key;
  },

  apply() {
    const elements = document.querySelectorAll('[data-i18n]');
    for (const el of elements) {
      const key = el.getAttribute('data-i18n');
      const text = this.t(key);
      if (text) el.textContent = text;
    }
  },

  _updateToggle() {
    const btn = document.getElementById('lang-toggle');
    if (btn) {
      btn.textContent = this._lang === 'en' ? '\u4e2d\u6587' : 'EN';
    }
  },

  /**
   * Get localized grade name
   */
  grade(score) {
    if (score >= 90) return this.t('grade-outstanding');
    if (score >= 85) return this.t('grade-excellent');
    if (score >= 80) return this.t('grade-very-good');
    if (score >= 75) return this.t('grade-good');
    if (score >= 70) return this.t('grade-fair');
    return this.t('grade-below');
  }
};
