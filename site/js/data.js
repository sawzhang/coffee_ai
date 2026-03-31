/**
 * Data loading utilities for Coffee Attribution AutoResearch
 */

const DataLoader = {
  _cache: {},

  async load(file) {
    if (this._cache[file]) return this._cache[file];
    try {
      const resp = await fetch(`data/${file}`);
      if (!resp.ok) throw new Error(`Failed to load ${file}`);
      const data = await resp.json();
      this._cache[file] = data;
      return data;
    } catch (e) {
      console.warn(`Could not load ${file}:`, e.message);
      return null;
    }
  },

  async getModel() { return this.load('model.json'); },
  async getResults() { return this.load('results.json'); },
  async getBeansSummary() { return this.load('beans_summary.json'); },
};
