/* The model's formula, run on the phone. Mirrors MarginModel in code/model.py:
   margin = intercept + sum(coef * (x - mean) / scale);  win% = normal CDF(margin / sigma).
   Used to recompute a game instantly when you switch a starting QB. */
(function (root) {
  function erf(x) { // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const ncdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

  // feature values for a game, with optional QB overrides {home: option, away: option}
  function features(game, ov) {
    const x = {};
    for (const c of game.contributions) x[c.feature] = c.value == null ? 0 : c.value;
    const q = (side) => {
      const o = ov && ov[side];
      const base = game[side + "_qb"];
      return o ? { rating: o.rating == null ? -2.8 : o.rating, isNew: o.new ? 1 : 0 }
               : { rating: base.rating == null ? -2.8 : base.rating, isNew: base.new ? 1 : 0 };
    };
    if (ov && (ov.home || ov.away)) {
      const h = q("home"), a = q("away");
      x.d_qb_rating = h.rating - a.rating;      // ratings are already in points per game (x35 dropbacks)
      x.d_qb_new = h.isNew - a.isNew;
      if ("fw_x_qb_rating" in x) x.fw_x_qb_rating = game.final_week ? x.d_qb_rating : 0;
    }
    return x;
  }

  function predict(game, model, ov) {
    const x = features(game, ov);
    let m = model.intercept;
    const parts = [];
    model.feats.forEach((f, i) => {
      const p = model.coef[i] * ((x[f] || 0) - model.mean[i]) / model.scale[i];
      m += p;
      parts.push({ feature: f, label: model.labels[i], value: x[f], points: p });
    });
    const prob = ncdf(m / model.sigma);
    return { margin: m, prob, parts, base: model.intercept };
  }

  const api = { predict, features, ncdf };
  if (typeof module !== "undefined") module.exports = api; else root.NFLModel = api;
})(this);
