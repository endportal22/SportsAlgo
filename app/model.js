/* The model's formula, run on the phone (same for every sport):
   margin = intercept + sum(coef * (x - mean) / scale)
   win%   = normal CDF(margin / sigma)        (link "normal", NFL)
          = 1 / (1 + exp(-(a + b * margin)))   (link "logistic", NHL, NBA)
   Used to recompute a game instantly when you switch a starter (QB, goalie) or mark NBA
   players in or out. */
(function (root) {
  function erf(x) { // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const ncdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

  // NBA: a team's lineup strength from who's playing. Each player's projected minutes are
  // scaled so the team adds up to 240 (shares add up to 5), times his ratings.
  function lineupValue(players, on) {
    const act = players.filter((p) => on(p));
    const tot = act.reduce((s, p) => s + p.min, 0) || 1;
    const v = { pm: 0, box: 0, rapm: 0 };
    for (const p of act) { const sh = 5 * p.min / tot; for (const k in v) v[k] += sh * (p[k] || 0); }
    return v;
  }
  const isOn = (side, ov) => (p) => { const o = ov && ov.lineup && ov.lineup[side]; return o && p.id in o ? o[p.id] : p.on; };

  // feature values for a game, with optional overrides:
  //   starters: {home: option, away: option}; NBA lineups: {lineup: {home: {pid: bool}, away: {...}}}
  function features(game, model, ov) {
    const x = {};
    for (const c of game.contributions) x[c.feature] = c.value == null ? 0 : c.value;
    const sw = model.switch || {};
    if (ov && (ov.home || ov.away) && sw.rating) {
      const pick = (side) => {
        const o = ov[side], base = game[side + "_starter"];
        const src = o || base;
        return { rating: src.rating == null ? 0 : src.rating, isNew: src.new ? 1 : 0 };
      };
      const h = pick("home"), a = pick("away");
      x[sw.rating] = h.rating - a.rating;   // ratings are already on the model's scale (points or goals per game)
      if (sw.new) x[sw.new] = h.isNew - a.isNew;
      if (sw.final_week) x[sw.final_week] = game.final_week ? x[sw.rating] : 0;
    }
    if (ov && ov.lineup && sw.lineup && game.home_lineup) {
      // each term: a lineup rating ({f, k}) or the change from the team's usual lineup ({f, delta: {k: weight}})
      const terms = sw.lineup.terms || [{ f: sw.lineup.pm, k: "pm" }, { f: sw.lineup.box, k: "box" },
        { f: sw.lineup.delta, delta: { pm: 1, box: sw.lineup.box_w } }];
      const val = (side) => {
        const lv = lineupValue(game[side + "_lineup"], isOn(side, ov)), t = game[side + "_typ"] || {};
        const out = {};
        for (const term of terms) {
          out[term.f] = term.k ? lv[term.k] : Object.entries(term.delta).reduce((s, [k, w]) => s + w * (lv[k] - (t[k] || 0)), 0);
        }
        return out;
      };
      const h = val("home"), a = val("away");
      for (const term of terms) x[term.f] = h[term.f] - a[term.f];
    }
    return x;
  }

  function prob(model, m) {
    return model.link === "logistic" ? 1 / (1 + Math.exp(-(model.a + model.b * m))) : ncdf(m / model.sigma);
  }

  function predict(game, model, ov) {
    const x = features(game, model, ov);
    let m = model.intercept;
    const parts = [];
    model.feats.forEach((f, i) => {
      const p = model.coef[i] * ((x[f] || 0) - model.mean[i]) / model.scale[i];
      m += p;
      parts.push({ feature: f, label: model.labels[i], value: x[f], points: p });
    });
    return { margin: m, prob: prob(model, m), parts, base: model.intercept };
  }

  const api = { predict, features, ncdf, prob, lineupValue };
  if (typeof module !== "undefined") module.exports = api; else root.NFLModel = api;
})(this);
