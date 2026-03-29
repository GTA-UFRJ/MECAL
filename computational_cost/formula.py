"""
Fórmula Analítica do Custo Computacional do MECAL
===================================================

Ajusta uma fórmula T_MECAL(n, k) que estima o tempo de execução
completo do algoritmo MECAL dado:
  n = número de registros no dataset
  k = número de clusters

Estrutura do modelo (derivada da complexidade teórica):

  T_MECAL(n, k) = T_embed(n) + T_kmeans(n, k) + T_llm(k)   [por coluna]

onde:
  T_embed(n)     = α_e * n + β_e
      Embedding é linear em n (encoding one sentence at a time in batches).

  T_kmeans(n, k) = α_km * n * k + γ_km
      KMeans tem complexidade O(n * k * d * i) com d (dim) e i (iters)
      fixos. Medido a k_ref=30; α_km generaliza para qualquer k via n*k.

  T_llm(k)       = μ_llm * k
      Uma chamada de LLM por cluster; tempo por chamada assumido constante.

Os coeficientes são estimados por OLS sobre os dados de cost_metrics.json.
"""

import os
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "cost_metrics.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ols(x: np.ndarray, y: np.ndarray):
    """OLS: y = slope*x + intercept.
    Returns (slope, intercept, R², se_slope, se_intercept, t_crit_95)."""
    res = stats.linregress(x, y)
    n = len(x)
    t_crit = float(stats.t.ppf(0.975, df=n - 2))
    return (
        float(res.slope), float(res.intercept), float(res.rvalue ** 2),
        float(res.stderr), float(res.intercept_stderr), t_crit,
    )


def predict(n: float, k: float, coef: dict) -> float:
    """Total predicted T_MECAL in seconds given combined coefficients."""
    return (
        coef["alpha_embed"] * n + coef["beta_embed"]
        + coef["alpha_kmeans"] * n * k + coef["gamma_kmeans"]
        + coef["mu_llm"] * k
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("MECAL — Fórmula Analítica de Custo Computacional")
    print("=" * 65)

    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    k_ref = data["config"]["n_clusters"]   # 30
    columns = data["config"]["columns_tested"]   # ["descrição", "solução"]

    per_col = {}

    for col in columns:
        print(f"\n{'─'*55}")
        print(f"Coluna: '{col}'")
        print(f"{'─'*55}")

        # ---- Embedding: T = α_e * n + β_e --------------------------------
        emb_data = data["embedding_cost"][col]
        n_vals  = np.array([d["n_records"]             for d in emb_data])
        t_embed = np.array([d["statistics"]["mean_s"]  for d in emb_data])

        α_e, β_e, r2_e, se_α_e, se_β_e, t_crit_e = ols(n_vals, t_embed)
        print(f"\n  [Embedding]  T = {α_e:.6e} * n  +  {β_e:.6f}")
        print(f"               R² = {r2_e:.6f}")
        print(f"               α_embed  = {α_e:.6e} ± {t_crit_e*se_α_e:.6e}  (95%)")
        print(f"               β_embed  = {β_e:.6f} ± {t_crit_e*se_β_e:.6f}  (95%)")

        # ---- KMeans: T = α_km * (n*k) + γ_km ----------------------------
        # Regride sobre n*k_ref para que α_km seja em unidades s/(registro·cluster)
        km_data  = data["kmeans_cost"][col]
        t_kmeans = np.array([d["statistics"]["mean_s"] for d in km_data])
        nk_vals  = n_vals * k_ref   # feature = n * k_ref

        α_km, γ_km, r2_km, se_α_km, se_γ_km, t_crit_km = ols(nk_vals, t_kmeans)
        print(f"\n  [KMeans]     T = {α_km:.6e} * (n*k)  +  {γ_km:.6f}")
        print(f"               R² = {r2_km:.6f}")
        print(f"               (ajustado em k_ref={k_ref}; generaliza por n·k)")
        print(f"               α_kmeans = {α_km:.6e} ± {t_crit_km*se_α_km:.6e}  (95%)")
        print(f"               γ_kmeans = {γ_km:.6f} ± {t_crit_km*se_γ_km:.6f}  (95%)")

        # ---- LLM: T = μ * k  (μ = tempo médio por cluster) ---------------
        μ = data["llm_cost"][col]["statistics"]["mean_s"]
        ci_lo = data["llm_cost"][col]["statistics"]["ci_95_lower"]
        ci_hi = data["llm_cost"][col]["statistics"]["ci_95_upper"]
        print(f"\n  [LLM]        T = {μ:.4f} * k")
        print(f"               μ_llm    = {μ:.4f} ± {(ci_hi - ci_lo)/2:.4f}  (95%)")

        per_col[col] = dict(
            n_vals=n_vals, t_embed=t_embed, t_kmeans=t_kmeans,
            alpha_embed=α_e, beta_embed=β_e, r2_embed=r2_e,
            se_alpha_embed=se_α_e, se_beta_embed=se_β_e, t_crit_embed=t_crit_e,
            alpha_kmeans=α_km, gamma_kmeans=γ_km, r2_kmeans=r2_km,
            se_alpha_kmeans=se_α_km, se_gamma_kmeans=se_γ_km, t_crit_kmeans=t_crit_km,
            mu_llm=μ, ci_lo_llm=ci_lo, ci_hi_llm=ci_hi,
        )

    # -----------------------------------------------------------------------
    # Coeficientes combinados (soma das duas colunas)
    # -----------------------------------------------------------------------
    α_e_tot  = sum(per_col[c]["alpha_embed"]   for c in columns)
    β_e_tot  = sum(per_col[c]["beta_embed"]    for c in columns)
    α_km_tot = sum(per_col[c]["alpha_kmeans"]  for c in columns)
    γ_km_tot = sum(per_col[c]["gamma_kmeans"]  for c in columns)
    μ_tot    = sum(per_col[c]["mu_llm"]        for c in columns)
    C        = β_e_tot + γ_km_tot   # constante aditiva

    # Propagação de erros (colunas independentes → soma em quadratura)
    t_crit = per_col[columns[0]]["t_crit_embed"]   # mesmo n → mesmo t_crit
    se_α_e_tot  = np.sqrt(sum(per_col[c]["se_alpha_embed"]  ** 2 for c in columns))
    se_β_e_tot  = np.sqrt(sum(per_col[c]["se_beta_embed"]   ** 2 for c in columns))
    se_α_km_tot = np.sqrt(sum(per_col[c]["se_alpha_kmeans"] ** 2 for c in columns))
    se_γ_km_tot = np.sqrt(sum(per_col[c]["se_gamma_kmeans"] ** 2 for c in columns))
    ci_α_e  = (α_e_tot  - t_crit * se_α_e_tot,  α_e_tot  + t_crit * se_α_e_tot)
    ci_β_e  = (β_e_tot  - t_crit * se_β_e_tot,  β_e_tot  + t_crit * se_β_e_tot)
    ci_α_km = (α_km_tot - t_crit * se_α_km_tot, α_km_tot + t_crit * se_α_km_tot)
    ci_γ_km = (γ_km_tot - t_crit * se_γ_km_tot, γ_km_tot + t_crit * se_γ_km_tot)
    ci_μ_tot = (
        sum(per_col[c]["ci_lo_llm"] for c in columns),
        sum(per_col[c]["ci_hi_llm"] for c in columns),
    )
    ci_C = (ci_β_e[0] + ci_γ_km[0], ci_β_e[1] + ci_γ_km[1])

    combined = dict(
        alpha_embed=α_e_tot, beta_embed=β_e_tot,
        alpha_kmeans=α_km_tot, gamma_kmeans=γ_km_tot,
        mu_llm=μ_tot,
    )

    print("\n" + "=" * 65)
    print("FÓRMULA COMBINADA  T_MECAL(n, k)  [em segundos]")
    print("=" * 65)
    print(
        f"\n  T_MECAL(n, k) =\n"
        f"      {α_e_tot:.6e} * n                     (embeddings, ∝ n)\n"
        f"    + {α_km_tot:.6e} * n * k               (KMeans, ∝ n·k)\n"
        f"    + {μ_tot:.4f} * k                      (LLM, ∝ k)\n"
        f"    + {C:.4f}                              (constante)\n"
    )
    print(
        f"  Forma fatorada:\n"
        f"  T_MECAL(n, k) = "
        f"({α_e_tot:.6e} + {α_km_tot:.6e} * k) * n\n"
        f"                + {μ_tot:.4f} * k\n"
        f"                + {C:.4f}\n"
    )

    # -----------------------------------------------------------------------
    # Validação: T_real vs T_predito (embedding + kmeans) para cada n @ k=30
    # O componente LLM é fixo em k_ref (não varia com n nos experimentos)
    # -----------------------------------------------------------------------
    print("  Validação (k=30):")
    print(f"  {'n':>7}  {'T_real (s)':>11}  {'T_pred (s)':>11}  {'erro %':>8}")

    n_ref = per_col["descrição"]["n_vals"]
    validation = {}
    for i, n in enumerate(n_ref):
        # real
        t_rem = (per_col["descrição"]["t_embed"][i]
                 + per_col["solução"]["t_embed"][i]
                 + per_col["descrição"]["t_kmeans"][i]
                 + per_col["solução"]["t_kmeans"][i])
        t_llm_real = (per_col["descrição"]["mu_llm"]
                      + per_col["solução"]["mu_llm"]) * k_ref
        t_real = t_rem + t_llm_real

        t_pred = predict(n, k_ref, combined)
        err = (t_pred - t_real) / t_real * 100
        print(f"  {int(n):>7}  {t_real:>11.3f}  {t_pred:>11.3f}  {err:>+8.2f}%")
        validation[int(n)] = dict(t_real=round(t_real, 4),
                                  t_pred=round(t_pred, 4),
                                  error_pct=round(err, 3))

    # -----------------------------------------------------------------------
    # Previsões ilustrativas
    # -----------------------------------------------------------------------
    examples = {}
    print("\n  Previsões para cenários hipotéticos:")
    print(f"  {'n':>8}  {'k':>5}  {'T_MECAL':>12}  {'T_MECAL (min)':>14}")
    for n_ex in [5_000, 15_000, 30_000, 60_000, 100_000]:
        for k_ex in [10, 20, 30, 50, 100]:
            t = predict(n_ex, k_ex, combined)
            key = f"n={n_ex},k={k_ex}"
            examples[key] = round(t, 2)
            print(f"  {n_ex:>8}  {k_ex:>5}  {t:>11.1f}s  {t/60:>13.2f} min")

    # -----------------------------------------------------------------------
    # Salvar JSON
    # -----------------------------------------------------------------------
    output = {
        "formula": {
            "expression": "T_MECAL(n,k) = alpha_embed*n + beta_embed + alpha_kmeans*n*k + gamma_kmeans + mu_llm*k",
            "factored":   "T_MECAL(n,k) = (alpha_embed + alpha_kmeans*k)*n + mu_llm*k + (beta_embed+gamma_kmeans)",
            "units":      "seconds",
            "variables":  {"n": "number of records", "k": "number of clusters"},
            "complexity_note": "KMeans term O(n*k) reflects standard complexity O(n*k*d*i) with fixed embedding dimension d and iterations i",
            "k_ref_used": k_ref,
        },
        "coefficients": {
            "per_column": {
                col: {
                    "embedding":  {
                                   "alpha":      per_col[col]["alpha_embed"],
                                   "alpha_unc95": per_col[col]["t_crit_embed"] * per_col[col]["se_alpha_embed"],
                                   "beta":       per_col[col]["beta_embed"],
                                   "beta_unc95":  per_col[col]["t_crit_embed"] * per_col[col]["se_beta_embed"],
                                   "r2":         per_col[col]["r2_embed"],
                    },
                    "kmeans":     {
                                   "alpha":      per_col[col]["alpha_kmeans"],
                                   "alpha_unc95": per_col[col]["t_crit_kmeans"] * per_col[col]["se_alpha_kmeans"],
                                   "gamma":      per_col[col]["gamma_kmeans"],
                                   "gamma_unc95": per_col[col]["t_crit_kmeans"] * per_col[col]["se_gamma_kmeans"],
                                   "r2":         per_col[col]["r2_kmeans"],
                    },
                    "llm":        {"mu_per_cluster":  per_col[col]["mu_llm"],
                                   "mu_unc95":        (per_col[col]["ci_hi_llm"] - per_col[col]["ci_lo_llm"]) / 2},
                } for col in columns
            },
            "combined": {
                "alpha_embed_s_per_record":            α_e_tot,
                "alpha_embed_unc95":                   t_crit * se_α_e_tot,
                "beta_embed_s":                        β_e_tot,
                "beta_embed_unc95":                    t_crit * se_β_e_tot,
                "alpha_kmeans_s_per_record_cluster":   α_km_tot,
                "alpha_kmeans_unc95":                  t_crit * se_α_km_tot,
                "gamma_kmeans_s":                      γ_km_tot,
                "gamma_kmeans_unc95":                  t_crit * se_γ_km_tot,
                "mu_llm_s_per_cluster":                μ_tot,
                "mu_llm_unc95":                        (ci_μ_tot[1] - ci_μ_tot[0]) / 2,
                "constant_s":                          C,
                "constant_unc95":                      (ci_C[1] - ci_C[0]) / 2,
            },
        },
        "validation_k30": validation,
        "example_predictions_s": examples,
    }

    json_path = os.path.join(RESULTS_DIR, "formula_coefficients.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Coeficientes salvos: {json_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    COLORS = {"descrição": "#3b82f6", "solução": "#f59e0b"}

    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
    fig.suptitle(
        "Fórmula Analítica do Custo Computacional do MECAL\n"
        "Regressão Linear OLS  ·  T_MECAL(n, k) = α_e·n + α_km·n·k + μ_llm·k + C",
        fontsize=13, fontweight="bold",
    )

    # --- Subplots 0 & 1: Embedding fit por coluna --------------------------
    for idx, col in enumerate(columns):
        ax = fig.add_subplot(gs[0, idx])
        pc = per_col[col]
        n_v, t_v = pc["n_vals"], pc["t_embed"]
        n_line = np.linspace(n_v.min() * 0.85, n_v.max() * 1.08, 300)
        t_fit  = pc["alpha_embed"] * n_line + pc["beta_embed"]

        ax.scatter(n_v, t_v, color=COLORS[col], s=90, zorder=5,
                   label="Experimental (média 30 runs)")
        ax.plot(n_line, t_fit, color=COLORS[col], linestyle="--", lw=2,
                label=f"Ajuste OLS  R²={pc['r2_embed']:.5f}")
        ax.set_xlabel("Registros n", fontsize=10)
        ax.set_ylabel("Tempo médio (s)", fontsize=10)
        ax.set_title(f"Embedding — '{col}'", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Subplot 2: KMeans fit (ambas colunas somadas vs n·k) ---------------
    ax_km = fig.add_subplot(gs[0, 2])
    n_v   = per_col["descrição"]["n_vals"]
    t_km_total = (per_col["descrição"]["t_kmeans"]
                  + per_col["solução"]["t_kmeans"])
    nk_v   = n_v * k_ref
    nk_line = np.linspace(nk_v.min() * 0.85, nk_v.max() * 1.08, 300)
    t_km_fit = α_km_tot * nk_line + γ_km_tot

    ax_km.scatter(nk_v, t_km_total, color="#8b5cf6", s=90, zorder=5,
                  label="Experimental (soma colunas)")
    ax_km.plot(nk_line, t_km_fit, color="#8b5cf6", linestyle="--", lw=2,
               label="Ajuste OLS")
    ax_km.set_xlabel("n × k", fontsize=10)
    ax_km.set_ylabel("T_kmeans total (s)", fontsize=10)
    ax_km.set_title("KMeans — ambas as colunas (k=30)", fontsize=11)
    ax_km.legend(fontsize=8)
    ax_km.grid(True, alpha=0.3)

    # --- Subplot 3-4: T_MECAL(n, k) para vários k -------------------------
    ax_f = fig.add_subplot(gs[1, :2])
    n_range = np.linspace(500, 80_000, 600)
    k_list  = [5, 10, 20, 30, 50, 100]
    cmap    = plt.cm.plasma

    for i, k_val in enumerate(k_list):
        t_line = predict(n_range, k_val, combined)
        color  = cmap(i / (len(k_list) - 1))
        ax_f.plot(n_range / 1_000, t_line / 60, color=color, lw=2,
                  label=f"k = {k_val}")

    # Pontos medidos (k=30)
    for n_v_pt in n_ref:
        idx = list(n_ref).index(n_v_pt)
        t_r = (per_col["descrição"]["t_embed"][idx]
               + per_col["solução"]["t_embed"][idx]
               + per_col["descrição"]["t_kmeans"][idx]
               + per_col["solução"]["t_kmeans"][idx]
               + μ_tot * k_ref)
        ax_f.scatter([n_v_pt / 1_000], [t_r / 60], color="white",
                     edgecolors="red", s=80, zorder=10, linewidths=1.5)

    ax_f.scatter([], [], color="white", edgecolors="red", s=80,
                 linewidths=1.5, label="Medido (k=30)")
    ax_f.set_xlabel("Registros n (× 1 000)", fontsize=11)
    ax_f.set_ylabel("T_MECAL (minutos)", fontsize=11)
    ax_f.set_title("Projeção de T_MECAL(n, k) para diferentes k", fontsize=12)
    ax_f.legend(fontsize=9, ncol=2)
    ax_f.grid(True, alpha=0.3)

    # --- Subplot 5: tabela de coeficientes ----------------------------------
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.axis("off")
    rows = [
        ["Coeficiente",                  "descrição",
         "solução",                       "Combinado"],
        ["α_embed  (s/reg)",
         f"{per_col['descrição']['alpha_embed']:.4e}",
         f"{per_col['solução']['alpha_embed']:.4e}",
         f"{α_e_tot:.4e}"],
        ["β_embed  (s)",
         f"{per_col['descrição']['beta_embed']:.4f}",
         f"{per_col['solução']['beta_embed']:.4f}",
         f"{β_e_tot:.4f}"],
        ["α_kmeans (s/reg/cluster)",
         f"{per_col['descrição']['alpha_kmeans']:.4e}",
         f"{per_col['solução']['alpha_kmeans']:.4e}",
         f"{α_km_tot:.4e}"],
        ["γ_kmeans (s)",
         f"{per_col['descrição']['gamma_kmeans']:.4f}",
         f"{per_col['solução']['gamma_kmeans']:.4f}",
         f"{γ_km_tot:.4f}"],
        ["μ_llm  (s/cluster)",
         f"{per_col['descrição']['mu_llm']:.4f}",
         f"{per_col['solução']['mu_llm']:.4f}",
         f"{μ_tot:.4f}"],
        ["Constante C  (s)",    "—",    "—",   f"{C:.4f}"],
    ]

    tbl = ax_t.table(cellText=rows[1:], colLabels=rows[0],
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.65)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#3b82f6")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Highlight combined column
    for row_i in range(1, len(rows)):
        tbl[(row_i, 3)].set_facecolor("#dbeafe")
    ax_t.set_title("Coeficientes do Modelo", fontsize=11,
                   fontweight="bold", pad=10)

    plot_path = os.path.join(RESULTS_DIR, "formula_analysis.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico salvo: {plot_path}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FÓRMULA FINAL (em segundos):")
    print("=" * 65)
    print(f"""
  T_MECAL(n, k) =
      {α_e_tot:.6e} * n          ← embeddings  (∝ n)
    + {α_km_tot:.6e} * n * k    ← KMeans      (∝ n·k)
    + {μ_tot:.4f}         * k   ← LLM         (∝ k)
    + {C:.4f}                   ← constante

  Onde n = #registros, k = #clusters, T em segundos.

  Incertezas 95% dos coeficientes combinados:
    α_embed  = {α_e_tot:.6e} ± {t_crit*se_α_e_tot:.6e}  s/reg
    α_kmeans = {α_km_tot:.6e} ± {t_crit*se_α_km_tot:.6e}  s/(reg·cluster)
    μ_llm    = {μ_tot:.4f} ± {(ci_μ_tot[1]-ci_μ_tot[0])/2:.4f}  s/cluster
    Constante= {C:.4f} ± {(ci_C[1]-ci_C[0])/2:.4f}  s
""")
    print("=" * 65)
    print("Análise concluída!")
    print("=" * 65)


if __name__ == "__main__":
    main()
