"""
Experimento de Custo Computacional do MECAL
============================================

Mede o custo computacional das três fases do algoritmo MECAL:

  1. Geração de embeddings (sentence-transformers) vs. tamanho do dataset
     - Frações testadas: 20%, 40%, 60%, 80%, 100%
     - Colunas: 'descrição' e 'solução'

  2. KMeans (k=30) vs. tamanho do dataset
     - Mesmas frações e colunas

  3. Custo de chamada à LLM (Ollama) por cluster (k=30, 100% do dataset)
     - Para cada cluster, chama a LLM individualmente e mede o tempo
     - Calcula estatísticas com IC 95% (t de Student) para cada coluna

Uso:
    python cost_analysis.py [--device cuda:0] [--model llama3.1]

Este script é completamente isolado do fluxo principal de execução do MECAL.
Lê apenas 'dataset/preprocessed.csv' e salva resultados em
'computational_cost/results/'.

Estratégia de warmup para confiança estatística:
  - Embedding e KMeans: 31 execuções por fração; o primeiro resultado é descartado
    (eliminação de outlier de inicialização). As 30 restantes são usadas para calcular
    IC 95% via t de Student.
  - LLM (Ollama): uma chamada de warmup com o prompt do primeiro cluster é feita e
    descartada antes de iniciar as medições reais.
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(THIS_DIR)
RESULTS_DIR = os.path.join(THIS_DIR, "results")

# ---------------------------------------------------------------------------
# Configuração do experimento
# ---------------------------------------------------------------------------
FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]
N_CLUSTERS = 30
N_RUNS = 31          # Total de execuções por fração; a primeira é descartada (warmup)
N_VALID_RUNS = N_RUNS - 1  # = 30 execuções usadas para estatística
RANDOM_SEED = 42
MAX_SAMPLES_PER_CLUSTER = 5
CONFIDENCE = 0.95

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_DEVICE = "cuda:0"

# Colunas que serão testadas com seus respectivos contextos LLM
COLUMNS = {
    "descrição": (
        "vulnerability descriptions (describing security issues, CVEs, and software flaws)"
    ),
    "solução": (
        "solution recommendations (describing how to fix vulnerabilities, "
        "update packages, configure security)"
    ),
}

# Paleta de cores por coluna
COLORS = {"descrição": "#3b82f6", "solução": "#f59e0b"}


# ===========================================================================
# Helpers: LLM
# ===========================================================================

def build_prompt(cluster_id: int, samples: list, existing_labels: dict, context: str) -> str:
    """Constrói o prompt para rotulagem de cluster (mesmo formato do MECAL.py)."""
    samples_text = "\n".join(
        f'  {i+1}. "{s[:500]}..."' if len(s) > 500 else f'  {i+1}. "{s}"'
        for i, s in enumerate(samples)
    )
    if existing_labels:
        existing_text = "\n".join(
            f'  - Cluster {cid}: "{lbl}"' for cid, lbl in existing_labels.items()
        )
    else:
        existing_text = "  (Nenhum rótulo atribuído ainda)"

    return f"""You are a data labeling expert for cybersecurity datasets. Your task is to analyze text samples from a cluster and assign a SHORT, DESCRIPTIVE LABEL (2-5 words in English).

CONTEXT: These are {context} from an OpenVAS security scanner dataset. The goal is to create privacy-preserving categories that group similar items together.

CLUSTER {cluster_id} SAMPLES:
{samples_text}

EXISTING LABELS (already assigned to other clusters):
{existing_text}

INSTRUCTIONS:
1. Analyze the samples and identify the common theme or pattern
2. Create a SHORT label (2-5 words) that describes this cluster
3. If the samples are semantically VERY similar to an existing cluster, you may suggest merging
4. The label should be generic enough to provide privacy but specific enough to be useful

RESPONSE FORMAT (JSON only, no other text):
{{
  "label": "Your Short Label Here",
  "merge_with_cluster": null,
  "reasoning": "Brief explanation of your choice"
}}

Respond with ONLY the JSON, no additional text:"""


def query_ollama_timed(prompt: str, model: str) -> tuple:
    """
    Envia um prompt ao Ollama e retorna (texto_resposta, tempo_em_segundos).
    O tempo inclui toda a latência de rede + geração de tokens.
    """
    t_start = time.perf_counter()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 150},
            },
            timeout=300,
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - t_start
        return response.json().get("response", "").strip(), elapsed
    except requests.exceptions.RequestException as e:
        elapsed = time.perf_counter() - t_start
        print(f"  [ERRO Ollama] {e}")
        return "", elapsed


def parse_llm_label(response_text: str, cluster_id: int) -> str:
    """Extrai o label da resposta JSON da LLM (fallback seguro)."""
    try:
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean.strip())
        return parsed.get("label", f"Cluster_{cluster_id}")
    except Exception:
        return f"Cluster_{cluster_id}"


# ===========================================================================
# Helpers: Estatística
# ===========================================================================

def compute_confidence_interval(times: list, confidence: float = 0.95) -> dict:
    """
    Calcula estatísticas descritivas + IC usando a distribuição t de Student.

    Com n amostras (uma por cluster), usa n-1 graus de liberdade.
    Adequado para amostras pequenas (< 30 observações).
    """
    n = len(times)
    arr = np.array(times, dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))

    if n > 1:
        lo, hi = stats.t.interval(
            confidence, df=n - 1, loc=mean, scale=stats.sem(arr)
        )
    else:
        lo, hi = mean, mean

    return {
        "n_observations": n,
        "mean_s": mean,
        "median_s": median,
        "std_s": std,
        "min_s": minimum,
        "max_s": maximum,
        f"ci_{int(confidence*100)}_lower": float(lo),
        f"ci_{int(confidence*100)}_upper": float(hi),
        "projected_total_s": mean * N_CLUSTERS,
    }


# ===========================================================================
# Geração de embeddings (com fallback para CPU)
# ===========================================================================

def encode_texts(texts: list, device: str) -> tuple:
    """
    Gera embeddings para `texts`. Retorna (embeddings_array, tempo_segundos).
    Faz fallback para CPU automaticamente se CUDA não estiver disponível.
    """
    try:
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        t0 = time.perf_counter()
        emb = st_model.encode(texts, show_progress_bar=False, device=device)
        t_embed = time.perf_counter() - t0
        return np.array(emb), t_embed
    except Exception as e:
        print(f"    [AVISO] Falha com {device}, usando CPU: {e}")
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        t0 = time.perf_counter()
        emb = st_model.encode(texts, show_progress_bar=False, device="cpu")
        t_embed = time.perf_counter() - t0
        return np.array(emb), t_embed


def encode_texts_with_progress(texts: list, device: str) -> tuple:
    """Como encode_texts, mas exibe barra de progresso (para 100% do dataset)."""
    try:
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        t0 = time.perf_counter()
        emb = st_model.encode(texts, show_progress_bar=True, device=device)
        t_embed = time.perf_counter() - t0
        return np.array(emb), t_embed
    except Exception as e:
        print(f"    [AVISO] Falha com {device}, usando CPU: {e}")
        st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        t0 = time.perf_counter()
        emb = st_model.encode(texts, show_progress_bar=True, device="cpu")
        t_embed = time.perf_counter() - t0
        return np.array(emb), t_embed


# ===========================================================================
# Plots
# ===========================================================================

def plot_embedding_cost(results: dict, output_dir: str):
    """Gráfico: tempo de embedding vs. fração do dataset com IC 95% (uma subplot por coluna)."""
    ci_key = f"ci_{int(CONFIDENCE*100)}"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle(
        f"Custo de Geração de Embeddings vs. Tamanho do Dataset"
        f" (média ± IC {int(CONFIDENCE*100)}%, {N_VALID_RUNS} runs)",
        fontsize=13, fontweight="bold",
    )

    for ax, col in zip(axes, COLUMNS.keys()):
        data = results["embedding_cost"][col]
        fracs_pct   = [int(d["fraction"] * 100) for d in data]
        means       = [d["statistics"]["mean_s"] for d in data]
        ci_lo       = [d["statistics"][f"{ci_key}_lower"] for d in data]
        ci_hi       = [d["statistics"][f"{ci_key}_upper"] for d in data]
        n_records   = [d["n_records"] for d in data]
        throughputs = [d["mean_throughput_texts_per_s"] for d in data]
        err_lo      = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi      = [hi - m for m, hi in zip(means, ci_hi)]

        ax.errorbar(
            fracs_pct, means,
            yerr=[err_lo, err_hi],
            fmt="-o", color=COLORS[col], linewidth=2.5, markersize=9,
            capsize=5, capthick=1.5, elinewidth=1.5,
            label=f"Média ± IC {int(CONFIDENCE*100)}%",
        )

        for x, y, n, tp in zip(fracs_pct, means, n_records, throughputs):
            ax.annotate(
                f"{y:.2f}s\n({tp:.0f} t/s)",
                (x, y),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                fontsize=8,
                color="#333333",
            )

        ax.set_xlabel("Fração do Dataset (%)", fontsize=11)
        ax.set_ylabel("Tempo médio (s)", fontsize=11)
        ax.set_title(f"Coluna: '{col}'", fontsize=12)
        ax.set_xticks(fracs_pct)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(10, 110)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "embedding_cost.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico salvo: {path}")


def plot_kmeans_cost(results: dict, output_dir: str):
    """Gráfico: tempo do KMeans vs. fração do dataset com IC 95% (uma subplot por coluna)."""
    ci_key = f"ci_{int(CONFIDENCE*100)}"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle(
        f"Custo do KMeans (k={N_CLUSTERS}) vs. Tamanho do Dataset"
        f" (média ± IC {int(CONFIDENCE*100)}%, {N_VALID_RUNS} runs)",
        fontsize=13, fontweight="bold",
    )

    for ax, col in zip(axes, COLUMNS.keys()):
        data = results["kmeans_cost"][col]
        fracs_pct = [int(d["fraction"] * 100) for d in data]
        means     = [d["statistics"]["mean_s"] for d in data]
        ci_lo     = [d["statistics"][f"{ci_key}_lower"] for d in data]
        ci_hi     = [d["statistics"][f"{ci_key}_upper"] for d in data]
        n_records = [d["n_records"] for d in data]
        err_lo    = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi    = [hi - m for m, hi in zip(means, ci_hi)]

        bars = ax.bar(fracs_pct, means, color=COLORS[col], alpha=0.82, width=12, zorder=3)
        ax.errorbar(
            fracs_pct, means,
            yerr=[err_lo, err_hi],
            fmt="none", color="black", capsize=5, capthick=1.5, elinewidth=1.5, zorder=4,
        )
        ax.grid(True, alpha=0.3, axis="y", zorder=0)

        y_offset = max(means) * 0.03 if max(means) > 0 else 0.001
        for x, y, n in zip(fracs_pct, means, n_records):
            ax.text(x, y + y_offset + max(means) * 0.06, f"{y:.4f}s\n(n={n})",
                    ha="center", fontsize=8)

        ax.set_xlabel("Fração do Dataset (%)", fontsize=11)
        ax.set_ylabel("Tempo médio (s)", fontsize=11)
        ax.set_title(f"Coluna: '{col}'", fontsize=12)
        ax.set_xticks(fracs_pct)

    plt.tight_layout()
    path = os.path.join(output_dir, "kmeans_cost.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico salvo: {path}")


def plot_llm_cost_ci(results: dict, output_dir: str):
    """
    Gráfico: tempo por requisição LLM para cada cluster (k=30),
    com linha da média e faixa de IC 95% — uma subplot por coluna.
    """
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), dpi=150)
    fig.suptitle(
        f"Tempo por Requisição LLM por Cluster (k={N_CLUSTERS}, IC {int(CONFIDENCE*100)}%)",
        fontsize=14, fontweight="bold",
    )

    for ax, col in zip(axes, COLUMNS.keys()):
        llm_data = results["llm_cost"][col]
        per_cluster_times = llm_data["per_cluster_times_s"]
        st = llm_data["statistics"]

        x = list(range(1, len(per_cluster_times) + 1))
        mean = st["mean_s"]
        ci_lo = st[f"ci_{int(CONFIDENCE*100)}_lower"]
        ci_hi = st[f"ci_{int(CONFIDENCE*100)}_upper"]
        projected = st["projected_total_s"]

        ax.bar(x, per_cluster_times, color=COLORS[col], alpha=0.65, label="Tempo real por cluster")
        ax.axhline(
            mean, color="black", linewidth=2, linestyle="--",
            label=f"Média: {mean:.2f}s",
        )
        ax.axhspan(
            ci_lo, ci_hi, alpha=0.18, color="gray",
            label=f"IC {int(CONFIDENCE*100)}%: [{ci_lo:.2f}, {ci_hi:.2f}]s",
        )

        ax.set_xlabel("Cluster (ordem de processamento)", fontsize=11)
        ax.set_ylabel("Tempo (s)", fontsize=11)
        ax.set_title(
            f"Coluna: '{col}'\n"
            f"σ={st['std_s']:.2f}s | min={st['min_s']:.2f}s | max={st['max_s']:.2f}s\n"
            f"Total projetado: {projected:.1f}s",
            fontsize=10,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x)

    plt.tight_layout()
    path = os.path.join(output_dir, "llm_cost_ci.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico salvo: {path}")


# ===========================================================================
# Experimento principal
# ===========================================================================

def run_cost_experiment(device: str, ollama_model: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "config": {
            "n_clusters": N_CLUSTERS,
            "fractions": FRACTIONS,
            "confidence": CONFIDENCE,
            "device": device,
            "ollama_model": ollama_model,
            "random_seed": RANDOM_SEED,
            "columns_tested": list(COLUMNS.keys()),
            "n_runs_per_fraction": N_RUNS,
            "n_valid_runs": N_VALID_RUNS,
            "warmup_discarded": True,
        },
        "embedding_cost": {col: [] for col in COLUMNS},
        "kmeans_cost": {col: [] for col in COLUMNS},
        "llm_cost": {},
    }

    # ------------------------------------------------------------------
    # Carregar dataset
    # ------------------------------------------------------------------
    preprocessed_path = os.path.join(MODULE_DIR, "dataset", "preprocessed.csv")
    if not os.path.exists(preprocessed_path):
        print(f"\nERRO: Arquivo não encontrado: {preprocessed_path}")
        print("Execute primeiro o preprocessing.py")
        sys.exit(1)

    print(f"\nCarregando dataset: {preprocessed_path}")
    df_full = pd.read_csv(preprocessed_path)
    df_full = df_full.dropna(subset=list(COLUMNS.keys()))
    df_full = df_full.reset_index(drop=True)
    n_total = len(df_full)
    print(f"Total de registros válidos: {n_total}")

    # ------------------------------------------------------------------
    # FASE 1 & 2: Embedding + KMeans por fração (31 runs, descarta 1ª)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 1 & 2: Embedding e KMeans (k={N_CLUSTERS}) por fração")
    print(f"  Rodadas: {N_RUNS} (1ª descartada como warmup, {N_VALID_RUNS} usadas)")
    print("=" * 60)

    for frac in FRACTIONS:
        n_sample = max(N_CLUSTERS + 1, int(n_total * frac))  # garante n >= k
        df_sample = df_full.sample(n=n_sample, random_state=RANDOM_SEED)
        pct = int(frac * 100)
        print(f"\n--- Fração {pct}% ({n_sample} registros) ---")

        for col in COLUMNS:
            texts = df_sample[col].fillna("").tolist()

            # ----------------------------------------------------------
            # Embedding: 31 runs, descarta a 1ª
            # ----------------------------------------------------------
            print(f"  [{col}] embeddings — {N_RUNS} runs (descarta 1ª)...", flush=True)
            embed_times = []
            last_emb_array = None
            for run in range(N_RUNS):
                emb_array, t_embed = encode_texts(texts, device)
                label = "[warmup, descartado]" if run == 0 else f"run {run}/{N_VALID_RUNS}"
                print(f"    {label}: {t_embed:.3f}s", flush=True)
                if run > 0:  # descarta o 1º resultado
                    embed_times.append(t_embed)
                last_emb_array = emb_array  # reutiliza o último embedding para KMeans

            emb_stats = compute_confidence_interval(embed_times, CONFIDENCE)
            mean_tp = len(texts) / emb_stats["mean_s"] if emb_stats["mean_s"] > 0 else 0.0
            print(
                f"  [{col}] embed — média={emb_stats['mean_s']:.3f}s "
                f"IC95%=[{emb_stats[f'ci_{int(CONFIDENCE*100)}_lower']:.3f}, "
                f"{emb_stats[f'ci_{int(CONFIDENCE*100)}_upper']:.3f}]s "
                f"({mean_tp:.0f} t/s)"
            )

            results["embedding_cost"][col].append({
                "fraction": frac,
                "n_records": n_sample,
                "run_times_s": [float(t) for t in embed_times],
                "statistics": emb_stats,
                "mean_throughput_texts_per_s": float(mean_tp),
            })

            # ----------------------------------------------------------
            # KMeans: 31 runs, descarta a 1ª (usa embeddings já gerados)
            # ----------------------------------------------------------
            print(f"  [{col}] KMeans k={N_CLUSTERS} — {N_RUNS} runs (descarta 1ª)...", flush=True)
            km_times = []
            for run in range(N_RUNS):
                km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init="auto")
                t0 = time.perf_counter()
                km.fit(last_emb_array)
                t_km = time.perf_counter() - t0
                label = "[warmup, descartado]" if run == 0 else f"run {run}/{N_VALID_RUNS}"
                print(f"    {label}: {t_km:.4f}s", flush=True)
                if run > 0:
                    km_times.append(t_km)

            km_stats = compute_confidence_interval(km_times, CONFIDENCE)
            print(
                f"  [{col}] kmeans — média={km_stats['mean_s']:.4f}s "
                f"IC95%=[{km_stats[f'ci_{int(CONFIDENCE*100)}_lower']:.4f}, "
                f"{km_stats[f'ci_{int(CONFIDENCE*100)}_upper']:.4f}]s"
            )

            results["kmeans_cost"][col].append({
                "fraction": frac,
                "n_records": n_sample,
                "run_times_s": [float(t) for t in km_times],
                "statistics": km_stats,
            })

    # Salvar resultado parcial
    _save_json(results)

    # ------------------------------------------------------------------
    # FASE 3: Custo LLM por cluster (100% dataset, k=30)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FASE 3: Custo LLM por cluster (k={N_CLUSTERS}, 100% dataset)")
    print("=" * 60)

    for col, context in COLUMNS.items():
        print(f"\n--- Coluna: '{col}' ---")
        texts_full = df_full[col].fillna("").tolist()

        # Embeddings com barra de progresso
        print(f"  Gerando embeddings ({len(texts_full)} textos)...")
        emb_full, t_emb_full = encode_texts_with_progress(texts_full, device)
        print(f"  Embedding concluído em {t_emb_full:.2f}s")

        # KMeans
        print(f"  Aplicando KMeans k={N_CLUSTERS}...")
        km_full = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init="auto")
        t0 = time.perf_counter()
        cluster_labels = km_full.fit_predict(emb_full)
        t_km_full = time.perf_counter() - t0
        print(f"  KMeans concluído em {t_km_full:.3f}s")

        # Agrupar textos por cluster
        cluster_samples_map = {k: [] for k in range(N_CLUSTERS)}
        for i, text in enumerate(texts_full):
            cluster_samples_map[cluster_labels[i]].append(text)

        # Ordenar por tamanho (maiores primeiro — igual ao MECAL.py)
        sorted_cluster_ids = sorted(
            cluster_samples_map.keys(),
            key=lambda k: len(cluster_samples_map[k]),
            reverse=True,
        )

        # Chamar LLM individualmente para cada cluster e medir tempo
        # Warmup: faz uma chamada com o prompt do 1º cluster e descarta o resultado
        first_cluster_id = sorted_cluster_ids[0]
        first_samples = cluster_samples_map[first_cluster_id]
        first_selected = random.sample(first_samples, min(MAX_SAMPLES_PER_CLUSTER, len(first_samples)))
        warmup_prompt = build_prompt(first_cluster_id, first_selected, {}, context)
        print(f"\n  [WARMUP] Chamada de aquecimento ao Ollama (resultado descartado)...", end=" ", flush=True)
        _, warmup_elapsed = query_ollama_timed(warmup_prompt, ollama_model)
        print(f"{warmup_elapsed:.2f}s (descartado)")

        print(f"\n  Chamando LLM para {N_CLUSTERS} clusters (uma chamada por cluster)...")
        per_cluster_times = []
        existing_labels = {}

        for i, cluster_id in enumerate(sorted_cluster_ids):
            samples = cluster_samples_map[cluster_id]
            if not samples:
                continue

            selected = random.sample(samples, min(MAX_SAMPLES_PER_CLUSTER, len(samples)))
            prompt = build_prompt(cluster_id, selected, existing_labels, context)

            print(
                f"  [{i+1:02d}/{N_CLUSTERS}] Cluster {cluster_id:02d} "
                f"({len(samples)} amostras)...",
                end=" ",
                flush=True,
            )
            response_text, elapsed = query_ollama_timed(prompt, ollama_model)
            per_cluster_times.append(elapsed)
            print(f"{elapsed:.2f}s")

            # Atualizar labels para próximas chamadas (simula uso real)
            existing_labels[cluster_id] = parse_llm_label(response_text, cluster_id)

        # Estatísticas com IC 95% (t de Student)
        ci_stats = compute_confidence_interval(per_cluster_times, CONFIDENCE)

        print(f"\n  Estatísticas ({col}):")
        print(f"    n = {ci_stats['n_observations']} chamadas")
        print(f"    Média   = {ci_stats['mean_s']:.2f}s")
        print(f"    Mediana = {ci_stats['median_s']:.2f}s")
        print(f"    Desvio  = {ci_stats['std_s']:.2f}s")
        print(f"    IC 95%  = [{ci_stats[f'ci_{int(CONFIDENCE*100)}_lower']:.2f}, "
              f"{ci_stats[f'ci_{int(CONFIDENCE*100)}_upper']:.2f}]s")
        print(f"    Total projetado = {ci_stats['projected_total_s']:.1f}s")

        results["llm_cost"][col] = {
            "n_clusters": N_CLUSTERS,
            "per_cluster_times_s": [float(t) for t in per_cluster_times],
            "statistics": ci_stats,
        }

        # Salvar após cada coluna (recuperação em caso de interrupção)
        _save_json(results)

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    print("\nGerando visualizações...")
    plot_embedding_cost(results, RESULTS_DIR)
    plot_kmeans_cost(results, RESULTS_DIR)
    plot_llm_cost_ci(results, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("Experimento de Custo Computacional Concluído!")
    print(f"Resultados em: {RESULTS_DIR}")
    print("=" * 60)


# ===========================================================================
# Utilitário
# ===========================================================================

def _save_json(results: dict):
    """Salva o JSON de resultados (chamado após cada fase para evitar perda de dados)."""
    path = os.path.join(RESULTS_DIR, "cost_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experimento de Custo Computacional do MECAL"
    )
    parser.add_argument(
        "--device", "-d", default=DEFAULT_DEVICE,
        help=f"Dispositivo para embeddings (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_OLLAMA_MODEL,
        help=f"Modelo Ollama (default: {DEFAULT_OLLAMA_MODEL})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MECAL — Experimento de Custo Computacional")
    print("=" * 60)

    # Verificar Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        print(f"\nOllama conectado. Modelos disponíveis: {available}")
        if args.model not in available and f"{args.model}:latest" not in available:
            print(f"AVISO: Modelo '{args.model}' não encontrado. Verifique se está instalado.")
    except Exception as e:
        print(f"\nERRO: Ollama não disponível: {e}")
        print("Certifique-se de que o Ollama está rodando (ollama serve)")
        sys.exit(1)

    run_cost_experiment(device=args.device, ollama_model=args.model)
