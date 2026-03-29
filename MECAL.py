"""
MECAL - Módulo de Mascaramento por Clusterização de Embeddings e Avaliação por LLM

Aplica o algoritmo MECAL para anonimizar colunas de texto através de 
clusterização semântica usando sentence embeddings e rotulagem automática via LLM (Ollama).
"""

import os
import json
import random
import time
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import requests


# ==========================================
# Configuração do Ollama
# ==========================================
_OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{_OLLAMA_BASE}/api/generate"
OLLAMA_MODEL = "llama3.1"  # Modelo padrão, pode ser alterado
MAX_SAMPLES_PER_CLUSTER = 5  # Número de amostras a enviar para a LLM


def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Envia um prompt para o Ollama e retorna a resposta.
    
    Args:
        prompt: Texto do prompt
        model: Nome do modelo Ollama a usar
    
    Returns:
        Resposta da LLM como string
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Baixa temperatura para respostas consistentes
                    "num_predict": 150   # Limitar tamanho da resposta
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"  Erro ao consultar Ollama: {e}")
        return ""


def generate_cluster_label(
    cluster_id: int,
    samples: List[str],
    existing_labels: Dict[int, str],
    context: str,
    model: str = OLLAMA_MODEL
) -> Tuple[str, Optional[int]]:
    """
    Gera um rótulo para um cluster usando a LLM.
    
    Args:
        cluster_id: ID do cluster atual
        samples: Lista de amostras de texto do cluster
        existing_labels: Dicionário com rótulos já atribuídos {cluster_id: label}
        context: Contexto sobre o tipo de dados (ex: "vulnerabilidades de segurança")
        model: Modelo Ollama a usar
    
    Returns:
        Tuple (label, merge_with_cluster_id) onde:
        - label: Rótulo gerado para o cluster
        - merge_with_cluster_id: ID do cluster para mesclar, ou None se não mesclar
    """
    # Selecionar amostras representativas (até MAX_SAMPLES)
    if len(samples) > MAX_SAMPLES_PER_CLUSTER:
        selected_samples = random.sample(samples, MAX_SAMPLES_PER_CLUSTER)
    else:
        selected_samples = samples[:MAX_SAMPLES_PER_CLUSTER]
    
    # Formatar amostras para o prompt
    samples_text = "\n".join([f"  {i+1}. \"{s[:500]}...\"" if len(s) > 500 else f"  {i+1}. \"{s}\"" 
                              for i, s in enumerate(selected_samples)])
    
    # Formatar rótulos existentes
    if existing_labels:
        existing_labels_text = "\n".join([f"  - Cluster {cid}: \"{label}\"" 
                                          for cid, label in existing_labels.items()])
    else:
        existing_labels_text = "  (Nenhum rótulo atribuído ainda)"
    
    # Construir o prompt
    prompt = f"""You are a data labeling expert for cybersecurity datasets. Your task is to analyze text samples from a cluster and assign a SHORT, DESCRIPTIVE LABEL (2-5 words in English).

CONTEXT: These are {context} from an OpenVAS security scanner dataset. The goal is to create privacy-preserving categories that group similar items together.

CLUSTER {cluster_id} SAMPLES:
{samples_text}

EXISTING LABELS (already assigned to other clusters):
{existing_labels_text}

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

If merging with an existing cluster makes sense, use:
{{
  "label": "Existing Label",
  "merge_with_cluster": <cluster_id_number>,
  "reasoning": "Why these clusters should be merged"
}}

Respond with ONLY the JSON, no additional text:"""

    # Consultar a LLM
    response = query_ollama(prompt, model)
    
    if not response:
        # Fallback se a LLM não responder
        return f"Cluster_{cluster_id}", None
    
    # Tentar parsear a resposta JSON
    try:
        # Limpar resposta (remover possíveis marcadores de código)
        response_clean = response.strip()
        if response_clean.startswith("```"):
            response_clean = response_clean.split("```")[1]
            if response_clean.startswith("json"):
                response_clean = response_clean[4:]
        response_clean = response_clean.strip()
        
        result = json.loads(response_clean)
        label = result.get("label", f"Cluster_{cluster_id}")
        merge_with = result.get("merge_with_cluster")
        reasoning = result.get("reasoning", "")
        
        print(f"    Cluster {cluster_id}: '{label}'" + 
              (f" (mesclar com {merge_with})" if merge_with is not None else "") +
              (f" - {reasoning[:80]}..." if len(reasoning) > 80 else f" - {reasoning}"))
        
        return label, merge_with if isinstance(merge_with, int) else None
        
    except json.JSONDecodeError:
        # Se não conseguir parsear, extrair label manualmente
        print(f"    Cluster {cluster_id}: Falha ao parsear JSON, usando fallback")
        # Tentar extrair label da resposta
        if '"label"' in response:
            try:
                start = response.index('"label"') + 9
                end = response.index('"', start + 1)
                label = response[start:end]
                return label, None
            except (ValueError, IndexError):
                pass
        return f"Cluster_{cluster_id}", None


def label_clusters_with_llm(
    cluster_samples: Dict[int, List[str]],
    context: str,
    model: str = OLLAMA_MODEL
) -> Dict[int, str]:
    """
    Rotula todos os clusters usando a LLM.
    
    Args:
        cluster_samples: Dicionário {cluster_id: [lista de textos]}
        context: Descrição do contexto dos dados
        model: Modelo Ollama a usar
    
    Returns:
        Dicionário {cluster_id: label}
    """
    print(f"  Iniciando rotulagem automática com LLM ({model})...")
    
    labels = {}
    merge_mapping = {}  # cluster_id -> cluster_id_to_merge_with
    llm_times = []  # tempo de cada chamada LLM
    
    # Ordenar clusters por tamanho (maiores primeiro para ter labels mais representativos)
    sorted_clusters = sorted(cluster_samples.keys(), 
                            key=lambda k: len(cluster_samples[k]), 
                            reverse=True)
    
    t_llm_total_start = time.perf_counter()
    
    for i, cluster_id in enumerate(sorted_clusters):
        samples = cluster_samples[cluster_id]
        
        if len(samples) == 0:
            labels[cluster_id] = "Empty_Cluster"
            continue
        
        print(f"  [{i+1}/{len(sorted_clusters)}] Rotulando cluster {cluster_id} ({len(samples)} amostras)...")
        
        # Gerar label e medir tempo individual
        t_call_start = time.perf_counter()
        label, merge_with = generate_cluster_label(
            cluster_id=cluster_id,
            samples=samples,
            existing_labels=labels,
            context=context,
            model=model
        )
        t_call = time.perf_counter() - t_call_start
        llm_times.append(t_call)
        print(f"    Tempo da chamada LLM: {t_call:.2f}s")
        
        if merge_with is not None and merge_with in labels:
            # Registrar mesclagem
            merge_mapping[cluster_id] = merge_with
            labels[cluster_id] = labels[merge_with]
            print(f"    -> Mesclado com cluster {merge_with}: '{labels[merge_with]}'")
        else:
            labels[cluster_id] = label
    
    t_llm_total = time.perf_counter() - t_llm_total_start
    
    # Estatísticas
    unique_labels = set(labels.values())
    n_clusters = len(sorted_clusters)
    print(f"  Rotulagem concluída: {len(unique_labels)} rótulos únicos para {len(labels)} clusters")
    if merge_mapping:
        print(f"  Mesclagens realizadas: {len(merge_mapping)}")
    if llm_times:
        print(f"  Tempo LLM total: {t_llm_total:.2f}s | "
              f"media/cluster: {sum(llm_times)/len(llm_times):.2f}s | "
              f"max: {max(llm_times):.2f}s")
    else:
        print(f"  Tempo LLM total: {t_llm_total:.2f}s (sem chamadas bem-sucedidas)")
    
    return labels


def mecal_cluster(texts: list, n_clusters: int = 20, device: str = 'cuda:0') -> tuple:
    """
    Aplica clusterização semântica em uma lista de textos.
    
    Args:
        texts: Lista de textos para clusterizar
        n_clusters: Número de clusters (default: 20)
        device: Dispositivo para o modelo (cuda:0 ou cpu)
    
    Returns:
        Tuple (clusters, cluster_samples, timing) onde:
        - clusters: array com o cluster de cada texto
        - cluster_samples: dict mapeando cluster_id -> lista de textos
        - timing: dict com tempos de embedding e kmeans (em segundos)
    """
    print(f"  Gerando embeddings para {len(texts)} textos...")
    
    # Usar CPU se CUDA não estiver disponível
    t_embed_start = time.perf_counter()
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        embeddings = model.encode(texts, show_progress_bar=True, device=device)
    except Exception as e:
        print(f"  Aviso: Falha com {device}, usando CPU: {e}")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        embeddings = model.encode(texts, show_progress_bar=True, device='cpu')
    t_embed = time.perf_counter() - t_embed_start
    print(f"  Embeddings gerados em {t_embed:.2f}s ({len(texts)/t_embed:.0f} textos/s)")
    
    embeddings_array = np.array(embeddings)
    
    print(f"  Aplicando KMeans com {n_clusters} clusters...")
    t_kmeans_start = time.perf_counter()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    clusters = kmeans.fit_predict(embeddings_array)
    t_kmeans = time.perf_counter() - t_kmeans_start
    print(f"  KMeans concluído em {t_kmeans:.2f}s")
    
    # Agrupar textos por cluster para análise
    cluster_samples = {k: [] for k in range(n_clusters)}
    for i, text in enumerate(texts):
        cluster_samples[clusters[i].item()].append(text)
    
    timing = {"embedding": t_embed, "kmeans": t_kmeans}
    return clusters, cluster_samples, timing


def apply_mecal(
    df: pd.DataFrame, 
    device: str = 'cuda:0',
    ollama_model: str = OLLAMA_MODEL,
    n_clusters: int = 20
) -> pd.DataFrame:
    """
    Aplica o algoritmo MECAL no DataFrame com rotulagem automática via LLM.
    
    Processa as colunas:
    - descrição -> classe
    - solução -> tipo_solução
    
    Args:
        df: DataFrame com colunas 'descrição' e 'solução'
        device: Dispositivo para processamento de embeddings
        ollama_model: Modelo Ollama para rotulagem
        n_clusters: Número inicial de clusters
    
    Returns:
        DataFrame com colunas categorizadas e dict de tempos.
    """
    df = df.copy()
    timings = {}  # acumulador de tempos por etapa
    
    # ==========================================
    # MECAL para coluna 'descrição' -> 'classe'
    # ==========================================
    print("\n[1/2] Processando coluna 'descrição'...")
    
    desc_texts = df['descrição'].fillna("").tolist()
    t0 = time.perf_counter()
    clusters_desc, samples_desc, timing_desc = mecal_cluster(desc_texts, n_clusters=n_clusters, device=device)
    timings['desc_embedding'] = timing_desc['embedding']
    timings['desc_kmeans']    = timing_desc['kmeans']
    
    # Rotulagem automática com LLM
    print("\n  Rotulagem automática para 'descrição':")
    t_llm_desc = time.perf_counter()
    cluster_to_class_desc = label_clusters_with_llm(
        cluster_samples=samples_desc,
        context="vulnerability descriptions (describing security issues, CVEs, and software flaws)",
        model=ollama_model
    )
    timings['desc_llm'] = time.perf_counter() - t_llm_desc
    
    df['classe'] = [cluster_to_class_desc[clusters_desc[i].item()] 
                   for i in range(len(desc_texts))]
    
    print(f"  Classes geradas: {df['classe'].nunique()} categorias únicas")
    
    # ==========================================
    # MECAL para coluna 'solução' -> 'tipo_solução'
    # ==========================================
    print("\n[2/2] Processando coluna 'solução'...")
    
    sol_texts = df['solução'].fillna("").tolist()
    clusters_sol, samples_sol, timing_sol = mecal_cluster(sol_texts, n_clusters=n_clusters, device=device)
    timings['sol_embedding'] = timing_sol['embedding']
    timings['sol_kmeans']    = timing_sol['kmeans']
    
    # Rotulagem automática com LLM
    print("\n  Rotulagem automática para 'solução':")
    t_llm_sol = time.perf_counter()
    cluster_to_class_sol = label_clusters_with_llm(
        cluster_samples=samples_sol,
        context="solution recommendations (describing how to fix vulnerabilities, update packages, configure security)",
        model=ollama_model
    )
    timings['sol_llm'] = time.perf_counter() - t_llm_sol
    
    df['tipo_solução'] = [cluster_to_class_sol[clusters_sol[i].item()] 
                         for i in range(len(sol_texts))]
    
    print(f"  Tipos de solução gerados: {df['tipo_solução'].nunique()} categorias únicas")
    
    # Remover apenas as colunas de texto original que foram substituídas pelas classes
    df = df.drop(columns=['descrição', 'solução'], errors='ignore')
    
    return df, timings


def run_mecal(
    input_path: str, 
    output_path: str, 
    device: str = 'cuda:0',
    ollama_model: str = OLLAMA_MODEL,
    n_clusters: int = 20
) -> pd.DataFrame:
    """
    Executa o pipeline completo do MECAL.
    
    Args:
        input_path: Caminho do dataset preprocessado
        output_path: Caminho para salvar o resultado
        device: Dispositivo para processamento
        ollama_model: Modelo Ollama para rotulagem
        n_clusters: Número inicial de clusters
    
    Returns:
        DataFrame processado
    """
    print("=" * 60)
    print("MECAL - Mascaramento por Clusterização de Embeddings")
    print("       com Rotulagem Automática via LLM (Ollama)")
    print("=" * 60)
    
    # Verificar conexão com Ollama
    print(f"\nVerificando conexão com Ollama ({ollama_model})...")
    try:
        test_response = requests.get(f"{_OLLAMA_BASE}/api/tags", timeout=5)
        test_response.raise_for_status()
        available_models = [m["name"] for m in test_response.json().get("models", [])]
        print(f"  Ollama conectado! Modelos disponíveis: {available_models}")
        if ollama_model not in available_models and f"{ollama_model}:latest" not in available_models:
            print(f"  AVISO: Modelo '{ollama_model}' não encontrado. Verifique se está instalado.")
    except requests.exceptions.RequestException as e:
        print(f"  ERRO: Não foi possível conectar ao Ollama: {e}")
        print("  Certifique-se de que o Ollama está rodando (ollama serve)")
        exit(1)
    
    print(f"\nLendo dataset de: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Shape inicial: {df.shape}")
    
    # Aplicar MECAL
    t_total_start = time.perf_counter()
    df, timings = apply_mecal(df, device=device, ollama_model=ollama_model, n_clusters=n_clusters)
    t_total = time.perf_counter() - t_total_start
    
    print(f"\nShape final: {df.shape}")
    print(f"Colunas finais: {list(df.columns)}")
    
    # Salvar resultado
    df.to_csv(output_path, index=False)
    print(f"\nDataset salvo em: {output_path}")
    
    # Salvar mapeamento de labels para referência
    labels_path = output_path.replace(".csv", "_labels.json")
    label_summary = {
        "classe": df['classe'].value_counts().to_dict(),
        "tipo_solução": df['tipo_solução'].value_counts().to_dict()
    }
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(label_summary, f, indent=2, ensure_ascii=False)
    print(f"Resumo de labels salvo em: {labels_path}")
    
    # ==========================================
    # Sumário de tempos de execução
    # ==========================================
    t_embed_total = timings.get('desc_embedding', 0) + timings.get('sol_embedding', 0)
    t_kmeans_total = timings.get('desc_kmeans', 0) + timings.get('sol_kmeans', 0)
    t_llm_total = timings.get('desc_llm', 0) + timings.get('sol_llm', 0)
    t_transformers_total = t_embed_total + t_kmeans_total
    
    print("\n" + "=" * 60)
    print("  SUMARIO DE TEMPOS DE EXECUCAO")
    print("=" * 60)
    print(f"  Embeddings (sentence-transformers):")
    print(f"    descrição:  {timings.get('desc_embedding', 0):7.2f}s")
    print(f"    solução:    {timings.get('sol_embedding', 0):7.2f}s")
    print(f"    subtotal:   {t_embed_total:7.2f}s")
    print(f"  KMeans:")
    print(f"    descrição:  {timings.get('desc_kmeans', 0):7.2f}s")
    print(f"    solução:    {timings.get('sol_kmeans', 0):7.2f}s")
    print(f"    subtotal:   {t_kmeans_total:7.2f}s")
    print(f"  Transformers (embed + kmeans): {t_transformers_total:7.2f}s")
    print(f"  LLM (rotulagem Ollama):")
    print(f"    descrição:  {timings.get('desc_llm', 0):7.2f}s")
    print(f"    solução:    {timings.get('sol_llm', 0):7.2f}s")
    print(f"    subtotal:   {t_llm_total:7.2f}s")
    print(f"  TOTAL pipeline:               {t_total:7.2f}s")
    print("=" * 60)
    print("MECAL concluido com sucesso!")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MECAL - Mascaramento por Clusterização de Embeddings")
    parser.add_argument("--model", "-m", default=OLLAMA_MODEL, 
                       help=f"Modelo Ollama a usar (default: {OLLAMA_MODEL})")
    parser.add_argument("--clusters", "-c", type=int, default=20,
                       help="Número de clusters (default: 20)")
    parser.add_argument("--device", "-d", default="cuda:0",
                       help="Dispositivo para embeddings (default: cuda:0)")
    args = parser.parse_args()
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(this_dir, "dataset", "preprocessed.csv")
    output_path = os.path.join(this_dir, "dataset", "after_MECAL.csv")
    
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_path):
        print(f"ERRO: Arquivo não encontrado: {input_path}")
        print("Execute primeiro o preprocessing.py para gerar o arquivo preprocessed.csv")
        exit(1)
    
    run_mecal(
        input_path, 
        output_path, 
        device=args.device,
        ollama_model=args.model,
        n_clusters=args.clusters
    )
