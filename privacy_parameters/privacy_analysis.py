"""
Privacy Analysis para MECAL_module
Compara métricas de privacidade entre texto original e categorias MECAL.

Experimentos:
1. Entropia - redução de informação
2. Valores únicos - redução de identificadores
3. Mutual Information - vazamento de informação
4. Equivalence Classes - tamanho dos grupos de anonimização
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt


def calculate_entropy(series: pd.Series) -> float:
    """Calcula entropia de Shannon de uma coluna categórica."""
    value_counts = series.value_counts(normalize=True)
    return entropy(value_counts, base=2)


def calculate_unique_ratio(series: pd.Series) -> dict:
    """Calcula métricas de unicidade para uma coluna."""
    total = len(series)
    unique = series.nunique()
    return {
        "total_records": total,
        "unique_values": unique,
        "uniqueness_ratio": unique / total if total > 0 else 0
    }


def calculate_equivalence_class_size(original: pd.Series, categorized: pd.Series) -> dict:
    """
    Calcula tamanho médio das classes de equivalência.
    Para cada categoria, conta quantos valores originais distintos mapeiam para ela.
    """
    df = pd.DataFrame({"original": original, "categorized": categorized})
    
    # Para cada categoria, contar valores originais distintos
    class_sizes = df.groupby("categorized")["original"].nunique()
    
    # Contar número de amostras por categoria
    sample_counts = df.groupby("categorized").size()
    
    # Encontrar menor cluster (menos amostras)
    smallest_cluster_name = sample_counts.idxmin() if len(sample_counts) > 0 else ""
    smallest_cluster_samples = int(sample_counts.min()) if len(sample_counts) > 0 else 0
    smallest_cluster_unique = int(class_sizes[smallest_cluster_name]) if smallest_cluster_name in class_sizes else 0
    
    # Encontrar cluster com menos valores únicos originais
    min_unique_cluster_name = class_sizes.idxmin() if len(class_sizes) > 0 else ""
    min_unique_cluster_size = int(class_sizes.min()) if len(class_sizes) > 0 else 0
    min_unique_cluster_samples = int(sample_counts[min_unique_cluster_name]) if min_unique_cluster_name in sample_counts else 0
    
    return {
        "min_class_size": int(class_sizes.min()) if len(class_sizes) > 0 else 0,
        "max_class_size": int(class_sizes.max()) if len(class_sizes) > 0 else 0,
        "mean_class_size": float(class_sizes.mean()) if len(class_sizes) > 0 else 0,
        "median_class_size": float(class_sizes.median()) if len(class_sizes) > 0 else 0,
        "num_categories": len(class_sizes),
        "smallest_cluster": {
            "category": str(smallest_cluster_name),
            "sample_count": smallest_cluster_samples,
            "unique_originals": smallest_cluster_unique
        },
        "min_unique_cluster": {
            "category": str(min_unique_cluster_name),
            "unique_originals": min_unique_cluster_size,
            "sample_count": min_unique_cluster_samples
        }
    }


def calculate_mutual_information(original: pd.Series, categorized: pd.Series) -> float:
    """
    Calcula informação mútua entre texto original e versão categorizada.
    Usa label encoding para ambas as séries.
    """
    # Converter para string e preencher NaN
    orig_labels = original.fillna("__NULL__").astype(str)
    cat_labels = categorized.fillna("__NULL__").astype(str)
    
    # Label encode
    orig_encoded = pd.factorize(orig_labels)[0]
    cat_encoded = pd.factorize(cat_labels)[0]
    
    return mutual_info_score(orig_encoded, cat_encoded)


def analyze_privacy(preprocessed_path: str, mecal_path: str, output_dir: str) -> dict:
    """
    Função principal de análise de privacidade.
    
    Args:
        preprocessed_path: Caminho do dataset com texto original (preprocessed.csv)
        mecal_path: Caminho do dataset com MECAL aplicado (after_MECAL.csv)
        output_dir: Diretório para salvar resultados
    
    Returns:
        Dicionário com métricas calculadas
    """
    print("=" * 60)
    print("Análise de Privacidade - MECAL")
    print("=" * 60)
    
    # Carregar datasets
    print("\nCarregando datasets...")
    preprocessed_df = pd.read_csv(preprocessed_path)
    mecal_df = pd.read_csv(mecal_path)
    
    print(f"Dataset preprocessado: {preprocessed_df.shape}")
    print(f"Dataset MECAL: {mecal_df.shape}")
    
    # Garantir mesmo número de linhas
    min_rows = min(len(preprocessed_df), len(mecal_df))
    preprocessed_df = preprocessed_df.iloc[:min_rows]
    mecal_df = mecal_df.iloc[:min_rows]
    
    # Preparar colunas para comparação
    desc_original = preprocessed_df['descrição'].fillna("")
    sol_original = preprocessed_df['solução'].fillna("")
    
    classe = mecal_df['classe'].fillna("")
    tipo_solucao = mecal_df['tipo_solução'].fillna("")
    
    # Calcular métricas
    results = {
        "descrição_vs_classe": {},
        "solução_vs_tipo_solução": {}
    }
    
    # ==========================================
    # Análise: descrição -> classe
    # ==========================================
    print("\n" + "=" * 50)
    print("Analisando: descrição -> classe")
    print("=" * 50)
    
    # 1. Entropia
    entropy_desc = calculate_entropy(desc_original)
    entropy_classe = calculate_entropy(classe)
    results["descrição_vs_classe"]["entropy"] = {
        "original_entropy": entropy_desc,
        "mecal_entropy": entropy_classe,
        "entropy_reduction": entropy_desc - entropy_classe,
        "entropy_reduction_pct": ((entropy_desc - entropy_classe) / entropy_desc * 100) if entropy_desc > 0 else 0
    }
    print(f"  Entropia: {entropy_desc:.4f} -> {entropy_classe:.4f} (redução: {entropy_desc - entropy_classe:.4f})")
    
    # 2. Unicidade
    unique_desc = calculate_unique_ratio(desc_original)
    unique_classe = calculate_unique_ratio(classe)
    results["descrição_vs_classe"]["uniqueness"] = {
        "original": unique_desc,
        "mecal": unique_classe,
        "uniqueness_reduction": unique_desc["unique_values"] - unique_classe["unique_values"]
    }
    print(f"  Valores únicos: {unique_desc['unique_values']} -> {unique_classe['unique_values']}")
    
    # 3. Mutual Information
    mi_desc = calculate_mutual_information(desc_original, classe)
    mi_desc_relative = (mi_desc / entropy_desc * 100) if entropy_desc > 0 else 0
    results["descrição_vs_classe"]["mutual_information"] = {
        "absolute": mi_desc,
        "relative_pct": mi_desc_relative,
        "privacy_preserved_pct": 100 - mi_desc_relative
    }
    print(f"  MI: {mi_desc:.4f} ({mi_desc_relative:.1f}% vazada, {100 - mi_desc_relative:.1f}% anonimizado)")
    
    # 4. Equivalence Classes
    eq_class_desc = calculate_equivalence_class_size(desc_original, classe)
    results["descrição_vs_classe"]["equivalence_classes"] = eq_class_desc
    print(f"  Classes de equiv.: min={eq_class_desc['min_class_size']}, max={eq_class_desc['max_class_size']}, média={eq_class_desc['mean_class_size']:.2f}")
    
    # ==========================================
    # Análise: solução -> tipo_solução
    # ==========================================
    print("\n" + "=" * 50)
    print("Analisando: solução -> tipo_solução")
    print("=" * 50)
    
    # 1. Entropia
    entropy_sol = calculate_entropy(sol_original)
    entropy_tipo = calculate_entropy(tipo_solucao)
    results["solução_vs_tipo_solução"]["entropy"] = {
        "original_entropy": entropy_sol,
        "mecal_entropy": entropy_tipo,
        "entropy_reduction": entropy_sol - entropy_tipo,
        "entropy_reduction_pct": ((entropy_sol - entropy_tipo) / entropy_sol * 100) if entropy_sol > 0 else 0
    }
    print(f"  Entropia: {entropy_sol:.4f} -> {entropy_tipo:.4f} (redução: {entropy_sol - entropy_tipo:.4f})")
    
    # 2. Unicidade
    unique_sol = calculate_unique_ratio(sol_original)
    unique_tipo = calculate_unique_ratio(tipo_solucao)
    results["solução_vs_tipo_solução"]["uniqueness"] = {
        "original": unique_sol,
        "mecal": unique_tipo,
        "uniqueness_reduction": unique_sol["unique_values"] - unique_tipo["unique_values"]
    }
    print(f"  Valores únicos: {unique_sol['unique_values']} -> {unique_tipo['unique_values']}")
    
    # 3. Mutual Information
    mi_sol = calculate_mutual_information(sol_original, tipo_solucao)
    mi_sol_relative = (mi_sol / entropy_sol * 100) if entropy_sol > 0 else 0
    results["solução_vs_tipo_solução"]["mutual_information"] = {
        "absolute": mi_sol,
        "relative_pct": mi_sol_relative,
        "privacy_preserved_pct": 100 - mi_sol_relative
    }
    print(f"  MI: {mi_sol:.4f} ({mi_sol_relative:.1f}% vazada, {100 - mi_sol_relative:.1f}% anonimizado)")
    
    # 4. Equivalence Classes
    eq_class_sol = calculate_equivalence_class_size(sol_original, tipo_solucao)
    results["solução_vs_tipo_solução"]["equivalence_classes"] = eq_class_sol
    print(f"  Classes de equiv.: min={eq_class_sol['min_class_size']}, max={eq_class_sol['max_class_size']}, média={eq_class_sol['mean_class_size']:.2f}")
    
    # ==========================================
    # Salvar resultados
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "privacy_metrics.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nResultados salvos em: {results_path}")
    
    # ==========================================
    # Gerar visualizações
    # ==========================================
    print("\nGerando visualizações...")
    
    # 1. Comparação de Entropia
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    categories = ['descrição', 'solução']
    original_entropy = [entropy_desc, entropy_sol]
    mecal_entropy = [entropy_classe, entropy_tipo]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, original_entropy, width, label='Original (Texto)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, mecal_entropy, width, label='MECAL (Categoria)', color='#27ae60')
    
    ax.set_ylabel('Entropia (bits)')
    ax.set_title('Redução de Entropia: MECAL vs Texto Original')
    ax.set_xticks(x)
    ax.set_xticklabels(['descrição -> classe', 'solução -> tipo_solução'])
    ax.legend()
    ax.bar_label(bars1, fmt='%.2f')
    ax.bar_label(bars2, fmt='%.2f')
    
    plt.tight_layout()
    entropy_path = os.path.join(output_dir, "entropy_comparison.png")
    fig.savefig(entropy_path)
    plt.close()
    print(f"  Salvo: {entropy_path}")
    
    # 2. Comparação de Valores Únicos
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    original_unique = [unique_desc['unique_values'], unique_sol['unique_values']]
    mecal_unique = [unique_classe['unique_values'], unique_tipo['unique_values']]
    
    bars1 = ax.bar(x - width/2, original_unique, width, label='Original (Texto)', color='#e74c3c')
    bars2 = ax.bar(x + width/2, mecal_unique, width, label='MECAL (Categoria)', color='#27ae60')
    
    ax.set_ylabel('Valores Únicos')
    ax.set_title('Redução de Valores Únicos: MECAL vs Texto Original')
    ax.set_xticks(x)
    ax.set_xticklabels(['descrição -> classe', 'solução -> tipo_solução'])
    ax.legend()
    ax.bar_label(bars1, fmt='%d')
    ax.bar_label(bars2, fmt='%d')
    
    plt.tight_layout()
    unique_path = os.path.join(output_dir, "uniqueness_comparison.png")
    fig.savefig(unique_path)
    plt.close()
    print(f"  Salvo: {unique_path}")
    
    # 3. Tabela Resumo
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Métrica', 'descrição->classe', 'solução->tipo_solução'],
        ['Entropia Original (bits)', f'{entropy_desc:.2f}', f'{entropy_sol:.2f}'],
        ['Entropia MECAL (bits)', f'{entropy_classe:.2f}', f'{entropy_tipo:.2f}'],
        ['Redução de Entropia (%)', f'{results["descrição_vs_classe"]["entropy"]["entropy_reduction_pct"]:.1f}%', f'{results["solução_vs_tipo_solução"]["entropy"]["entropy_reduction_pct"]:.1f}%'],
        ['Valores Únicos Original', f'{unique_desc["unique_values"]}', f'{unique_sol["unique_values"]}'],
        ['Valores Únicos MECAL', f'{unique_classe["unique_values"]}', f'{unique_tipo["unique_values"]}'],
        ['MI (info vazada %)', f'{mi_desc_relative:.1f}%', f'{mi_sol_relative:.1f}%'],
        ['Privacidade Preservada (%)', f'{100 - mi_desc_relative:.1f}%', f'{100 - mi_sol_relative:.1f}%'],
        ['Média Eq. Classes', f'{eq_class_desc["mean_class_size"]:.1f}', f'{eq_class_sol["mean_class_size"]:.1f}'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Estilizar header
    for j in range(3):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Resumo das Métricas de Privacidade - MECAL', fontsize=14, fontweight='bold', pad=20)
    summary_path = os.path.join(output_dir, "privacy_summary.png")
    fig.savefig(summary_path, bbox_inches='tight')
    plt.close()
    print(f"  Salvo: {summary_path}")
    
    print("\n" + "=" * 60)
    print("Análise de Privacidade Concluída!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.dirname(this_dir)
    
    preprocessed_path = os.path.join(module_dir, "datasets", "preprocessed.csv")
    mecal_path = os.path.join(module_dir, "datasets", "after_MECAL.csv")
    output_dir = os.path.join(this_dir, "results")
    
    # Verificar se os arquivos existem
    if not os.path.exists(preprocessed_path):
        print(f"ERRO: Arquivo não encontrado: {preprocessed_path}")
        print("Execute primeiro o preprocessing.py")
        exit(1)
    
    if not os.path.exists(mecal_path):
        print(f"ERRO: Arquivo não encontrado: {mecal_path}")
        print("Execute primeiro o MECAL.py")
        exit(1)
    
    analyze_privacy(preprocessed_path, mecal_path, output_dir)
