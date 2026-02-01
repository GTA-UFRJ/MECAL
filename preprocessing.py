"""
Preprocessing para MECAL_module
Renomeia colunas do dataset OpenVAS para o formato esperado pelo MECAL.
"""

import os
import pandas as pd

def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Lê o dataset original do OpenVAS e renomeia as colunas de texto.
    
    Renomeações:
    - definition.description -> descrição
    - definition.solution -> solução
    
    Args:
        input_path: Caminho do dataset original
        output_path: Caminho para salvar o dataset preprocessado
    
    Returns:
        DataFrame preprocessado
    """
    print(f"Lendo dataset de: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Shape original: {df.shape}")
    
    # Renomear colunas principais
    column_mapping = {
        'definition.description': 'descrição',
        'definition.solution': 'solução',
    }
    
    # Verificar quais colunas existem
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            print(f"  Renomeando: {old_col} -> {new_col}")
        else:
            print(f"  AVISO: Coluna '{old_col}' não encontrada!")
    
    df = df.rename(columns=column_mapping)
    
    # Manter apenas colunas relevantes para o MECAL
    # (descrição e solução são obrigatórias, outras são opcionais para contexto)
    essential_cols = ['descrição', 'solução']
    optional_cols = [
        'definition.name',
        'definition.cve', 
        'definition.cvss3.base_score',
        'definition.cvss2.base_score',
        'definition.severity',
        'definition.family',
    ]
    
    # Selecionar colunas que existem
    cols_to_keep = [c for c in essential_cols if c in df.columns]
    cols_to_keep += [c for c in optional_cols if c in df.columns]
    
    if len(cols_to_keep) > 0:
        df = df[cols_to_keep]
        print(f"Colunas mantidas: {cols_to_keep}")
    
    # Remover linhas com valores nulos nas colunas essenciais
    original_len = len(df)
    df = df.dropna(subset=[c for c in essential_cols if c in df.columns])
    print(f"Removidas {original_len - len(df)} linhas com valores nulos")
    
    print(f"Shape final: {df.shape}")
    
    # Salvar
    df.to_csv(output_path, index=False)
    print(f"Dataset salvo em: {output_path}")
    
    return df


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(this_dir, "datasets", "dataset_original.csv")
    output_path = os.path.join(this_dir, "datasets", "preprocessed.csv")
    
    preprocess_dataset(input_path, output_path)
