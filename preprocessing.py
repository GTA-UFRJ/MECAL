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
    - definition.solution    -> solução
    
    Todas as demais colunas são preservadas.
    
    Args:
        input_path: Caminho do dataset original
        output_path: Caminho para salvar o dataset preprocessado
    
    Returns:
        DataFrame preprocessado
    """
    print(f"Lendo dataset de: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Shape original: {df.shape}")
    print(f"Colunas originais: {list(df.columns)}")
    
    # Renomear colunas de texto livre
    column_mapping = {
        'definition.description': 'descrição',
        'definition.solution':    'solução',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            print(f"  Renomeando: {old_col} -> {new_col}")
        else:
            print(f"  AVISO: Coluna '{old_col}' não encontrada!")
    
    df = df.rename(columns=column_mapping)
    
    # Remover prefixo de colunas que ainda contêm "." (ex: asset.host_name -> host_name).
    # Aplica-se a qualquer coluna remanescente com ponto — generalizado para qualquer schema.
    dot_rename = {
        col: col.rsplit(".", 1)[-1]
        for col in df.columns
        if "." in col
    }
    if dot_rename:
        for old, new in dot_rename.items():
            print(f"  Simplificando: {old} -> {new}")
        df = df.rename(columns=dot_rename)
    
    # Remover linhas com valores nulos nas colunas essenciais
    essential_cols = [c for c in ['descrição', 'solução'] if c in df.columns]
    original_len = len(df)
    df = df.dropna(subset=essential_cols)
    removed = original_len - len(df)
    if removed > 0:
        print(f"Removidas {removed} linhas com valores nulos")
    
    print(f"Shape final: {df.shape}")
    print(f"Colunas finais: {list(df.columns)}")
    
    df.to_csv(output_path, index=False)
    print(f"Dataset salvo em: {output_path}")
    
    return df




if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(this_dir, "dataset", "dataset_original.csv")
    output_path = os.path.join(this_dir, "dataset", "preprocessed.csv")
    
    preprocess_dataset(input_path, output_path)
