# MECAL Module

Módulo isolado para testes unitários do algoritmo **MECAL** (Mascaramento por Clusterização de Embeddings e Avaliação Humana).

## Estrutura

```
MECAL_module/
├── datasets/
│   ├── dataset_original.csv     # Input: dataset OpenVAS original
│   ├── preprocessed.csv         # Output: após preprocessing
│   └── after_MECAL.csv          # Output: após MECAL
├── privacy_parameters/
│   ├── privacy_analysis.py      # Experimentos de privacidade
│   └── results/                 # Resultados dos experimentos
├── preprocessing.py             # Script de pré-processamento
├── MECAL.py                     # Algoritmo MECAL
└── README.md                    # Este arquivo
```

## Fluxo de Execução

### 1. Pré-processamento

Renomeia as colunas do dataset OpenVAS:
- `definition.description` → `descrição`
- `definition.solution` → `solução`

```bash
python preprocessing.py
```

**Input:** `datasets/dataset_original.csv`  
**Output:** `datasets/preprocessed.csv`

### 2. MECAL

Aplica clusterização semântica nas colunas de texto com **rotulagem automática via LLM (Ollama)**:
- `descrição` → `classe` (categorias de vulnerabilidade)
- `solução` → `tipo_solução` (tipos de solução)

**Pré-requisitos:**
1. Ollama instalado e rodando (`ollama serve`)
2. Modelo disponível (default: `llama3.2`)

```bash
# Instalar modelo (se necessário)
ollama pull llama3.2

# Executar MECAL
python MECAL.py
```

**Opções avançadas:**
```bash
python MECAL.py --model llama3.2 --clusters 20 --device cuda:0
```

| Parâmetro | Descrição | Default |
|-----------|-----------|---------|
| `--model`, `-m` | Modelo Ollama a usar | `llama3.2` |
| `--clusters`, `-c` | Número inicial de clusters | `20` |
| `--device`, `-d` | Dispositivo para embeddings | `cuda:0` |

**Input:** `datasets/preprocessed.csv`  
**Output:** 
- `datasets/after_MECAL.csv` - Dataset anonimizado
- `datasets/after_MECAL_labels.json` - Resumo dos labels gerados

### 3. Análise de Privacidade

Executa experimentos de métricas de privacidade:
- Entropia
- Valores únicos
- Mutual Information
- Classes de equivalência

```bash
cd privacy_parameters
python privacy_analysis.py
```

**Input:** `datasets/preprocessed.csv` + `datasets/after_MECAL.csv`  
**Output:** `privacy_parameters/results/`

## Dependências

```
pandas
numpy
scikit-learn
sentence-transformers
scipy
matplotlib
requests
```

## Instalação

```bash
pip install pandas numpy scikit-learn sentence-transformers scipy matplotlib
```

## Execução Completa

```bash
cd MECAL_module
python preprocessing.py
python MECAL.py
cd privacy_parameters
python privacy_analysis.py
```

## Métricas de Saída

O arquivo `privacy_parameters/results/privacy_metrics.json` contém:

- **entropy**: Redução de entropia (bits de informação)
- **uniqueness**: Redução de valores únicos
- **mutual_information**: Vazamento de informação entre texto original e categoria
- **equivalence_classes**: Tamanho dos grupos de anonimização

As visualizações são salvas como:
- `entropy_comparison.png` - Comparação de entropia
- `uniqueness_comparison.png` - Comparação de valores únicos
- `privacy_summary.png` - Tabela resumo
