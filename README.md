# Mascaramento por Agrupamento e Rotulagem com LLMs para Compartilhamento de Datasets de Incidentes em Redes

**Artigo:** Mascaramento por Agrupamento e Rotulagem com LLMs para Compartilhamento de Datasets de Incidentes em Redes

**Resumo:** A disseminação de bases de dados de segurança de redes é frequentemente limitada por atributos sensíveis presentes em *logs* textuais de ferramentas como OpenVAS e Nessus. Este trabalho propõe o algoritmo MECAL (Mascaramento por Clusterização de Embeddings e Rotulagem Automática) para anonimizar esses atributos preservando sua utilidade. O método utiliza Transformers para agrupar semanticamente as descrições de incidentes e emprega LLMs (*Large Language Models*) para gerar rótulos genéricos de alto nível para cada grupo. Os resultados demonstram que a substituição dos textos originais por rótulos gerados melhora a qualidade dos dados, como evidenciado pelo aumento das métricas de F1-Score e Mutual Information, viabilizando o compartilhamento seguro de informações de defesa cibernética.

---

# Estrutura do README

| Seção | Descrição |
|-------|-----------|
| [Selos Considerados](#selos-considerados) | Selos solicitados para este artefato |
| [Informações Básicas](#informações-básicas) | Requisitos de hardware e software |
| [Dependências](#dependências) | Pacotes e versões necessárias |
| [Preocupações com Segurança](#preocupações-com-segurança) | Riscos e cuidados na execução |
| [Instalação](#instalação) | Como instalar e configurar o ambiente |
| [Teste Mínimo](#teste-mínimo) | Execução básica para validar a instalação |
| [Experimentos](#experimentos) | Reprodução dos resultados do artigo |
| [Execução com GPU Completa](#execução-com-gpu-completa-opcional) | Opção para execução com GPU nos embeddings |
| [LICENSE](#license) | Licença do artefato |

## Estrutura do Repositório

```
MECAL/
├── dataset/                             # Não versionado — montar como volume no Docker
│   ├── dataset_original.csv             # Input: dataset OpenVAS original
│   ├── preprocessed.csv                 # Gerado por preprocessing.py
│   └── after_MECAL.csv                  # Gerado por MECAL.py
├── privacy_parameters/
│   ├── privacy_analysis.py              # Experimentos de privacidade
│   └── results/                         # Saídas geradas
├── computational_cost/
│   ├── cost_analysis.py                 # Experimento de custo computacional
│   ├── formula.py                       # Regressão OLS e fórmula analítica
│   └── results/                         # Saídas geradas
├── preprocessing.py                     # Pré-processamento do dataset
├── MECAL.py                             # Algoritmo principal
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

# Selos Considerados

Os selos considerados são: **Disponíveis (SeloD)**, **Funcionais (SeloF)**, **Sustentáveis (SeloS)** e **Experimentos Reprodutíveis (SeloR)**.

---

# Informações Básicas

## Hardware

| Componente | Mínimo | Referência (resultados do artigo) |
|------------|--------|-----------------------------------|
| CPU | x86-64 moderno | Intel Core i9-9900 |
| RAM | 4 GB | 32 GB |
| GPU | Não obrigatória, mas altamente recomendada para o Ollama¹ | NVIDIA GeForce RTX 4070 12 GB (CUDA 12.8, driver 571.96) |
| Armazenamento | ~10 GB livres | — |

> O modelo `llama3.1` (~4.7 GB) é baixado automaticamente pelo Ollama.
>
> ¹ O Ollama pode rodar em CPU, mas o tempo de inferência por cluster é significativamente maior. Para um dataset de tamanho típico com 20 clusters, a execução em CPU pode levar várias horas. GPU é altamente recomendada para obter resultados em tempo razoável.

## Software

| Componente | Versão |
|------------|--------|
| Sistema Operacional | Linux, Windows 10+ ou macOS |
| Docker Desktop | ≥ 4.0 (Windows/macOS) |
| Docker Engine | ≥ 20.10 (Linux) |
| Ollama | ≥ 0.1 |
| Python (execução local) | ≥ 3.10 |
| CUDA (execução local com GPU, opcional) | ≥ 12.0 |

---

# Dependências

## Execução via Docker (recomendado)

Todas as dependências Python são instaladas automaticamente no container a partir do `requirements.txt`:

| Pacote | Versão testada | Função |
|--------|----------------|--------|
| sentence-transformers | ≥ 2.2 | Geração de embeddings via Transformers |
| scikit-learn | ≥ 1.3 | K-Means e métricas |
| torch (CPU) | ≥ 2.0 | Backend dos Transformers |
| pandas | ≥ 2.0 | Manipulação de dados |
| numpy | ≥ 1.24 | Operações numéricas |
| scipy | ≥ 1.10 | Análise estatística |
| matplotlib | ≥ 3.7 | Geração de gráficos |
| requests | ≥ 2.28 | Comunicação com a API do Ollama |

## Modelo LLM

| Recurso | Detalhes |
|---------|----------|
| Ollama | Instalação nativa no host — [https://ollama.com](https://ollama.com) |
| Modelo | `llama3.1` (~4.7 GB de download) |
| Porta | `11434` (padrão do Ollama, deve estar livre no host) |

O Ollama detecta e usa automaticamente a GPU do host, caso disponível (NVIDIA com drivers instalados).

## Dataset

O dataset original do OpenVAS **não pode ser distribuído** por conter informações sensíveis de infraestrutura real. Para viabilizar a reprodução dos experimentos, o repositório inclui um **dataset sintético** gerado com estrutura compatível com o dataset original.

As métricas de privacidade obtidas com o dataset sintético estão dentro de 5% dos valores reportados no artigo:

| Métrica | Dataset sintético | Artigo | Diferença |
|---------|:-----------------:|:------:|:---------:|
| H(descrição) — Entropia | 9.9644 | 9.96 | < 0.1% |
| H(solução) — Entropia | 5.9556 | 5.92 | < 0.6% |
| Valores únicos (descrição) | 2607 | 2722 | ~4.2% |
| Valores únicos (solução) | 918 | 946 | ~3.0% |
| Unicidade (descrição) | 0.1717 | 0.1792 | ~4.2% |
| Unicidade (solução) | 0.0604 | 0.0623 | ~3.0% |

O arquivo `dataset/dataset_original.csv` mencionado nas instruções de instalação refere-se a este dataset sintético, já disponível no repositório.

---

# Preocupações com Segurança

Não há riscos relevantes para os revisores.

- O modelo LLM (`llama3.1`) executa **inteiramente de forma local** via Ollama. Nenhum dado é enviado a serviços externos.
- O artefato não abre portas de rede além de `localhost:11434` (Ollama) e da rede interna do Docker.
- O dataset incluído no repositório é **sintético** e não contém informações de sistemas reais de produção.

---

# Instalação

## Opção 1: Docker (recomendado)

O `docker-compose.yml` gerencia apenas o container MECAL. O Ollama roda nativamente no host e é acessado pelo container via `host.docker.internal:11434`. Essa arquitetura evita a necessidade do NVIDIA Container Toolkit: o Ollama usa a GPU do host diretamente, enquanto os embeddings rodam com CPU no container. Note que os experimentos de custo computacional foram utilizados totalmente em GPU, incluindo embeddings. Portanto, para reproduzir estes experimentos especificamente, é recomendado utilizar a opção descrita na seção "**Execução com GPU Completa (opcional)**".

### Pré-requisitos

1. Instalar o [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Instalar o [Ollama](https://ollama.com) e iniciá-lo (o container acessa o Ollama via `host.docker.internal`, por isso é necessário escutar em todas as interfaces):

```bash
# Windows (em um terminal separado):
set OLLAMA_HOST=0.0.0.0:11434 && ollama serve

# Linux/macOS (em um terminal separado):
OLLAMA_HOST=0.0.0.0:11434 ollama serve

ollama pull llama3.1
```

3. Clonar o repositório (o dataset sintético já está incluído):

```bash
git clone <URL_DO_REPOSITÓRIO>
cd MECAL
```

4. **Linux apenas:** descomente a seção `extra_hosts` no `docker-compose.yml`:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

### Build e execução

```bash
docker compose up --build
```

---

## Opção 2: Execução Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar Ollama (em um terminal separado)
# Windows:
set OLLAMA_HOST=0.0.0.0:11434 && ollama serve
# Linux/macOS:
OLLAMA_HOST=0.0.0.0:11434 ollama serve

ollama pull llama3.1

# Executar
python preprocessing.py
python MECAL.py
```

---

# Teste Mínimo

Após a instalação, execute o fluxo completo via Docker:

```bash
docker compose up --build
```

**Saída esperada:**
1. Build do container Python (~2 min na primeira vez)
2. Log de pré-processamento com contagem de registros processados
3. Log do MECAL: geração de embeddings → K-Means → chamadas ao Ollama por cluster → substituição dos textos
4. Arquivos gerados em `./dataset/`:
   - `preprocessed.csv` — dataset após limpeza
   - `after_MECAL.csv` — dataset anonimizado
   - `after_MECAL_labels.json` — rótulos gerados pelo LLM por coluna

**Tempo esperado:** 5–30 minutos dependendo do hardware (CPU para embeddings, GPU do host para Ollama).

---

# Experimentos

> **Nota sobre o dataset:** o dataset original do OpenVAS não pode ser distribuído por razões de segurança. Os experimentos utilizam o **dataset sintético** incluído no repositório (`dataset/dataset_original.csv`), cujas métricas de privacidade diferem em no máximo 5% dos valores reportados no artigo — conforme detalhado na seção [Dependências › Dataset](#dataset).

## Reivindicação 1 — Anonimização MECAL reduz identificadores preservando utilidade semântica

O MECAL substitui textos livres por rótulos semânticos gerados por LLM. A análise de privacidade quantifica o impacto dessa substituição nas métricas de entropia, unicidade, Mutual Information e classes de equivalência.

### Execução

```bash
# Passo 1: gerar dataset anonimizado (via Docker)
docker compose up --build

# Passo 2: análise de privacidade
docker compose run --rm mecal python privacy_parameters/privacy_analysis.py
```

Ou localmente:

```bash
python preprocessing.py
python MECAL.py
python privacy_parameters/privacy_analysis.py
```

**Resultados esperados em `privacy_parameters/results/`:**

| Arquivo | Conteúdo |
|---------|----------|
| `entropy_comparison.png` | Comparação de entropia antes/depois do MECAL |
| `uniqueness_comparison.png` | Redução de valores únicos (identificadores) |
| `privacy_summary.png` | Resumo das métricas de privacidade |

**Tempo esperado:** 5–30 min (MECAL) + ~1 min (análise)
**Recursos:** ~2 GB RAM

---

## Reivindicação 2 — Custo computacional segue a fórmula analítica T_MECAL(n, k)

O custo do MECAL é modelado como:

```
T_MECAL(n, k) = α_embed · n  +  α_kmeans · n·k  +  μ_llm · k  +  C
```

| Componente | Escala | Descrição |
|------------|--------|-----------|
| `α_embed · n` | O(n) | Encoding em batches, linear no número de registros |
| `α_kmeans · n·k` | O(n·k) | K-Means: O(n·k·d·i) com dimensão d e iterações i fixos |
| `μ_llm · k` | O(k) | Uma chamada ao Ollama por cluster |
| `C` | O(1) | Overhead fixo |

### Execução

```bash
# Coleta de dados (31 runs por fração de dataset, 1 descartado como warmup)
python computational_cost/cost_analysis.py --device cuda:0 --model llama3.1

# Regressão OLS e derivação da fórmula
python computational_cost/formula.py
```

**Resultados esperados em `computational_cost/results/`:**

| Arquivo | Conteúdo |
|---------|----------|
| `cost_metrics.json` | Tempos brutos e estatísticas com IC 95% |
| `formula_coefficients.json` | Coeficientes com incertezas a 95%, validação e previsões |
| `formula_analysis.png` | Gráficos de ajuste e curvas T_MECAL(n, k) |

> **Nota:** os coeficientes de referência em `computational_cost/results/` foram obtidos em uma **NVIDIA GeForce RTX 4070 12 GB** (CUDA 12.8, driver 571.96). Os valores em outros ambientes serão diferentes, mas a estrutura linear/quadrática da fórmula deve se manter.

**Tempo esperado:** ~2 horas (31 runs × múltiplas frações de dataset)
**Recursos:** GPU recomendada; ~1 GB RAM

---

# Execução com GPU Completa (opcional)

A configuração padrão usa CPU para os embeddings e delega a GPU ao Ollama no host. Para executar **tudo em GPU** (embeddings via CUDA + Ollama com GPU), use a execução local sem Docker:

### Pré-requisitos

- NVIDIA GPU com drivers atualizados e CUDA instalado
- [Ollama](https://ollama.com) instalado (detecta e usa a GPU automaticamente)
- Python 3.10+

### Passos

```bash
# Instalar torch com CUDA (não usar requirements.txt neste caso)
pip install torch sentence-transformers pandas numpy scikit-learn scipy matplotlib requests

ollama serve          # em um terminal separado
ollama pull llama3.1

python preprocessing.py
python MECAL.py --device cuda:0
```

O parâmetro `--device cuda:0` direciona os embeddings para a GPU. Múltiplas GPUs podem ser especificadas com `cuda:1`, `cuda:2`, etc.

> O `requirements.txt` aponta para o índice CPU-only do PyTorch (adequado ao container Docker). Para uso local com GPU, instale o `torch` diretamente pelo PyPI conforme o comando acima.

---

# LICENSE

Este artefato é distribuído sob a licença **MIT**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
