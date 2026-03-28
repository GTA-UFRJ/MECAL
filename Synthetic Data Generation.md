Vou analisar o artigo em detalhes para criar um plano preciso. Deixa eu mapear todas as métricas e estruturas necessárias.

## Plano de Geração de Dataset Sintético OpenVAS para Replicar Resultados do MECAL

---

### Visão Geral da Tarefa

O objetivo é gerar um dataset sintético de 15.187 registros OpenVAS que, ao ser processado pelo algoritmo MECAL, reproduza os resultados publicados no artigo. Isso significa que **as propriedades estatísticas devem estar embutidas na geração**, não nos algoritmos de avaliação.

---

### 1. Estrutura do Dataset

**Colunas obrigatórias:**

| Coluna | Tipo | Descrição |
|---|---|---|
| `descricao` | texto livre | Descrição técnica da vulnerabilidade (alta entropia) |
| `solucao` | texto livre | Solução recomendada (entropia menor que descrição) |
| `severidade` | float | CVSS score (0.0–10.0) |
| `ip_address` | string | Endereço IP privado (quasi-identificador) |
| `porta` | int | Porta de rede |
| `protocolo` | string | TCP/UDP |
| `host` | string | Nome do host |
| `cve_id` | string | Referência CVE (quando aplicável) |

**Dimensões:** 15.187 linhas × ~8 colunas

---

### 2. Distribuição Semântica Alvo (14 Classes)

O dataset deve ser gerado de forma que o K-means com embeddings all-MiniLM-L6-v2 e K=20 convirja para **14 classes semânticas únicas** após rotulagem por LLM. As classes e tamanhos aproximados (baseados na Tabela 4: média=194.4, min=143 "SNMP", max=690) são:

| Classe Semântica | Tamanho Aprox. | Notas |
|---|---|---|
| SNMP Configuration Issues | 143 | menor grupo (kmin) |
| Outdated SSL/TLS Libraries | ~690 | maior grupo |
| Buffer Overflow in Kernel | ~350 | grupos 1 e 5 do artigo convergem aqui |
| Weak Ciphers Configuration | ~250 | |
| Expired Certificates | ~200 | |
| Driver Compatibility Issues | ~180 | |
| SQL Injection Vulnerabilities | ~190 | |
| Cross-Site Scripting (XSS) | ~175 | |
| Outdated Software Packages | ~620 | |
| DNS Configuration Issues | ~160 | |
| HTTP Security Headers Missing | ~200 | |
| Remote Code Execution Risk | ~400 | |
| Authentication Bypass | ~165 | |
| Privilege Escalation | ~465 | |

**Total: ~15.187** — ajustar proporcionalmente para bater exatamente.

Os grupos 1 e 5 do K-means (mapeando para "Buffer Overflow in Kernel") precisam ter descrições **semanticamente similares mas sintaticamente diferentes**, para simular a convergência de rótulos descrita na Tabela 1.

---

### 3. Propriedades Estatísticas que Devem ser Alcançadas

Estas são as métricas-alvo extraídas das Tabelas 3, 4 e 5 do artigo. O dataset deve ser construído de forma que, ao rodar o MECAL, os resultados batem:

#### 3.1 Métricas de Privacidade (Tabela 3)

**Atributo `descricao` → `classe`:**
- Entropia de Shannon original: **9,96 bits**
- Entropia após MECAL: **3,46 bits**
- Valores únicos originais: **2.722**
- Valores únicos após MECAL: **14**
- Razão de unicidade original: **17,92%** → 2.722 únicos / 15.187 total ≈ 17,93% ✓

**Atributo `solucao` → `tipo_solucao`:**
- Entropia de Shannon original: **5,92 bits**
- Entropia após MECAL: **3,16 bits**
- Valores únicos originais: **946**
- Valores únicos após MECAL: **20**
- Razão de unicidade original: **6,23%** → 946 / 15.187 ≈ 6,23% ✓

#### 3.2 Distribuição das Classes (Tabela 4)

**Para `descricao`:**
- Total de grupos: **14**
- Tamanho médio: **194,4** registros
- kmin (menor grupo): **143** ("SNMP")
- Maior grupo: **690**

**Para `solucao`:**
- Total de grupos: **20**
- Tamanho médio: **47,3** registros (note: 15.187 / 20 ≈ 759 — este número parece ser da distribuição de soluções únicas, não registros)
- kmin: **35** ("Configure HTTP policies and HSTS")
- Maior grupo: **240**

> **Nota importante:** Os 20 grupos de `solucao` com média 47,3 somam ~946 registros únicos de texto, coerente com os 946 valores únicos originais sendo agrupados em 20 categorias. A geração deve criar 946 textos únicos de solução distribuídos em 20 famílias semânticas.

#### 3.3 Métricas de Utilidade (Tabelas 5 e 6)

**Detecção de Anomalias:**
- F1-Score MECAL: **0,7511**
- RMSE MECAL: **0,4262**
- F1-Score Supressão: **0,7094**
- RMSE Supressão: **0,4614**

**Qualidade de Agrupamento (HDBSCAN):**

| Cenário | DBCV | Silhouette | ARI | NMI |
|---|---|---|---|---|
| A: Baseline (texto completo) | 0,43 | 0,75 | 1,00 | 1,00 |
| B: MECAL | 0,52 | 0,82 | 0,29 | **0,84** |
| C: Supressão | 0,52 | 0,83 | - | - |

---

### 4. Estratégia de Geração dos Textos

#### 4.1 Geração das Descrições (2.722 textos únicos)

**Estrutura típica de um registro OpenVAS:**
```
[Software/Service] [version X.X.X] on [OS] is affected by [CVE-XXXX-XXXXX]. 
[Technical description of the flaw]. An attacker could [impact]. 
Affected: [path/config file]. Port: [N]/[protocol].
```

**Requisitos por classe semântica:**
- Cada classe deve ter N textos únicos proporcionais ao tamanho do grupo
- Variações sintáticas dentro da mesma classe (versões diferentes, sistemas diferentes, caminhos diferentes) mas semanticamente coesos
- Quasi-identificadores embutidos: IPs privados (10.x.x.x, 192.168.x.x, 172.16.x.x), caminhos (`/etc/ssl/`, `/var/www/`, `C:\Windows\`), versões de software específicas
- A classe "SNMP" deve ter descrições suficientemente distintas das demais para ser o menor grupo natural

**Para atingir Entropia=9,96 bits no campo `descricao`:**
- H = 9,96 implica que cada texto é quase único: -Σ P(xi)·log2(P(xi)) ≈ 9,96
- Com 2.722 únicos em 15.187 registros, ~12.465 registros são repetições
- Portanto: gerar 2.722 templates únicos e replicá-los com frequências desiguais seguindo distribuição log-normal (vulnerabilidades comuns aparecem muito, raras aparecem pouco)

**Distribuição de frequências para `descricao`:**
- ~500 textos aparecem 1 vez (raros)
- ~800 textos aparecem 2-3 vezes
- ~900 textos aparecem 4-10 vezes
- ~400 textos aparecem 11-30 vezes
- ~122 textos aparecem 31+ vezes
- Total deve somar 15.187

#### 4.2 Geração das Soluções (946 textos únicos)

**Estrutura típica:**
```
Upgrade to [Software] version [X.X.X] or later. / 
Apply patch [KB/CVE]. Configure [setting] to [value].
```

**Para atingir Entropia=5,92 bits:**
- Com 946 únicos em 15.187 registros, maior repetição que descrições
- Distribuição mais concentrada: soluções comuns ("Upgrade OpenSSL", "Apply patches") aparecem centenas de vezes
- 20 famílias semânticas de solução, cada uma com ~47 variações textuais únicas

#### 4.3 Geração dos Atributos Estruturados

**CVSS (severidade):**
- Distribuição realista: maioria entre 4.0-7.9 (medium/high)
- ~15% críticos (CVSS ≥ 9.0), ~20% baixos (< 4.0)
- Correlação com classe: "Remote Code Execution" e "Buffer Overflow" devem ter CVSS mais altos

**IPs:**
- Faixas privadas: 10.0.0.0/8, 192.168.0.0/16, 172.16.0.0/12
- ~50-200 hosts únicos (simulando rede corporativa)
- Cada host pode ter múltiplas vulnerabilidades

---

### 5. Garantias para Replicabilidade das Métricas

#### 5.1 Para NMI=0,84 (métrica mais crítica)

O NMI alto entre clusters HDBSCAN no MECAL vs. baseline indica que as **classes semânticas são recuperáveis via agrupamento numérico (CVSS + metadados)**. Isso requer:
- Correlação real entre CVSS e classe semântica
- Classes com perfis de severidade distintos e consistentes
- Portas/protocolos correlacionados com tipo de vulnerabilidade (ex: SNMP sempre porta 161/UDP)

#### 5.2 Para Entropia Original=9,96 bits

Validar durante geração com:
```python
from scipy.stats import entropy
import numpy as np

value_counts = df['descricao'].value_counts(normalize=True)
h = -np.sum(value_counts * np.log2(value_counts))
assert abs(h - 9.96) < 0.05
```

#### 5.3 Para F1-Score=0,75 na detecção de anomalias

O pipeline PCA(25 componentes) + KDE deve identificar os mesmos top 5% outliers no MECAL e no baseline. Isso requer:
- Distribuição multimodal clara nos dados (não uniforme)
- Outliers reais: registros com combinações raras de (classe, CVSS, porta)
- ~759 registros (5% de 15.187) devem ser identificáveis como outliers em ambas as representações

---

### 6. Script de Geração — Arquitetura Recomendada

```
generate_synthetic_openvas/
├── main.py                    # Orquestrador principal
├── config.py                  # Parâmetros-alvo (todas as métricas)
├── generators/
│   ├── text_generator.py      # Geração dos textos OpenVAS
│   ├── metadata_generator.py  # IPs, portas, CVSS, CVEs
│   └── distribution.py        # Controle de frequências e entropia
├── validators/
│   ├── privacy_metrics.py     # Entropia, unicidade, cardinalidade
│   └── structural_metrics.py  # Verifica coerência semântica
└── output/
    └── openvas_synthetic.csv
```

**Fluxo do `main.py`:**
1. Definir as 14 classes semânticas com seus tamanhos
2. Para cada classe, gerar templates de descrição e solução
3. Replicar templates seguindo distribuição log-normal controlada
4. Gerar metadados correlacionados por classe
5. Embaralhar o dataset
6. Validar métricas de entropia e unicidade
7. Ajustar frequências iterativamente até convergir nas métricas-alvo
8. Exportar CSV

---

### 7. Validação Pós-Geração

Após gerar o dataset, rodar o pipeline completo do MECAL e verificar:

```python
TARGETS = {
    "descricao_entropia_original": 9.96,
    "solucao_entropia_original": 5.92,
    "descricao_valores_unicos": 2722,
    "solucao_valores_unicos": 946,
    "descricao_unicidade": 0.1792,
    "solucao_unicidade": 0.0623,
    "descricao_classes_mecal": 14,
    "solucao_classes_mecal": 20,
    "descricao_kmin": 143,
    "anomalia_f1_mecal": 0.7511,
    "anomalia_f1_supressao": 0.7094,
    "clustering_nmi_mecal": 0.84,
    "clustering_silhouette_mecal": 0.82,
}
TOLERANCE = 0.05  # 5% de margem para cada métrica
```

---

### 8. Dependências Python

```
sentence-transformers>=2.2.0   # all-MiniLM-L6-v2
scikit-learn>=1.3.0             # KMeans, PCA, silhouette
hdbscan>=0.8.33                 # clustering utilitário
scipy>=1.11.0                   # entropia, KDE
pandas>=2.0.0                   # manipulação do dataset
numpy>=1.24.0                   # operações numéricas
openai / anthropic               # LLM para rotulagem (Etapa 3 do MECAL)
```

---

### 9. Observações Críticas para o Claude Code

1. **A geração de textos deve usar templates parametrizados**, não LLM (para ser determinística e controlável). O LLM só entra na Etapa 3 do MECAL (rotulagem).

2. **O controle de entropia é iterativo**: gerar, medir, ajustar a distribuição de frequências até H ≈ 9,96 para descrições e H ≈ 5,92 para soluções.

3. **A correlação CVSS↔classe é obrigatória** para que o NMI=0,84 seja atingível — sem ela, os clusters numéricos não recuperam a estrutura semântica.

4. **Os 946 textos únicos de solução** mapeiam para 20 grupos, mas os registros totais são 15.187 — a maioria das soluções se repete muito (ex: "Upgrade OpenSSL" aparece em centenas de registros de classes diferentes).

5. **O dataset final deve ter random_state fixado** (seed=42 ou seed=0 seguindo o artigo) para reprodutibilidade total.