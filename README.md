# Previsao de Volume de Trafego em Pracas de Pedagio

Trabalho Pratico da disciplina de Deep Learning - serie temporal multivariada.

| Item | Valor |
|---|---|
| Integrantes | Daniel Flavio, Guilherme Martins |
| Frente | 3.2 Redes Recorrentes (SimpleRNN vs GRU vs LSTM) |
| Tarefa | Regressao (serie temporal multivariada) |
| Variavel alvo | Volume de trafego da praca no dia t+1 |
| Entrega | ate 15/10 (relatorio + GitHub + estudo de hiperparametros + tabelas/graficos) |
| Plano | [PLANO_DE_TAREFAS.md](PLANO_DE_TAREFAS.md) |
| Decisoes | [docs/decisoes_dados.md](docs/decisoes_dados.md) |

## 1. Objetivo

Prever o volume de trafego de uma praca de pedagio no dia seguinte (t+1) a partir da evolucao
temporal de multiplos atributos de entrada. O objetivo secundario e comparar **SimpleRNN, GRU e
LSTM** sob condicoes equivalentes (desempenho x capacidade x custo computacional).

## 2. Dados

- **Fonte:** ANTT - Volume de Trafego nas Pracas de Pedagio (dados.gov.br).
- **Recorte:** 3 arquivos anuais - 2024, 2025 e 2026 (~300 MB cada), unificados numa serie continua.
- **Copia compartilhada:** Google Drive (links abaixo). Os arquivos **nao** entram no Git.
- `download_data.py` baixa os arquivos e registra URL, data de download e **SHA-256** (reprodutibilidade).

| Ano | Link (Google Drive) |
|---|---|
| 2024 | https://drive.google.com/file/d/1vUHJ-L4f9kRrRmRa5qlfdb68jSn5oOqN/view |
| 2025 | https://drive.google.com/file/d/1hDvgQXU4UvhpE3dfSUxx72fO3Phter3t/view |
| 2026 | https://drive.google.com/file/d/1ej0hH6aIV3HvMaoDd9jXfSZJd2hF8lzw/view |

Caminhos locais (fora do Git): `data/raw/*.csv` (originais) e `data/trafego_pedagio.parquet` (derivado).

## 3. Ambiente

Requer **Python 3.11+**. Se o TensorFlow nao tiver wheel para a sua versao, crie o venv com uma
suportada (ex.: `py -3.12 -m venv .venv`).

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell (CMD: .venv\Scripts\activate.bat | Linux/macOS: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import tensorflow as tf; print('TF', tf.__version__)"   # verificacao (WARNING nao e erro)
```

Registro exato do ambiente testado: `requirements.lock.txt`. No VS Code, selecione o interpretador
de `.venv` (`Ctrl+Shift+P` > "Python: Select Interpreter"); as extensoes recomendadas (**Python**,
**Pylance**, **Jupyter**) sao oferecidas ao abrir a pasta (`.vscode/extensions.json`).

### nbstripout: notebooks sem saida no Git

O repositorio versiona notebooks **sem as saidas**. O arquivo no seu disco **mantem** as saidas; o
Git recebe apenas a versao limpa. Depois de instalar as dependencias, rode **uma vez por maquina**:

```bash
nbstripout --install --attributes .gitattributes
nbstripout --status    # confirma que instalou
```

- Sem esse passo, o `git add` de qualquer `.ipynb` **falha** (o filtro e obrigatorio).
- Para limpar um notebook **ja commitado**: `git add --renormalize notebooks/arquivo.ipynb`.

### 3.1 Trocar de maquina / rodar do zero

Os dados NAO vem no clone (`data/` esta no `.gitignore`). Para reconstruir tudo na outra maquina:

```bash
git pull
pip install -r requirements.txt
nbstripout --install --attributes .gitattributes   # so uma vez por maquina
python download_data.py      # baixa os 3 CSVs e confere o SHA-256
python src/preprocess.py     # gera data/trafego_pedagio.parquet
python src/dataset.py        # gera a serie praca x dia e as janelas
```

Ja vem versionado (nao precisa reconstruir): `results/` (tabelas e figuras), `report/`,
`slides/` e `docs/`.

## 4. Estrutura

```text
data/             dados brutos e derivados (fora do Git)
notebooks/        exploracao e analises (EDA)
src/              preprocess.py, dataset.py, model.py, train.py, optuna_search.py
results/          tabelas, figuras e export da busca do Optuna
report/           relatorio tecnico
docs/             enunciado, material de apoio e registro de decisoes
download_data.py  download + manifesto (URL, data, SHA-256)
```

## 5. Convencoes

- **Seed global:** `GLOBAL_SEED = 42` em toda inicializacao, embaralhamento e amostragem.
- **Sem shuffle** em qualquer etapa que envolva a ordem temporal.
- **Dados e artefatos gerados nao entram no Git** (ver `.gitignore`); versionamos exports legiveis em `results/`.
- **Pre-processamento ajustado somente no treino** e aplicado em validacao/teste.
- **ASCII puro** em arquivos de configuracao e documentacao.

## 6. Status

Acompanhe as fases F0-F8 em [PLANO_DE_TAREFAS.md](PLANO_DE_TAREFAS.md). Decisoes sobre os dados
ficam em [docs/decisoes_dados.md](docs/decisoes_dados.md). Desvios de planejamento sao registrados aqui.

- **2026-10-06 - F0 e F1 concluidas.**
- F1 gerou `data/trafego_pedagio.parquet` (9.184.154 linhas, ~20 MB) via `src/preprocess.py`.
- Achados da F1: encoding **Latin-1**; `mes_ano` e **diario, nao mensal**; drift em
  `categoria_eixo` (codigo vs rotulo em Q1/2025) e 4.853 chaves repetidas em Q1/2025 (somadas na
  agregacao). Ver D-001..D-008 em `docs/decisoes_dados.md`.
- Pendencia operacional: apos clonar/puxar, cada integrante roda
  `nbstripout --install --attributes .gitattributes` (ver secao 3).

- **2026-10-13 - F2 a F7 executadas; F8 (relatorio/slides) em andamento.**
- F2 (`src/eda.py`, `src/eda_cobertura.py`): periodo 2024-01-01 a 2026-08-31 (974 dias);
  43 concessionarias e 345 pares `concessionaria|praca`; `volume_total` **sem zeros**, skew 14,9;
  serie descontinua (303 pracas com >= 180 dias; 47 cobrem os 974 dias). 4 figuras em `results/figures/`.
- F3 (`src/dataset.py`): serie `praca x dia` (180.737 linhas), janelas `X (window x 13)` e
  `y = Total t+1`, dois splits (cronologico e por grupos), padronizacao **so no treino**.
  Testes de vazamento temporal passando.
- F4 (`src/model.py`, `src/train.py`): SimpleRNN/GRU/LSTM + baselines. **A persistencia e um
  baseline forte** (val MAE ~1.470, R2 ~0,94): referencia obrigatoria para as RNNs.
- F5 (`src/optuna_search.py`): busca com Optuna, metrica RMSE **na validacao**, MedianPruner.
- Decisoes de desenho registradas: **D-009 a D-015** (anomalias incluidas em D-015).
- F6 (`src/experimentos.py`, dados completos): familias **GRU 1.537 / LSTM 1.550 / SimpleRNN 1.566**
  (MAE na validacao) - empate tecnico; janela **30 dias** melhor; **2 camadas** melhor que 1;
  **128 unidades** melhoram o MAE com **12x mais parametros** (retorno decrescente).
- **Achado importante:** a config vencedora da busca, retreinada com dados completos, rendeu
  **pior** que a config de referencia do piloto. Evidencia, com numero, de que o orcamento da
  busca (CPU-only) foi pequeno. O modelo final usa a **config de referencia** (`results/config_final.json`).
- F7 (`src/avaliar_teste.py`): no **teste**, MAE **1.126** (persistencia 1.536) no split cronologico
  e MAE **1.469** (persistencia 1.774) no split por pracas - o modelo bate a regua nos dois cenarios.
- Apresentacao: `slides/apresentacao.html` e `slides/apresentacao.pdf` (14 slides, 16:9).
- Relatorio: `report/relatorio.md`. Anomalias: `results/anomalias_por_praca.csv`.
- **Pendencias de entrega (ate 15/10):** ensaiar a apresentacao (15 min) e deixar o
  **link do GitHub publico** para o professor.
