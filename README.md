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

Prever o volume de trafego de uma praca de pedagio no mes seguinte (t+1) a partir da evolucao
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
