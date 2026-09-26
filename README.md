# Previsao de Volume de Trafego em Pracas de Pedagio

Trabalho Pratico da disciplina de Deep Learning - modelagem, hiperparametrizacao e avaliacao experimental.

| Item | Valor |
|---|---|
| Integrantes | Daniel Flavio, Guilherme Martins |
| Frente | 3.2 Redes Recorrentes (SimpleRNN vs GRU vs LSTM) |
| Tarefa | Regressao (serie temporal multivariada) |
| Variavel alvo | Volume de trafego da praca no mes t+1 |
| Entrega | ate 15/10 (relatorio + GitHub + estudo de hiperparametros + tabelas/graficos) |
| Plano de tarefas | [PLANO_DE_TAREFAS.md](PLANO_DE_TAREFAS.md) |

## 1. Objetivo

Prever o volume de trafego de uma praca de pedagio no mes seguinte (t+1) usando a evolucao
temporal de multiplos atributos de entrada (serie temporal multivariada). O objetivo secundario
e comparar as familias recorrentes **SimpleRNN, GRU e LSTM** sob condicoes experimentais
equivalentes, relacionando desempenho, capacidade e custo computacional.

## 2. Fonte dos dados

- **Conjunto:** ANTT - Volume de Trafego nas Pracas de Pedagio
- **Portal:** Portal Brasileiro de Dados Abertos (dados.gov.br)
- **Recorte:** 3 arquivos anuais - 2024, 2025 e 2026 (aprox. 300 MB cada), unificados em uma
  unica serie continua.
- Os arquivos **nao** ficam versionados no Git. A origem (URL), a data de download e o
  **SHA-256** de cada arquivo sao registrados por `download_data.py` para reprodutibilidade.

## 3. Onde os dados ficam

> **DECISAO PENDENTE (F0):** definir a copia compartilhada entre os integrantes -
> Hugging Face privado **ou** Google Drive compartilhado.

Caminhos locais esperados:

```text
data/raw/                 arquivos originais (csv) - ignorados pelo Git
data/trafego_pedagio.parquet   dataset derivado, ja harmonizado - ignorado pelo Git
```

## 4. Ambiente e instalacao

- **Python:** 3.11+ recomendado. O TensorFlow pode nao publicar wheel para toda versao do
  Python; se o `pip install` falhar (ex.: no Python 3.13), crie o venv com uma versao suportada:
  `py -3.12 -m venv .venv`.
- As dependencias ficam em `requirements.txt` (framework: **TensorFlow/Keras**).

```bash
# 1. criar o ambiente virtual
python -m venv .venv

# 2. ativar (use o comando do SEU terminal)
#    PowerShell:
.\.venv\Scripts\Activate.ps1
#    CMD:
.venv\Scripts\activate.bat
#    Git Bash (Windows):
source .venv/Scripts/activate
#    Linux / macOS:
source .venv/bin/activate

# 3. atualizar o pip e instalar as dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. verificar a instalacao
python -c "import tensorflow as tf; print('TF', tf.__version__)"

# 5. (opcional) fixar as versoes exatas depois que tudo funcionar
pip freeze > requirements.txt
```

> Apos ativar, o prompt exibe `(.venv)`. Para sair, use `deactivate`.
> No VS Code, selecione o interpretador em `Ctrl+Shift+P` > "Python: Select Interpreter" >
> escolha o que aponta para `.venv`.

## 5. Estrutura do repositorio

```text
data/            dados brutos e derivados (fora do Git)
notebooks/       exploracao e analises (EDA)
src/             preprocess.py, dataset.py, model.py, train.py, optuna_search.py
results/         tabelas, figuras e export da busca do Optuna
report/          relatorio tecnico
docs/            enunciado e material de apoio
download_data.py  registrar origem, data e SHA-256 dos arquivos
requirements.txt  dependencias do projeto
README.md         este arquivo
```

## 6. Convencoes do projeto

- **Seed global:** `GLOBAL_SEED = 42`. Toda inicializacao de pesos, embaralhamento e amostragem
  deve usar essa seed, para que os experimentos sejam reproduziveis.
- **Dados e artefatos gerados NAO entram no Git** (ver `.gitignore`): dados, modelos treinados
  e o banco cru do Optuna. O que versionamos sao os **exports legiveis** em `results/`.
- **Sem embaralhamento (shuffle)** em nenhuma etapa que envolva a ordem temporal.
- **ASCII puro** em arquivos de configuracao e documentacao.
- Transformacoes de pre-processamento sao ajustadas **somente no treino** e aplicadas em
  validacao e teste.

## 7. Status

Acompanhe o andamento pelas fases F0-F8 no [PLANO_DE_TAREFAS.md](PLANO_DE_TAREFAS.md).
Desvios de planejamento devem ser registrados aqui.
