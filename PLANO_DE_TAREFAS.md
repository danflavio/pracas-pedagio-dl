# Plano de Tarefas - Trabalho Pratico de Deep Learning

**Tema:** Previsao de volume de trafego em pracas de pedagio (serie temporal multivariada)
**Frente escolhida:** 3.2 Redes Recorrentes (SimpleRNN vs GRU vs LSTM)
**Dados:** ANTT - Volume de Trafego nas Pracas de Pedagio, anos **2024 + 2025 + 2026** (mesmo assunto, 3 arquivos ~300 MB cada), obtidos no **dados.gov.br**
**Entrega:** ate **15/10** - relatorio tecnico + link do GitHub + registro da busca de hiperparametros + tabelas/graficos
**Apresentacao:** 15 min de slides
**Integrantes:** Daniel Flavio - Guilherme Martins (dupla)

---

## 0. Decisoes ja fechadas

- [x] Fonte de dados: Portal Brasileiro de Dados Abertos (dados.gov.br)
- [x] Colisao com outro grupo: resolvida
- [x] Frente: Redes Recorrentes (3.2)
- [x] Escopo dos dados: juntar os 3 arquivos (2024, 2025, 2026) em uma unica serie continua

## 1. Decisoes pendentes (resolver na Fase 2 - com padrao recomendado)

| Decisao | Recomendacao padrao | Responsavel | Prazo |
|---|---|---|---|
| Unidade de observacao | `concessionaria x praca x data x tipo_de_veiculo` (agregando sentido/cobranca/categoria) | A/B | 29/09 |
| Variavel alvo | Volume total da praca no dia **t+1** (regressao de 1 passo) | A/B | 29/09 |
| Estrategia de previsao | Principal: 1 passo a frente. Bonus: multi-step t+1..t+3 | A/B | 30/09 |
| Protocolo de split | **1o Cronologico global** (correto para prever o futuro) **+ 2o por grupos/pracas** (generalizacao p/ pracas novas) | A/B | 30/09 |
| Validacao cruzada | **Opcional (PDF Sec.2.3):** k-fold **temporal** (rolling-origin / janela expansiva, ex.: 3 folds) para estimar a **variancia** dos experimentos; a serie diaria cobre ~33 meses (~975 dias por praca) | A/B | 30/09 |
| Framework | TensorFlow/Keras (camadas `SimpleRNN`, `GRU`, `LSTM`) ou PyTorch | A/B | 27/09 |
| Repositorio | GitHub da dupla (dados fora do git) | A/B | 26/09 |

> **Justificativa do split (vai para o relatorio):** como o objetivo e prever o *futuro da mesma praca*, o split cronologico e legitimo (o futuro e genuinamente nao visto). O split por grupos e reportado como teste de generalizacao para **pracas novas**. Reportar os **dois** protocolos e discutir e o que garante os pontos de "analise critica".

---

## 2. Fases e tarefas

Legenda de responsaveis: **A** = Daniel Flavio - **B** = Guilherme Martins - **A+B** = os dois

> **Status em 2026-10-13.** F0-F3 concluidas. **F4 a F7 executadas** (numeros no `README.md`,
> secao 6, e em `results/`); F8 (relatorio + slides) em andamento. Os checkboxes abaixo refletem o
> **plano original** e nao foram todos marcados um a um; onde houve desvio de plano, o registro
> esta em `docs/decisoes_dados.md` (D-001 a D-015) e no README.

### F0 - Setup do projeto - 25-26/09

- [x] 26/09 - **B** - Criar repositorio no GitHub com a estrutura:
  ```
  data/            (no .gitignore)
  notebooks/
  src/             preprocess.py, dataset.py, model.py, train.py, optuna_search.py
  results/         tabelas, figuras, estudo do Optuna
  report/
  download_data.py
  requirements.txt
  README.md
  ```
- [x] 26/09 - **A** - Criar `.gitignore` com `data/`, `*.csv`, `*.parquet`, `*.h5`, `*.keras`, `.ipynb_checkpoints/`
- [x] 26/09 - **A+B** - Definir ambiente: `requirements.txt` (numpy, pandas, pyarrow, scikit-learn, tensorflow/torch, optuna, optuna-dashboard, matplotlib, seaborn)
- [x] 26/09 - **A+B** - Definir onde os dados ficam (Hugging Face privado **ou** Google Drive compartilhado) e documentar no README
- [x] 26/09 - **A+B** - Definir **seed global** (ex.: 42) e registrar no README

**Pronto quando:** repo criado, `.gitignore` impedindo commit de CSV, ambiente instalavel via `requirements.txt`.

---

### F1 - Aquisicao e auditoria dos 3 arquivos - 26-28/09

- [x] 26/09 - **A** - Baixar/colar os 3 CSVs (2024, 2025, 2026) no local de dados
- [x] 27/09 - **A** - `download_data.py`: registrar URL de origem, data de download e **SHA-256** de cada arquivo (reprodutibilidade)
- [x] 27/09 - **B** - Auditar **schema de cada ano**: nomes/ordem das colunas, encoding (UTF-8 BOM?), separador (`;`), decimal com virgula (`"135,00"`)
- [x] 27/09 - **B** - Checar **deriva de schema entre anos** (colunas que aparecem/somem, valores novos em `sentido`, `tipo_cobranca`, `tipo_de_veiculo`)
- [x] 28/09 - **A** - Concatenar os 3 anos em um unico DataFrame com coluna `ano_origem` (rastreabilidade)
- [x] 05/10 - **A+B** - Investigar `categoria_eixo` com tipos misturados (DtypeWarning) e definir o tipo antes do Parquet (ver `docs/decisoes_dados.md`, D-006)
- [x] 28/09 - **A** - Converter para **Parquet** (`data/trafego_pedagio.parquet`) - reduz muito o tamanho e acelera o carregamento
- [x] 28/09 - **B** - Checar duplicatas (sem duplicatas entre arquivos; 4.853 chaves repetidas dentro de 2025, somadas na agregacao - ver D-008)

**Pronto quando:** um unico dataset em Parquet, com schema harmonizado, somas conferidas contra o total esperado de cada ano.

---

### F2 - EDA e definicao do problema - 28/09-01/10  (executado em 13/10)

- [x] 13/10 - **A** - Parse de `mes_ano` (`dd/mm/yyyy`, **data diaria**) -> `datetime`; `volume_total` -> float (feito na F1; D-002/D-003)
- [x] 13/10 - **A** - Definir formalmente **unidade de observacao** e a **variavel alvo** (D-009)
- [x] 13/10 - **B** - Estatisticas descritivas: pracas, dias, contagens por dimensao (`src/eda.py`)
- [x] 13/10 - **B** - **Distribuicao de `volume_total`**: skew 14,9, **sem zeros**; histograma em `results/figures/01_hist_volume.png`
- [x] 13/10 - **A** - Frequencia de **ausentes e duplicados**; dias faltantes por praca (D-011, D-008)
- [ ] 13/10 - **B** - **Valores potencialmente anomalos** (PDF Sec.2.1): definir criterio (ex.: IQR ou z-score robusto por praca) para picos/quedas atipicas; decidir manter/winsorizar/marcar/remover - **PENDENTE**
- [x] 13/10 - **A** - Visualizacoes: serie temporal, histograma, dia da semana, heatmap praca x mes (`results/figures/`)
- [x] 13/10 - **B** - **Cobertura temporal**: 2024-01-01 a 2026-08-31 (974 dias); 303 pracas com >= 180 dias (D-010)
- [x] 13/10 - **A+B** - **Fechar decisoes pendentes** e registrar no relatorio (D-009 a D-014)
- [x] 13/10 - **A+B** - Justificar selecao/exclusao de atributos e **checar risco de vazamento** (canais em D-009; teste em `src/dataset.py`)

**Pronto quando:** secao "Descricao do problema e dos dados" do relatorio escrita, com estatisticas, graficos e decisoes justificadas. *(Falta apenas o item de anomalias.)*

---

### F3 - Pre-processamento e janelamento - 01-04/10  (executado em 13/10)

- [x] 13/10 - **A** - Agregar/pivotar para `praca x dia`, canais de volume (D-009)
- [x] 13/10 - **A** - Criar **features de calendario** (dia da semana one-hot, mes seno/cosseno) - consequencia da granularidade diaria (D-007)
- [x] 13/10 - **B** - Codificacao de categoricas: **nao aplicavel** - a unidade agrega sobre sentido/cobranca/categoria; restam os 3 tipos de veiculo como canais (D-009)
- [x] 13/10 - **A** - **Padronizacao fitada SOMENTE no treino** (D-014); parametros devolvidos por `preparar()`
- [x] 13/10 - **B** - **Janelas (sliding windows)**: `X` (window x 13), `y` = Total t+1; dimensoes documentadas (D-013)
- [x] 13/10 - **A** - **Split 1 - cronologico global** + **Split 2 - por grupos (pracas)** (D-013)
- [x] 13/10 - **B** - Pracas com historico curto/serie descontinua: filtro >= 180 dias (D-010) + janela 100% observada (D-011)
- [x] 13/10 - **A** - `src/dataset.py` reprodutivel (mesmas janelas para todos os modelos)
- [ ] 13/10 - **B** - **Validacao cruzada temporal (opcional, PDF Sec.2.3)**: nao executada - declarar como **limitacao** no relatorio
- [x] 13/10 - **A+B** - Testes rapidos: sem vazamento temporal, sem shuffle, scaler so no treino (`testes_rapidos`)

**Pronto quando:** um unico pipeline gera os 3 conjuntos (treino/val/teste) de forma reprodutivel, para os dois protocolos de split (CV temporal opcional nao executada).

---

### F4 - Baselines + arquiteturas iniciais - 04-08/10

- [ ] 04/10 - **B** - **Baselines obrigatorios**: persistencia (y = ultimo valor) e media movel - referencia minima de erro
- [ ] 04/10 - **B** - (Opcional forte) MLP simples sobre janela achatada, como 2o baseline
- [ ] 05/10 - **A** - Implementar as **3 familias**: `SimpleRNN`, `GRU`, `LSTM` - mesmas camadas, mesmos dados, mesmo seed
- [ ] 05/10 - **B** - `src/train.py` com early stopping, `ModelCheckpoint`, `ReduceLROnPlateau`, gradient clipping e curvas de treino/validacao
- [ ] 06/10 - **A** - Rodada-piloto (poucas epocas) para validar o pipeline e medir tempo por epoca
- [ ] 06/10 - **B** - **Baseline de referencia por familia** (hiperparametros "razoaveis" fixos) para calibrar depois com o Optuna
- [ ] 07/10 - **A** - Metricas: **MAE, RMSE, R2** (e discutir erro na unidade: veiculos); implementar avaliacao em val
- [ ] 08/10 - **A+B** - Curvas de treino/val: identificar subajuste/overfitting iniciais

**Pronto quando:** as 3 familias treinam ponta a ponta, com metricas em val e curvas salvas em `results/`.

---

### F5 - Hiperparametrizacao com Optuna - 08-11/10

- [ ] 08/10 - **A** - Configurar o **estudo**: metrica = RMSE (ou MAE) **na validacao**, `seed` fixo, pruner (`MedianPruner`/`Hyperband`)
- [ ] 08/10 - **B** - Definir **espaco de busca** e justifica-lo no relatorio:
  - [ ] `model_type` = {SimpleRNN, GRU, LSTM} (categorical -> comparacao sob condicoes equivalentes)
  - [ ] `window_size` em dias (ex.: {7, 14, 30, 90})
  - [ ] `units` por camada em 32-256 (log)
  - [ ] `n_layers` em 1-3
  - [ ] `dropout` em 0-0,5 - `recurrent_dropout` em 0-0,3
  - [ ] `learning_rate` em 1e-4-1e-2 (log)
  - [ ] `batch_size` em {32, 64, 128, 256}
  - [ ] otimizador {Adam, RMSprop} - `clipnorm`
  - [ ] paciencia do early stopping
- [ ] 09/10 - **A** - **Storage do Optuna**: `RDBStorage` em Postgres gratis (Neon/Supabase) **ou** SQLite - persistir para nao perder a busca
- [ ] 09/10 - **A** - Rodar a busca (**40-80 trials**, conforme orcamento de GPU) e **salvar o estudo** (`study.db` / export em `results/`)
- [ ] 10/10 - **B** - Subir **Optuna Dashboard** para visualizar
- [ ] 10/10 - **B** - Extrair: melhores tentativas, **importancia dos hiperparametros**, grafico de coordenadas paralelas, efeito no tempo de treino e no overfitting
- [ ] 11/10 - **A+B** - Selecionar e justificar a **configuracao final** e as **3 variacoes promissoras** por familia

**Pronto quando:** `results/` contem o espaco de busca, no de trials, metrica otimizada, melhores tentativas e a justificativa da config final (entregavel explicito do PDF).

---

### F6 - Experimentos exigidos pela frente 3.2 - 11-13/10

- [ ] 11/10 - **A** - **Comparacao SimpleRNN vs GRU vs LSTM** com a melhor config, sob condicoes identicas (dados, janela, seed, epocas) - tabela + grafico
- [ ] 12/10 - **A** - **Profundidade e largura**: camadas recorrentes (1-3) x unidades (32-256) -> desempenho vs **no de parametros** vs tempo
- [ ] 12/10 - **B** - **Tamanhos de janela**: em dias (ex.: 7, 14, 30, 90) -> qual melhora a previsao? efeito em custo, estabilidade e overfitting
- [ ] 13/10 - **B** - **Dependencias de longo prazo**: janelas maiores ajudam? discutir com evidencia
- [ ] 13/10 - **A** - **Estrategia de previsao**: relatar 1 passo vs multi-step (se implementado) e as **dimensoes de entrada/saida**
- [ ] 13/10 - **A+B** - **Custo**: tempo de treino, tempo de inferencia (ms/predicao) e no de parametros por arquitetura

**Pronto quando:** existem tabelas e graficos cobrindo **todos** os itens da frente 3.2 (o PDF exige cada um).

---

### F7 - Avaliacao final no teste - 13-14/10

- [ ] 13/10 - **A** - Treinar a config final e avaliar **uma unica vez** no **conjunto de teste** (nao ajustar nada depois)
- [ ] 14/10 - **A** - Relatar MAE, RMSE e R2 explicando o significado na **unidade da variavel alvo** (veiculos)
- [ ] 14/10 - **B** - Reportar os **dois protocolos de split** (cronologico e por grupos/pracas) e discutir diferenca
- [ ] 14/10 - **B** - Analise de **erros por praca/concessionaria** (onde o modelo falha?) e limitacoes/incertezas
- [ ] 14/10 - **A+B** - Comparar contra os baselines de persistencia/media movel

**Pronto quando:** secao "Avaliacao final no teste e custo vs efetividade" escrita com tabela comparativa.

---

### F8 - Relatorio, slides e revisao final - 13-15/10

- [ ] 13/10 - **B** - Montar relatorio na estrutura do PDF: (1) Introducao/objetivo, (2) Fonte e caracterizacao, (3) Arquiteturas e espaco de busca, (4) Validacao e selecao, (5) Teste e custo, (6) Limitacoes e conclusoes
- [ ] 14/10 - **A** - Montar **slides (15 min)**: problema, dados, metodo, resultados, custo, conclusoes
- [ ] 14/10 - **B** - Ensaiar a apresentacao e cronometrar (15 min duros)
- [ ] 15/10 - **A+B** - Revisao final: conferir entrega (GitHub, estudo do Optuna, tabelas/graficos) e submeter
- [ ] 15/10 - **A+B** - Garantir que o **link do GitHub esta publico/liberado** para o professor

**Pronto quando:** tudo submetido antes de 15/10, com link do GitHub testado em navegador anonimo.

---

## 3. Cronograma resumido

| Periodo | Fase | Marco |
|---|---|---|
| 25-26/09 | F0 Setup | Repo + ambiente prontos |
| 26-28/09 | F1 Aquisicao/auditoria | 3 arquivos unificados em Parquet |
| 28/09-01/10 | F2 EDA/alvo | Problema, alvo e anomalias definidos |
| 01-04/10 | F3 Pre-proc/janelas | Pipeline reprodutivel + 2 splits (+ CV opcional) |
| 04-08/10 | F4 Baselines/arquiteturas | 3 familias treinando |
| 08-11/10 | F5 Optuna | Estudo salvo + config final |
| 11-13/10 | F6 Experimentos 3.2 | Tabelas/graficos completos |
| 13-14/10 | F7 Teste | Resultado final no teste |
| 13-15/10 | F8 Relatorio/slides | Entrega submetida |

## 4. Riscos e armadilhas (checar antes de cada fase)

- [ ] **Vazamento de dados:** scaler/encoder fitados so no treino; nenhum canal pode conter o alvo do dia t+1
- [ ] **Serie curta (~33 meses):** validar com janelas de tamanhos diferentes; majoritariamente modelo global (pooling de pracas)
- [ ] **Sem shuffle** em nenhum passo da serie temporal
- [ ] **Zero-inflacao** e pracas com historico curto: decisao explicita e justificada
- [ ] **Valores anomalos:** criterio explicito (IQR/z-score por praca) e decisao registrada (manter/winsorizar/marcar/remover)
- [ ] **Comparacoes justas:** mesmo seed, mesmos dados, mesmo no de epocas quando comparar familias
- [ ] **Reprodutibilidade:** fixar seeds e versionar o estudo do Optuna
- [ ] **Se um ano tiver schema divergente**, documentar e tratar antes de concatenar

## 5. Mapa rubrica -> evidencia

| Criterio (peso) | O que gera a nota | Onde fica |
|---|---|---|
| I. Justificativas (3 pts) | Decisoes de alvo, split, features, janela, espaco de busca | F2, F3, F5, F6 e relatorio |
| II. Corretude da modelagem (2 pts) | SimpleRNN/GRU/LSTM corretos, dimensoes de E/S, baselines | F4, F6 |
| III. Hiperparametrizacao (2 pts) | Estudo Optuna salvo, espaco justificado, metrica em val | F5 |
| IV. Interpretacao (2 pts) | Curvas, overfitting, custo vs desempenho, limitacoes | F6, F7, F8 |

## 6. Cobertura dos requisitos do PDF

Mapa explicito de **cada exigencia do enunciado -> onde e atendida**, para nao perder ponto.

### Requisitos comuns (Sec.2)

| Requisito do PDF | Exigencia | Onde e atendido |
|---|---|---|
| Sec.2.1 Problema e dados | Entradas/alvo com tipos, unidades e limitacoes; estatisticas + visualizacoes; ausentes/duplicados/**anomalos**; distribuicao do alvo; justificativa de features e **risco de vazamento** | **F2** |
| Sec.2.2 Pre-processamento | Padronizacao/normalizacao, one-hot/codificacao justificada, ausentes, formato de entrada; **parametros so no treino** | **F3** |
| Sec.2.3 Split | Treino/val/teste com finalidades distintas; **cronologico sem shuffle**; **por grupos**; **k-fold opcional** | **F3** |
| Sec.2.4 Hiperparametrizacao | **Optuna**; justificar HPs, intervalos, no de trials e **metrica so na validacao**; efeito em metrica/tempo/overfitting | **F5** |
| Sec.2.5 Avaliacao | Metrica principal + complementares; em regressao **MAE/RMSE/R2** na unidade do alvo; **curvas treino/val**; **custo de treino e de inferencia**; limitacoes/incerteza/falhas | **F4, F6, F7** |

> **Definicao do problema (Sec.1):** tarefa de **regressao**; unidade de observacao e significado da variavel alvo declarados em **F2** (ver secao 1).

### Frente 3.2 - Redes Recorrentes

| Exigencia do PDF | Onde e atendido |
|---|---|
| Serie temporal **multivariada**; entradas multiplas -> >=1 alvo; **ordem temporal preservada** | F2, F3 |
| Comparar **SimpleRNN vs GRU vs LSTM** em condicoes equivalentes | **F6** |
| No de **camadas recorrentes** x **unidades** (capacidade x custo) | **F6** |
| Diferentes **tamanhos de janela** + **horizonte de previsao** | **F6** |
| **Dependencias de longo prazo** (janelas maiores: custo, estabilidade, overfitting) | **F6** |
| **Estrategia de previsao** (1 passo vs multi-step) e **dimensoes de E/S** | **F6** |
| Comparar **tempo de treino** e efetividade das 3 familias | **F6** |

### Entregaveis (Sec.4) e relatorio (Sec.5)

| Entregavel | Onde |
|---|---|
| Relatorio tecnico (integrantes, dados, problema, metodo, experimentos, resultados, discussao, conclusoes) | **F8** (estrutura de 6 secoes) |
| Link do GitHub com o codigo | **F0** + revisao em **F8** |
| **Registro da busca de hiperparametros** (espaco, config do estudo, melhores tentativas, justificativa) | **F5** |
| Tabelas e graficos comparativos das arquiteturas/estrategias | **F6** (e **F7**) |

---

*Documento vivo: marque os checkboxes conforme avancar e registre desvios no README do repositorio.*
