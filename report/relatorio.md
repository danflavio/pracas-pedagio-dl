<p align="center"><img src="assets/logo-ifg.png" alt="IFG" height="110"></p>

# Previsão de volume de tráfego em praças de pedágio

**Disciplina:** Deep Learning · **Frente:** 3.2 (redes recorrentes)
**Alunos:** Daniel Flavio e Guilherme Martins · **Data:** outubro de 2026

---

## 1. O que queremos fazer

Prever **quantos veículos** vão passar em uma praça de pedágio **no dia seguinte**.

É um problema de série temporal: o que aconteceu nos dias anteriores ajuda a estimar o
próximo dia. Não é classificar nem agrupar — é prever um número. Por isso a tarefa é de
**regressão**.

Queremos também comparar três tipos de rede recorrente — **SimpleRNN, GRU e LSTM** — para
ver qual funciona melhor e quanto cada uma custa.

## 2. Os dados

Usamos os dados abertos da **ANTT** (volume de tráfego nas praças de pedágio), disponíveis no
Portal Brasileiro de Dados Abertos. Juntamos três arquivos anuais em uma série única.

| Item | Número |
|---|---|
| Período | 2024-01-01 a 2026-08-31 (974 dias) |
| Linhas no arquivo original | 9.184.154 |
| Praças (concessionária + praça) | 345 |
| Concessionárias | 43 |

Dois detalhes dos dados mudaram o rumo do trabalho:

1. **A série é diária.** A coluna chamada `mes_ano`, apesar do nome, guardava **uma data por
   dia**. Isso foi uma descoberta importante: se tivéssemos tratado como mensal, perderíamos o
   efeito de dia da semana.
2. **A série é descontínua.** Apenas 47 praças têm todos os 974 dias. Metade tem menos de um
   ano de histórico, porque praças novas entraram no meio do período.

A consequência do item 2 é direta: **não dá para treinar um modelo por praça**. Muitas praças
teriam pouquíssimos dias. A solução foi treinar **um único modelo, com todas as praças juntas**
(modelo global).

## 3. Como o modelo funciona

Cada praça vira uma sequência de dias. Para cada dia, o modelo recebe **13 sinais**:

- **4 de volume:** carros de passeio, comerciais, motos e o total;
- **9 de calendário:** dia da semana e época do ano.

O modelo olha uma **janela de N dias** e responde com **um número**: o total do dia seguinte.

| Etapa | Formato |
|---|---|
| Entrada (X) | janela de N dias × 13 sinais |
| Saída (y) | total de veículos do dia seguinte |

As três arquiteturas usam exatamente a mesma entrada e a mesma saída. **Só muda a célula da
rede.** Isso é o que torna a comparação justa.

## 4. Como avaliamos

Avaliar série temporal tem regras, e respeitá-las é o que dá credibilidade ao resultado:

- **Treino, validação e teste ficam em ordem de data.** O teste é o futuro — ninguém viu.
- **Nada é embaralhado.** A ordem dos dias nunca é quebrada.
- **A normalização usa só o treino.** Se usássemos o futuro para padronizar, o resultado mentiria.
- **Janela só vale se todos os dias existem.** Não preenchemos buraco de série com dado inventado.

Também treinamos dois modelos simples para servir de **régua**:

| Modelo simples (baseline) | MAE | RMSE | R² |
|---|---|---|---|
| Repetir o último dia (persistência) | 1.470 | 3.101 | 0,94 |
| Média dos últimos dias | 1.737 | 3.946 | 0,90 |

**Atenção a este ponto:** o baseline mais simples já tem R² de 0,94. Ou seja, um R² alto aqui
não prova nada sozinho. Um modelo só tem valor se **bater essas duas réguas**.

## 5. Resultado

**[PENDENTE — tabela final no conjunto de teste e comparação entre as famílias]**

## 6. Limitações e conclusão

**Limitações (declaradas de propósito, não escondidas):**

- **Sem GPU.** O TensorFlow no Windows não usa placa de vídeo, então tudo foi treinado na CPU.
  Isso obrigou a reduzir o tempo de busca de parâmetros.
- **Sem validação cruzada** (era opcional). Não medimos a variância dos resultados.
- **Só prevemos 1 dia à frente.** Não testamos t+2 nem t+3.
- **Buraco de série é descartado**, então usamos um pouco menos de dados do que existe.

**Conclusão:** [PENDENTE]

---

## Como rodar o projeto

```bash
python download_data.py          # baixa os dados e registra a "impressão digital" (SHA-256)
python src/preprocess.py         # junta os 3 anos em um único arquivo
python src/eda.py                # gráficos e estatísticas da análise
python src/dataset.py            # monta as janelas e os conjuntos de treino/validação/teste
python src/train.py              # treina as três redes + baselines
python src/optuna_search.py      # busca os melhores parâmetros
python src/experimentos.py       # experimentos da frente 3.2
python src/avaliar_teste.py      # avaliação final no teste
```

Todas as decisões sobre os dados estão registradas em [`../docs/decisoes_dados.md`](../docs/decisoes_dados.md).
