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
| Praças (concessionária + praça) | 345 no arquivo; **303** com histórico ≥ 180 dias entram no modelo (D-010) |
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

**Um terceiro cuidado com os dados.** Marcamos os dias fora do padrão de cada praça (critério
IQR, com limite de 3 vezes o IQR). Eles são apenas **2% dos dias** e **2,6% do volume** — e os
maiores casos são claramente eventos reais, como o **Reveillon numa rodovia do litoral**.
Decidimos **manter** esses dias: apagá-los seria esconder justamente o que é mais difícil prever.

## 3. Como o modelo funciona

Cada praça vira uma sequência de dias. Para cada dia, o modelo recebe **13 sinais**:

- **4 de volume:** carros de passeio, comerciais, motos e o total;
- **9 de calendário:** dia da semana e época do ano.

O modelo olha uma **janela de N dias** e responde com **um número**: o total do dia seguinte. Todos
os canais de volume estão em **veículos por dia**, e o alvo na mesma unidade.

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

### 5.1 Comparando as três arquiteturas

Para isolar o efeito da célula recorrente, fixamos a **mesma receita** para as três e mudamos
**somente a célula** (mesmos dados, mesma janela, mesma semente). A receita usada é a **vencedora da
busca** (unidades = 54), o que faz desta uma comparação controlada:

| Arquitetura | MAE | RMSE | Parâmetros | Tempo de treino |
|---|---|---|---|---|
| **GRU** | **1.537** | **2.702** | 11.233 | 9 min |
| LSTM | 1.550 | 2.747 | 14.743 | 7 min |
| SimpleRNN | 1.566 | 2.725 | 3.727 | 3 min |

As três ficam **muito próximas**. O SimpleRNN chega perto gastando **um terço dos parâmetros** —
o que, na prática, é um resultado relevante: mais complexidade não trouxe ganho proporcional.

### 5.2 Quantos dias olhar e qual tamanho usar

| Janela | RMSE | | Camadas | MAE |
|---|---|---|---|---|
| **30 dias** | **2.725** | | **2** | **1.493** |
| 14 dias | 3.020 | | 3 | 1.503 |
| 7 dias | 3.061 | | 1 | 1.566 |

| Unidades | Parâmetros | MAE | Treino | Por previsão |
|---|---|---|---|---|
| 32 | 1.505 | 1.713 | 1 min | 0,07 ms |
| 64 | 5.057 | 1.598 | 4 min | 0,10 ms |
| **128** | 18.305 | **1.514** | 7 min | 0,12 ms |

**Dentro desta receita controlada**, janelas mais longas ajudam, duas camadas rendem mais que uma (a
terceira não acrescenta) e mais unidades reduzem o erro — mas com custo crescente: 12 vezes mais
parâmetros para cerca de 12% de MAE. São **tendências relativas dentro da receita da busca**; não
são os números do modelo final. A receita de referência (1 camada, 64 unidades) teve MAE **melhor
que todas** estas variantes (ver 5.4).

### 5.3 Uma observação importante sobre a busca

Os números acima saíram da configuração vencedora da busca. Quando essa configuração foi
retreinada com os dados completos, rendeu MAE a partir de **1.493** — **pior** que a
configuração de referência do piloto (**1.108**). Isso mostra, com evidência, que **o orçamento
da busca foi pequeno demais** (sem GPU). Por isso o modelo final usa a configuração de
referência, e não a da busca.

**Resumindo a diferença:** as tabelas 5.1 e 5.2 vêm da **receita da busca** (unidades 54, rmsprop);
a avaliação final (5.4) usa a **receita de referência** (unidades 64, adam). Por isso os valores
**não são diretamente comparáveis** entre si — cada bloco responde a uma pergunta diferente
(comparar famílias vs. medir o desempenho final).

### 5.4 Avaliação final no teste

Depois de escolher a configuração (**GRU, janela 30, 1 camada, 64 unidades, Adam**), o teste foi
usado **uma única vez**, sem ajustar nada depois.

**Cenário 1 — continuar a série da mesma praça** (divisão por data):

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Repetir o último dia | 1.536 | 3.303 | 0,92 |
| Média dos últimos dias | 1.831 | 4.099 | 0,88 |
| **GRU (modelo final)** | **1.126** | **2.773** | **0,95** |

**Cenário 2 — prever em praças novas** (divisão por praça):

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Repetir o último dia | 1.774 | 3.699 | 0,92 |
| Média dos últimos dias | 2.028 | 4.423 | 0,89 |
| **GRU (modelo final)** | **1.469** | **3.276** | **0,94** |

O modelo bate a régua **nos dois cenários**: erro cerca de **27% menor** no primeiro e **17%
menor** no segundo. Para dar escala: o volume médio de uma praça é de cerca de 13 mil veículos
por dia, então um erro de 1.126 equivale a aproximadamente **8%** do movimento diário.

A vantagem cai em praças novas — o que faz sentido: prever uma praça que o modelo **nunca viu** é
mais difícil do que continuar a série de uma praça conhecida.

### 5.5 O que as curvas mostram

As curvas de treino/validação (`results/figures/curva_*.png`) caem **juntas** nas primeiras épocas
e depois a **validação estabiliza**, enquanto o treino continua caindo. Surge então um **gap
crescente** entre as duas (treino bem abaixo da validação): é **overfitting leve**, não subajuste.
Quem contém isso é o **early stopping** com `restore_best_weights`, que guarda o ponto de menor
erro de validação e descarta as épocas seguintes. Não foi preciso regularização mais forte
(dropout) nessa receita.

## 6. Limitações e conclusão

**Limitações (declaradas de propósito, não escondidas):**

- **Sem GPU.** O TensorFlow no Windows não usa placa de vídeo, então tudo foi treinado na CPU.
  Isso obrigou a reduzir o tempo de busca de parâmetros.
- **Sem validação cruzada** (era opcional). Não medimos a variância dos resultados.
- **Só prevemos 1 dia à frente.** Não testamos t+2 nem t+3.
- **Buraco de série é descartado**, então usamos um pouco menos de dados do que existe.
- **Dias atípicos** (feriados, eventos) são onde o erro tende a ser maior: o modelo aprende o padrão
  semanal, mas não "sabe" que um dia é feriado.

**Conclusão:** as redes recorrentes funcionaram. Reduziram o erro em cerca de **27%** ao continuar a
série de uma praça e **17%** em praças novas, sempre batendo a régua mais forte.

O achado mais interessante, porém, **não foi a arquitetura vencedora**. As três famílias
empataram na prática, e o SimpleRNN entregou quase o mesmo resultado com **um terço dos
parâmetros**. O ganho mais consistente veio de decisões sobre os dados — sobretudo a **janela de 30
dias** — e não da escolha da célula.

Uma última observação, que só aparece porque testamos: a configuração vencedora da busca de
parâmetros rendeu **pior** que a configuração de referência quando retreinada com os dados
completos. Isso expõe o limite do hardware disponível (sem placa de vídeo) de forma concreta, em
vez de só declarada.

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
