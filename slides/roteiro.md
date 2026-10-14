# Roteiro da apresentação — 15 minutos

**Deck:** `slides/apresentacao_v2.pdf` (14 slides, estilo hand-drawn) · **Dupla:** Daniel Flavio e Guilherme Martins
**Regra de ouro:** o slide é apoio, **você** fala. Nunca leia o slide em voz alta.

Tempo-alvo: **~13:55**, com ~1 min de folga dentro do limite de 15. Melhor sobrar do que estourar.

---

## Divisão sugerida

| Quem | Slides | Assunto | Tempo |
|---|---|---|---|
| Daniel | 1 a 7 | Problema, dados, método e a régua | ~7:00 |
| Guilherme | 8 a 14 | Arquiteturas, resultados e conclusão | ~7:00 |

(O combinado pode ser outro — só decidam **antes** do ensaio.)

---

## Slide a slide

### 1 · Capa — 0:20
- Cumprimentar, dizer quem são e o tema em uma frase.
- "Vamos mostrar como prevemos o volume de veículos de uma praça de pedágio para o dia seguinte, comparando três redes recorrentes."

### 2 · O problema — 0:50
- O que queremos: **quantos veículos** passam amanhã naquela praça, olhando os dias anteriores.
- É **regressão** — prever um número, não uma categoria.
- Para que serve: dimensionar cabines e equipes, estimar arrecadação, antecipar feriados.

### 3 · Os dados — 0:55
- Dados abertos da **ANTT**, de 2024 a 2026. **9,2 milhões** de linhas, **345** praças, **974** dias.
- Aponte o gráfico: **sexta é o dia mais cheio; sábado e domingo, os mais vazios.**
- "Esse padrão semanal é justamente o que a gente não queria perder."

### 4 · Dois achados que mudaram o plano — 1:15
- **Achado 1:** a coluna chamada `mes_ano` era, na verdade, **diária**. Se tivéssemos tratado como mensal, perderíamos o efeito do dia da semana.
- **Achado 2:** a série é **descontínua** — só 47 praças têm o período inteiro; metade tem menos de um ano.
- Consequência: **não dá para treinar um modelo por praça.** Treinamos **um modelo único, com todas as praças**.
- Mencione rapidamente: marcamos dias fora do padrão (2% deles) e decidimos **manter**, porque são eventos reais (Reveillon numa rodovia do litoral).
- **Pausa:** "Foram essas decisões sobre os dados que definiram todo o resto."

### 5 · O que o modelo enxerga — 0:55
- **13 sinais por dia:** 4 de volume (passeio, comercial, moto, total) + 9 de calendário.
- Uma **janela de N dias** entra; a saída é **um número**: o total do dia seguinte.
- Enfatize: as três arquiteturas recebem **exatamente** a mesma entrada — só muda a célula.

### 6 · As regras da avaliação — 1:05
- Treino, validação e teste em **ordem de data** — o teste é o futuro, ninguém viu.
- **Nada é embaralhado.** Normalização ajustada **só no treino**.
- Janela só vale se **todos os dias existem** — não inventamos dado em buraco de série.
- Frase de efeito: "É aqui que a maioria dos trabalhos escorrega — usar o futuro para treinar."

### 7 · A régua — 1:05
- A previsão mais simples é dizer que **amanhã = hoje** (persistência). Ela já dá **MAE 1.470** e **R² 0,94**.
- **Ponto-chave:** um R² alto aqui **não prova nada**. Um modelo só vale se **bater a régua**.
- Este é o slide que mostra maturidade — não pule.

### 8 · As três arquiteturas — 0:45
- SimpleRNN: memória curta e simples. GRU: portões que decidem o que lembrar. LSTM: memória de longo prazo, mais pesada.
- Uma frase só para cada. Não entre em fórmula.

### 9 · O ajuste automático — 1:15
- Usamos **Optuna**, com a métrica (**RMSE**) medida **na validação**. O teste fica intocado até o fim.
- **Ressalva honesta:** sem placa de vídeo, a busca rodou reduzida (15 tentativas, parte do treino).
- **Achado importante:** a configuração vencedora da busca, retreinada com os dados completos, rendeu **pior** que a de referência. Por isso o **modelo final usa a de referência**.
- "Não é fracasso — é o limite do hardware aparecendo em número, e não só na desculpa."

### 10 · Quem ganha: SimpleRNN, GRU ou LSTM? — 1:20
- GRU **1.537**, LSTM 1.550, SimpleRNN 1.566 → **empate técnico**.
- Destaque: o SimpleRNN chega perto gastando **um terço dos parâmetros**.
- **Diga a ressalva do slide:** esses números vêm de uma **receita controlada** (a da busca), usada só para isolar a arquitetura. **O modelo final usa outra receita** — e rende melhor (próximo slide).

### 11 · Quanto passado olhar e qual tamanho — 1:05
- **Janela de 30 dias** foi melhor que 14 e 7.
- **Duas camadas** batem uma — mas a **terceira não acrescenta**.

### 12 · Desempenho tem preço — 0:55
- 32 → 128 unidades: erro cai, mas são **12 vezes mais parâmetros para cerca de 12%** de ganho.
- Retorno **decrescente**. E o tempo de inferência é baixo em todos (~0,1 ms por previsão).

### 13 · O número que vale: o teste — 1:25
- **Dois cenários.** Primeiro: continuar a série da **mesma praça**. Segundo: prever em **praças novas**.
- Resultado: erro **27% menor** que a régua no primeiro e **17% menor** no segundo.
- Dê escala: o erro de 1.126 veículos equivale a cerca de **8% do movimento diário**.
- Explique a queda: prever praça **nunca vista** é mais difícil. Isso é esperado — e honesto.

### 14 · Fechamento — 0:45
- O achado central: **as três famílias empataram**. O ganho veio das **decisões sobre os dados** (janela de 30 dias), não da escolha da célula.
- Limitações: sem GPU, sem validação cruzada, só 1 dia à frente.
- Próximo passo: GPU e previsão de vários dias.
- "Obrigado. Ficamos à disposição para perguntas."

---

## Perguntas prováveis (e respostas curtas)

1. **"Por que a busca não achou a melhor configuração?"**
   Porque teve orçamento pequeno: sem GPU, rodou em parte do treino e com poucas tentativas. A escolha final foi guiada pela validação — e a configuração de referência teve erro melhor (1.108 contra 1.493 ou mais).

2. **"R² de 0,95 é bom?"**
   Com cuidado: a persistência já dá 0,92–0,94. O número que importa é o **erro médio 27% menor** que a régua — não o R² sozinho.

3. **"Por que não um modelo por praça?"**
   Porque a série é descontínua: metade das praças tem menos de um ano de histórico. Um único modelo com todas as praças aproveita muito mais dados.

4. **"Como vocês evitaram vazamento de dados?"**
   Split por data (sem embaralhar), normalização ajustada só no treino e janela só aceita quando todos os dias existem. Há um teste automático no `dataset.py` que verifica isso.

5. **"Por que 13 sinais?"**
   4 de volume (passeio, comercial, moto e total) e 9 de calendário (dia da semana e época do ano).

6. **"Qual arquitetura vocês usariam na prática?"**
   Se custo importa: **SimpleRNN** (um terço dos parâmetros, quase o mesmo erro). Se o erro é crítico: **GRU**.

---

## Cinco números que você precisa saber de cor

| Número | Significa |
|---|---|
| **13** | sinais por dia (4 de volume + 9 de calendário) |
| **30 dias** | tamanho de janela que funcionou melhor |
| **1.470** | erro da régua (persistência) — o que temos que bater |
| **1.126** | erro do modelo final (mesma praça) |
| **27%** | quanto o modelo reduziu o erro em relação à régua |

---

## Ensaio

1. Rode o PDF em **tela cheia** e vá no relógio. Se passar de 14:30, corte.
2. Onde costuma estourar: **slides 4, 9 e 13**. Se precisar encurtar, corte daqui.
3. Não leia os slides. Se estiver lendo, é sinal de que não ensaiou o suficiente.
4. Ensaiem **a passagem de um para o outro** (slide 7 → 8): combinem a frase exata da troca.
5. Tenham o **`pipeline.pdf`** à mão como material de apoio — se alguém pedir detalhe de uma fase, ele responde sem gastar tempo da fala.
