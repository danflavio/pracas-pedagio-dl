# Registro de Decisoes - Dados

Documento vivo. Registra as decisoes tomadas na auditoria e no preparo dos dados,
com a evidencia que sustenta cada uma. Serve de base para a secao de justificativas
do relatorio tecnico (F8).

Convencoes:
- ASCII puro (sem acentos), igual ao restante da documentacao tecnica do projeto.
- Cada decisao tem um id (D-00N) e nao deve ser reescrita. Se a decisao mudar,
  adicione uma nova entrada e marque a antiga como "substituida por D-00M".

---

## D-001 - Encoding dos CSVs: Latin-1 (nao UTF-8)

- Data: 2026-10-05
- Decisao: ler os 3 CSVs com `encoding="latin-1"`.
- Evidencia: a leitura como UTF-8 falha (bytes `c3 4f` em 2024/2025 e `e1 74` em 2026);
  nao ha BOM (o primeiro byte e `63` = "c"); os acentos ocupam 1 byte (ex.: `e1` = "a" com acento).
- Alternativas: `utf-8` (falha na decodificacao); `utf-8-sig` (nao ha BOM, nao se aplica).
- Impacto: sem isso, nomes de concessionaria/praca/categoria quebram ou viram mojibake.

## D-002 - Separador e formato numerico

- Data: 2026-10-05
- Decisao: ler com `sep=";"` e `decimal=","`; sem `thousands`.
- Evidencia: cabecalho e linhas usam `;`; valores como `13,00`; varredura completa nao
  encontrou separador de milhar.
- Alternativas: usar `,` como separador de campos (incorreto: quebraria o decimal).
- Impacto: sem `decimal=","`, `volume_total` seria lido como texto (`str`) em vez de `float64`.

## D-003 - `mes_ano` chega como texto

- Data: 2026-10-05
- Decisao: converter `mes_ano` (formato `dd/mm/yyyy`, **data diaria**) para data na F2 (ver D-007).
- Evidencia: o pandas infere `str`.
- Alternativas: manter como texto (impede ordenacao temporal e janelamento).
- Impacto: sem a conversao, nao ha como montar a serie temporal nem as janelas deslizantes.

## D-004 - Categoria "Veiculo Pequeno" existe apenas em 2024

- Data: 2026-10-05
- Decisao: excluir a categoria "Veiculo Pequeno" de `tipo_de_veiculo` na etapa de pre-processamento.
- Evidencia: presente somente em 2024; participacao de ~8.4e-09 (~0,0000008%) do volume de 2024.
- Alternativas: manter (categoria inconsistente entre os anos); remapear para outra categoria
  (perde o significado original do dado).
- Impacto: mantem o espaco de categorias consistente entre os anos; o efeito no alvo (soma) e
  desprezivel.

## D-005 - Schema estavel entre os anos

- Data: 2026-10-05
- Decisao: unificar os 3 anos sem remapeamento de colunas.
- Evidencia: mesmas 8 colunas e mesma ordem nos 3 anos (amostra de 100 mil linhas; valores
  categoricos de sentido/tipo_cobranca/tipo_de_veiculo conferidos no arquivo inteiro). ATENCAO:
  os TIPOS nao sao uniformes - ver D-006 (categoria_eixo com tipos misturados).
- Alternativas: tratar colunas ausentes ou renomeadas (nao ha).
- Impacto: `pd.concat` pode ser usado com seguranca para as colunas.

---

## D-006 - `categoria_eixo` com representacao mista (codigo vs rotulo)

- Data: 2026-10-06
- Decisao (F1, imediata): ler `categoria_eixo` como `str` na unificacao e no Parquet, para manter
  o tipo consistente entre os anos e eliminar o `DtypeWarning`.
- Evidencia: 2024 e 2026 trazem apenas codigo numerico (ex.: `2`, `10`); em 2025, 15.783 linhas
  (0,48% do ano) usam rotulo por extenso (ex.: `Veiculo Comercial 2 eixos`, `Moto`), concentradas
  em jan-mar/2025. A amostra de 100 mil linhas de 2025 havia inferido `int64`; o texto aparece
  fora da amostra.
- Alternativas: `to_numeric` direto (converte o texto em `NaN` - nao padroniza); usar a coluna como
  feature (exigiria mapear codigo <-> rotulo).
- Impacto: como a unidade de observacao agrega sobre `categoria_eixo`, o alvo nao e afetado.
- Pendente (F3): decidir o destino da coluna - (a) descartar, (b) usar como feature, ou
  (c) padronizar para numerico. A opcao (c) exige definir o mapeamento codigo <-> rotulo
  (atencao a inconsistencias do texto de 2025: `7 Eixos` vs `7 eixos`; `Acima de 10 Eixos`).

## D-007 - Granularidade diaria (nao mensal)

- Data: 2026-10-06
- Decisao: modelar na granularidade **diaria**; alvo = volume da praca no **dia t+1**.
- Evidencia: `mes_ano` tem 366 (2024), 365 (2025) e 243 (2026) datas distintas, com todos os dias
  01-31 - ou seja, e uma data diaria apesar do nome. O `head()` enganava por o arquivo estar
  ordenado por data.
- Alternativas: agregar para mensal (perde padroes de dia da semana/feriado e reduz de ~975 para
  ~33 pontos por praca, prejudicando o treino das RNNs).
- Impacto: alvo passa a ser t+1 em dias; janelas redefinidas em dias; exige features de calendario
  (dia da semana, mes, feriado). Atualiza PLANO (F2/F5/F6) e README.

## D-008 - Duplicatas com chave repetida (dentro de 2025)

- Data: 2026-10-06
- Decisao: manter as linhas e **somar o `volume_total`** das chaves repetidas na agregacao por
  praca/dia (a unidade final ja agrega somando).
- Evidencia: nao ha duplicata entre arquivos (cada arquivo traz apenas datas do proprio ano:
  2024 -> 2024, 2025 -> 2025, 2026 -> 2026). Ha 4.853 chaves repetidas dentro de 2025 (0,15%),
  todas em jan-mar/2025; 59 sao linhas exatas e as demais tem mesma chave com volume diferente.
  2024 e 2026 nao tem duplicatas.
- Alternativas: remover as linhas repetidas (perderia volume, sem base para escolher qual manter);
  manter cru sem decidir (infla o total se agregado de outro jeito).
- Impacto: como a unidade final agrega (soma) por praca/dia, somar os duplicados e o comportamento
  natural. Limite: assume-se que as linhas repetidas sao medidas distintas, nao erro de exportacao
  (concentracao em Q1/2025, junto do drift da D-006).

---

## Contagem de linhas

- 2024 = 3.312.763 | 2025 = 3.274.157 | 2026 = 2.597.234 (total 9.184.154).

---

## D-009 - Unidade de observacao, canais e alvo (F2/F3)

- Data: 2026-10-13
- Decisao: unidade = **(concessionaria, praca, dia)**. Canais de entrada = volume diario por
  `tipo_de_veiculo` {Passeio, Comercial, Moto} + Total, mais features de calendario (dia da
  semana one-hot, mes em seno/cosseno). Alvo = **Total da praca no dia t+1**.
- Evidencia: 345 pares `concessionaria|praca` e 974 dias (2024-01-01 a 2026-08-31). O Total
  e a soma dos tipos e confere com o bruto (2.379.466.795 veiculos). `volume_total` tem
  skew 14,9 (cauda longa), sem zeros.
- Alternativas: alvo mensal (descartado em D-007); usar so os componentes sem o Total (perde
  a informacao de nivel agregado, que e o preditor mais forte).
- Impacto: define a forma do modelo: X = (window x 13 canais), y = 1 valor.

## D-010 - Filtro de pracas: minimo de 180 dias observados

- Data: 2026-10-13
- Decisao: manter pracas com **>= 180 dias** observados -> **303 de 345**.
- Evidencia: cobertura por praca - 47 com 974 dias, 182 com >= 365, 303 com >= 180, 334 com
  >= 90; mediana de 432 dias. Series muito curtas nao formam janelas suficientes e
  prejudicam a diversidade do split por grupos.
- Alternativas: >= 365 dias (182 pracas - menos diversidade); manter todas (pracas com 23
  dias nao geram amostra util).
- Impacto: serie final com 180.737 linhas (179.124 observadas, 1.613 faltantes).

## D-011 - Buraco no meio da serie: so aceitar janela 100% observada

- Data: 2026-10-13
- Decisao: uma janela (entrada + alvo) so e valida se **todos** os dias estao observados.
- Evidencia: 106 pracas tem buracos no meio, somando 1.613 dias (parte relevante nas pontas,
  por novas concessoes/free-flow que entram durante a serie - EPR Parana, Nova 364, Nova 381).
- Alternativas: interpolar os buracos (inventa dado); mascarar (o Keras nao trata NaN
  nativamente).
- Impacto: descarta-se poucas janelas; **nenhum valor de volume e inventado**.

## D-012 - Combinacao ausente (praca/dia sem aquele tipo) = volume 0

- Data: 2026-10-13
- Decisao: em dia **observado**, um `tipo_de_veiculo` sem linha recebe volume **0**.
- Evidencia: o bruto so registra combinacoes com volume > 0 (minimo = 1; nao ha zeros).
  Logo, ausencia de linha significa "nao passou aquele tipo naquele dia", nao "dado perdido".
- Alternativas: tratar como ausente (geraria NaN em dia que de fato existiu).
- Impacto: distingue "tipo ausente em dia existente" (0) de "dia inteiro ausente" (missing,
  tratado em D-011).

## D-013 - Protocolos de split (cronologico global + por grupos/pracas)

- Data: 2026-10-13
- Decisao: (1) **cronologico global**, cortando pelo **dia do alvo**: treino <= 2025-11-30,
  val 2025-12-01 a 2026-04-30, teste 2026-05-01 a 2026-08-31; (2) **por grupos/pracas**,
  70/15/15 com seed 42.
- Evidencia: cortar pelo dia do alvo (e nao pelo inicio da janela) garante que o alvo mais
  recente do treino seja anterior ao alvo mais antigo da validacao (verificado em
  `src/dataset.py:testes_rapidos`).
- Alternativas: split aleatorio de linhas (vazamento - descartado).
- Impacto: com janela 30, X_train (105.754 x 30 x 13), X_val (30.678 x 30 x 13),
  X_test (27.467 x 30 x 13).

## D-014 - Padronizacao ajustada SOMENTE no treino (calculada em float64)

- Data: 2026-10-13
- Decisao: padronizar os canais de volume com media/desvio do **treino**; calcular essas
  estatisticas em **float64**.
- Evidencia: media calculada em float32 sobre ~3,17 milhoes de valores gera deriva de ~1e-3;
  recalculada em float64, o residuo cai para ~1e-4.
- Alternativas: padronizar com o dataset inteiro (vazamento - viola PDF Sec.2.2).
- Impacto: atende ao requisito "parametros de pre-processamento ajustados somente no treino".
