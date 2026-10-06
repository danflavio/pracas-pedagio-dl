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

---

## Pendente

- D-008 (a verificar) - Duplicatas na virada de ano: o mesmo registro pode aparecer em dois
  arquivos (ex.: dezembro/2024 no arquivo de 2024 e no de 2025). Verificar antes de considerar a
  unificacao concluida.
- Contagem de linhas: 2024 = 3.312.763 | 2025 = 3.274.157 | 2026 = 2.597.234 (total 9.184.154).
