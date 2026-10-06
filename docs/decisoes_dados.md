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
- Decisao: converter `mes_ano` (formato `dd/mm/yyyy`, dia sempre 01) para data na F2.
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

## D-006 - `categoria_eixo` com tipos misturados (pendente)

- Data: 2026-10-05
- Decisao: (pendente) investigar os valores nao numericos de `categoria_eixo` e definir o tipo
  antes de gravar o Parquet.
- Evidencia: na leitura completa dos anos o pandas emite
  `DtypeWarning: Columns (5: categoria_eixo) have mixed types`. A amostra de 100 mil linhas havia
  inferido `int64`; o problema esta em linhas fora da amostra.
- Alternativas: forcar `dtype` no `read_csv` (mascara o problema, nao resolve); investigar quais
  valores existem e por que.
- Impacto: sem resolver, a coluna pode virar texto de forma inconsistente entre os anos.

---

## Pendente

- D-007 (a verificar) - Duplicatas na virada de ano: o mesmo registro pode aparecer em dois
  arquivos (ex.: dezembro/2024 no arquivo de 2024 e no de 2025). Verificar antes de considerar a
  unificacao concluida.
- Contagem de linhas: 2024 = 3.312.763 | 2025 = 3.274.157 | 2026 = 2.597.234 (total 9.184.154).
