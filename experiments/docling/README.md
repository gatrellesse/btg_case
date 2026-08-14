# Consenso entre modalidades — experimento

Mede o que o pipeline do Docling não produz: **concordância entre leitores sobre
a mesma região**. O pipeline padrão elimina os clusters que já têm texto
programático antes de chamar o OCR, então as duas fontes nunca leem a mesma
coisa — por construção não há segunda opinião, e nenhum dos quatro scores de
confiança dele desce ao nível do campo (`table_score` sequer é atribuído).

## Rodar

```bash
python experiments/docling/make_hybrid.py          # constrói o PDF híbrido
mkdir -p out/consensus/docs && cp <pdfs> out/consensus/docs/

python experiments/docling/extract.py --mode parser --docs out/consensus/docs
python experiments/docling/extract.py --mode ocr    --docs out/consensus/docs
python experiments/docling/extract.py --mode vlm    --docs out/consensus/docs

python experiments/docling/consensus.py            # -> out/consensus/consensus.json
```

## As três extrações

| modo | configuração | `ReaderKind` | no scan |
|---|---|---|---|
| `parser` | `do_ocr=False` | `text_layer` | abstém-se (0 bytes) |
| `ocr` | `OcrMode.FULL_PAGE` | `ocr_deterministic` | lê tudo |
| `vlm` | `VlmPipeline`, granite-docling 258M | `ocr_generative` | lê tudo |

`FULL_PAGE` importa: em `post_process_cells` ele descarta as células do PDF, de
modo que a leitura sai 100% de pixels mesmo em documento nativo. Sem isso o OCR
nunca serviria de segunda opinião num PDF com camada de texto.

## Ordem: ler → tipar → validar → votar

Cada passo existe por um caso medido neste lote.

**Tipar antes de comparar.** `0,1124300000` e `0,112430000` são strings
diferentes e o mesmo `Decimal` — o VLM perdeu um zero à direita no doc 07.
Comparar texto cru inventaria um conflito onde não há.

**Tipar é estrito, nunca conserta.** O OCR leu a data de pagamento do doc 07
como `.../08/22026`. Isso não é uma data; sai da votação. Se fosse "reparado"
para `21/08/2026` por adivinhação, a concordância seria fabricada — o leitor
nunca leu aquilo. Reparo vem depois do vencedor, com proveniência própria.

**Validar antes de votar.** Os dois ISIN corrompidos pelo VLM têm 11 caracteres
onde a ISO 6166 exige 12. Estrutura decide sozinha, sem segunda opinião e sem
revisão humana. O conflito de 3 vias vira candidato único sobrevivente.

**Checksum é INFO, não veto.** Os ISIN deste lote são sintéticos e *nenhum*
fecha o mod 10 — o de verdade (`US0378331005`) fecha, então a implementação está
certa e o dado é que é fabricado. Reprovar por checksum reprovaria o lote
inteiro. Mesma convenção do `ValidationStatus.INFO` do projeto.

**Eliminado sai do voto, não do placar.** Candidato reprovado é sinal de que
aquele leitor sofreu naquela região: `score *= 0.9` por eliminado. Sem isso um
campo onde dois de três leitores produziram lixo sairia "unânime entre os
sobreviventes".

## Peso por classe de mecanismo

Não são votos equivalentes:

```
text_layer         0.02    string do próprio programa que gerou o PDF
ocr_deterministic  0.10    classifica tinta renderizada
ocr_generative     0.25    gera tokens — 2 dos 8 ISIN corrompidos
```

Concordância entre mecanismos **disjuntos** multiplica as taxas. Entre dois
mecanismos derivados dos mesmos pixels, não: eles erram junto (confundem o mesmo
par de glifos pelo mesmo motivo), então o combinado é `min(err) * 0.5`.

Consequência que a contagem de votos erraria: `text_layer` sozinho (0.98) vale
mais que `ocr + vlm` concordando (0.95). No doc 03, `parser` e `ocr` estavam
certos e o VLM errado — maioria simples acerta ali por sorte de aritmética, não
por evidência.

## Três eixos, conjuntivos

| eixo | pergunta | onde mora |
|---|---|---|
| transcrição | os leitores concordam sobre o que está escrito? | este módulo |
| identificação | a região é a do campo? | `config/anchors.yaml`, âncora mais longa |
| validade | tipa, passa estrutura, fecha a aritmética? | tipagem + `repair.py` |

Não se somam. No doc 01 os três leitores leem `0,4275` na mesma célula e o
documento não tem valor líquido nenhum: transcrição unânime, identificação
falha. Média entre eixos esconderia. A âncora mais longa (`valor bruto por acao
ordinaria` ganha de `valor liquido`) é o que impede o campo de existir.

O eixo que falha também decide *o que se pergunta ao humano*: conflito de
transcrição é "qual destas duas strings está certa?" com os dois recortes lado a
lado; falha de identificação é "esta caixa é o valor líquido?". Perguntas,
telas e custos diferentes.

## Resultado medido (3 documentos, 27 campos)

```
high: 10     medium: 6     single_reader: 11
```

- **nativo** — 9 campos `high` (0.9995), três leitores em todos
- **scan** — parser se abstém; 6 campos `medium` (0.95); ISIN resolvido em
  0.81 com o VLM eliminado pela estrutura, sem humano
- **híbrido** — quase tudo cai para `single_reader ocr` (0.90), *exceto*
  `tax_rate`, que fica `high`: a alíquota aparece na prosa, que continua na
  camada de texto. Na mesma página, campo de prosa tem corroboração tripla e
  campo de tabela tem um leitor só

O híbrido é o caso que separa "decide por documento" de "decide por região", e
foi construído para isso — `make_hybrid.py` rasteriza a tabela do doc 01 e apaga
o texto por baixo (verificado: 181 palavras na página, 0 na região da tabela).

## Custo

O VLM roda a 12–23 s/página contra ~1 s do parser. Três passadas completas em
todo documento não se paga: a forma sensata é acionar a segunda e a terceira
modalidade só em campo crítico, ou quando a primeira passada devolve
`parse_score` baixo ou nenhuma âncora casada.
