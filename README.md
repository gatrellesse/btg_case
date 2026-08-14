# Agente de extração de eventos corporativos — Asset Servicing

## Como rodar

```bash
git clone <repo> && cd case_btg
./run.sh
```

Só isso. O script cria o venv, instala e processa o lote de exemplo; a primeira execução
baixa ~5GB (docling, torch, modelos de OCR) e leva alguns minutos, as seguintes reusam.
Precisa de Python 3.12+.

**Env — a chave é requisito.** As modalidades de modelo fazem parte do desenho, e um lote
sem elas mede outra coisa. Sem chave o script para e diz qual variável falta.

```bash
export ANTHROPIC_API_KEY=...                # ou CLAUDE_API_KEY, ou uma linha no .env
export AS_MODEL=anthropic:claude-sonnet-5   # padrão

# o provider é o prefixo do modelo — a chave cobrada muda junto
export OPENAI_API_KEY=...   AS_MODEL=openai:gpt-4.1
export GOOGLE_API_KEY=...   AS_MODEL=google-gla:gemini-2.5-flash
```

Trocar de provider é trocar uma string: os nós de modelo são agentes Pydantic AI tipados
contra o mesmo schema. Vale para o motor `graph`, que é o padrão; o `linear` da v1 fala
com o cliente Gemini direto.

Qualquer opção do CLI atravessa o script. Para chamar direto, sem ele:

```bash
PYTHONPATH=src ./.venv/bin/python -m asset_servicing.cli run \
  --docs "Case AI Dev - Envio/documents" \
  --golden "Case AI Dev - Envio/golden_records/golden records.csv" \
  --out out/
```

**Saídas em `out/`, na ordem de quem abre:** `exceptions_report.pdf` (o relatório do
operador, com o recorte da região ao lado de cada valor) · `viewer.html` (a página com
todo campo desenhado sobre ela) · `run_summary.json` (STP, exception rate, motivos) · um
JSON por documento, que é a camada de máquina.

---

## Frameworks e modelos

| | Papel |
|---|---|
| **LangGraph** | estado, fan-out, arestas condicionais, ciclos com contador. Dono da topologia |
| **Pydantic AI** | os nós de modelo: saída tipada, tools por assinatura, retry de schema |
| **Pydantic** | os schemas do registro — um JSON fora do contrato falha alto, não vira outra coisa |
| **Docling** | a ingestão em três modos **isolados**: `parser`, `ocr` (`FULL_PAGE`), `vlm` |
| **PP-OCRv6** (ONNX) | o OCR determinístico, em CPU |
| **granite-docling 258M** | o VLM que emite DocTags. Terceiro leitor, e só leitor |
| **TableFormer** | estrutura de tabela; é dele a geometria que o VLM pede emprestada |
| **pymupdf** | camada de texto, render, recortes e o relatório em PDF |
| **rapidfuzz** | casamento aproximado onde o OCR derruba caracteres |
| **PyYAML** | `config/`: marcadores, âncoras, classes de campo, glossário |

Quatro papéis usam modelo: **classificar · extrair · varrer o que faltou · justificar**.
Nenhum decide sozinho — todos operam sobre o que a camada determinística já leu.

---

## Arquitetura

### Ingestão por consenso

```
PDF ─┬─► docling:parser   do_ocr=False        camada de texto · a string do gerador
     ├─► docling:ocr      OcrMode.FULL_PAGE   PP-OCRv6 · descarta células do PDF
     └─► docling:vlm      granite-docling     DocTags · tokens sobre a página
                    │
                    ▼
       tipar → validar → votar      região a região, por sobreposição
         estrito, não conserta · estrutura elimina · sobreviventes votam
                    │
                    ▼
       blocks[]  text · bbox · score · reader_kind · alternatives[]
                 score = quantos mecanismos DISJUNTOS concordaram
```

Um modelo de linguagem usado como leitor primário pode devolver um valor plausível que não
está na página. Por isso a leitura é feita por mecanismos que não alucinam e não mudam de
comportamento entre chamadas — o LLM entra depois, sobre o que eles já leram, nunca no
lugar deles.

Três leitores, três reservas diferentes: o **parser** lê texto programático, mas a camada
de texto de um PDF pode ser adulterada sem alterar a aparência. O **OCR** é determinístico
e independe do que o PDF diz ser, mas erra pontuação e funde caracteres. O **VLM** é o mais
recente e o mais sujeito a alucinar. Nenhum decide sozinho: um leitor repetido não é
consenso, é o mesmo viés três vezes.

O roteamento segue o que o PDF oferece — nativo, o parser entra como terceiro leitor ao
lado de OCR e VLM; escaneado, sem camada de texto, a ingestão se apoia só em OCR e VLM. Uma
extração só sai automática com pelo menos duas fontes concordando (Parser + OCR/VLM, ou
OCR + VLM); leitor sozinho, inclusive o parser, nunca basta.

| Reader | bbox | layout | hosted | papel |
|---|---|---|---|---|
| `TextLayerReader` (pymupdf) | ✅ | parcial | ❌ | tier 0 nativo |
| `RapidOCRReader` (PP-OCR/ONNX) | ✅ | ❌ | ❌ | tier 0 escaneado |
| `PaddleOCRVLReader` (0.9B) | ✅ | ✅ + ordem | ❌ (GPU) | tier 2 |
| `MistralOCRReader` (OCR 4) | ✅ | ✅ | ✅ | tier 2 de produção |

Trocar de motor de leitura é decisão de arquitetura, não de config — a tabela acima é o
argumento: cada leitor declara os sinais de que o pipeline depende (bbox, layout, hosted),
e um leitor que não os entrega não pode ser trocado sem repensar o resto.

Um sinal ficou de fora da tabela de propósito: a confiança que o próprio motor reporta
sobre si mesmo. Ela existe nos três, mas não é comparável entre eles — um VLM
autoregressivo não tem essa métrica, o OCR reporta uma média por página, o parser não tem
o que reportar. O único sinal que sobra, e que vale para os três igualmente, é
**concordância**: duas leituras independentes batendo no mesmo texto.

### O grafo

```
                    blocks[]
                       │
   ┌───────────────────┼                        CLASSIFICAÇÃO
   ▼                   ▼                           fan-out
rule_classifier   text_classifier
   regex              LLM/texto
   └───────────────────│
                       ▼
                   consensus   2/2 para AUTO · peso igual · sem veto
                       │
   ┌───────────────────┼                           EXTRAÇÃO
   ▼                   ▼                           fan-out
rule_extractor    text_extractor
   └───────────────────┼
                       ▼
                     merge     confronta nos campos críticos
                       ▼
                   grounding ──┐ evidência ausente → reprompt_extract (máx 1)
                       ▼       └──────────────────────────────────────┘
                     repair
                       ▼
                     sweep     última varredura: o que faltou, contra as 3 leituras
                       ▼
                    validate ──┐ leitura ambígua → disambiguate (máx 1)
                       ▼       └──────────────────────────────────────┘
                     triage    veredito em código
                       ▼
                    reporter   justificativa restrita aos reason codes
```

**Por que cada nó existe:**

| Camada | Nó | Por quê |
|---|---|---|
| Classificação | `rule_classifier` | regex sobre marcadores definitórios da lei — opinião determinística, sem modelo; prior fraco já é resultado válido |
| | `text_classifier` | a mesma pergunta ao LLM, sobre o texto extraído — erra de um jeito não correlacionado com a regra |
| | `consensus` | 2/2 para AUTO, peso igual, sem veto de nenhuma modalidade; qualquer divergência vai para revisão, porque classificar errado muda o tratamento tributário |
| Extração | `rule_extractor` | âncoras de rótulo do config, offline — o segundo par de olhos que o LLM enfrenta, a custo ≈ zero |
| | `text_extractor` | extração tipada por schema do tipo já classificado, com fallback genérico obrigatório |
| | `merge` | reúne candidatos por campo; não vota, guarda os dois — num campo financeiro, divergência não se resolve por maioria |
| Grounding e reparo | `grounding` | confere cada evidência contra o texto **cru**, antes de qualquer normalização — depois de virar ISO, uma data não aparece literalmente em lugar nenhum |
| | `reprompt_extract` | evidência ausente muda o prompt (+ exigência verbatim) e tenta de novo, no máx. 1×: só reexecuta porque o prompt ganhou informação nova |
| | `repair` | normaliza para pt-BR e deriva o que a regra do evento permite; padroniza formato, nunca decide |
| Varredura final | `sweep` | relê as três leituras inteiras atrás do que faltou (e do que um mecanismo só sustentava); 3 guardas em código decidem o que entra, para nunca reescrever o que já foi votado |
| Validação | `validate` | as seis tools determinísticas, sempre todas — cobertura de um controle não pode depender de o modelo ter lembrado de chamá-lo |
| | `disambiguate` | só resolve quando exatamente um candidato satisfaz a restrição (a cronologia do evento); dois sobreviventes seguem para o humano |
| | `triage` | veredito em código, a partir da severidade dos reason codes: AUTO, REVIEW ou BLOCKED |
| Reporter | `reporter` | escreve a narrativa a partir dos reason codes já atribuídos — a disposição já está calculada e ele não a toca |

