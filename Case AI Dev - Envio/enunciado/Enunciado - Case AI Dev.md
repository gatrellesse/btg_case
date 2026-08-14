# Case Técnico — AI Developer (Asset Servicing / BTG Pactual)
## Contexto
Asset Servicing processa avisos de eventos corporativos (proventos, grupamentos, bonificações) que chegam em formatos heterogêneos — PDFs nativos, escaneados, com layouts e terminologia variados — e precisam virar registros estruturados, confiáveis e auditáveis para alimentar processos downstream (cálculo de provento, custódia, conciliação, base regulatória).
Neste contexto, erros de extração são erros financeiros e regulatórios: classificar errado o tipo de provento muda o tratamento tributário; datas incoerentes quebram conciliações e valores inventados viram prejuízo.
## Tarefa
Construa um agente code-first que receba o lote de documentos fornecido e produza, para cada documento, um registro estruturado validado, com tratamento de incerteza.
Decida a arquitetura, as bibliotecas e o desenho de tools utilizadas. Fique à vontade para usar o Provider de preferência (OpenAI, Anthropic, Google, etc.).
### Requisitos mínimos
1. Extração dos campos: emissor, ISIN, ticker, tipo de evento, datas relevantes (aprovação, data com, ex, pagamento, conforme o caso), valor/proporção e moeda.
2. Classificação correta do tipo de evento (dividendo, JCP, bonificação, grupamento, etc.) — atenção ao que distingue um do outro e ao tratamento que cada tipo exige.
3. Validação de cada registro contra a base de referência `golden_records.csv` e contra regras de coerência (ex.: consistência entre datas, consistência entre valor bruto e líquido), usando tool / function calling.
4. Confiança e rastreabilidade: atribua níveis de confiança às extrações justificando-as.
5. Roteamento de incerteza: documentos ou campos de baixa confiança devem ser derivados para atuação humana. A decisão também deve ser justificada.
6. Saída. Um JSON por documento + um relatório de exceções curto. O schema é decisão sua, mas a saída deve permitir que um operador de Asset Servicing confie no registro e audite cada valor sem reabrir o documento — inclusive quando o aviso está incompleto, ambíguo ou pouco legível. Esperamos minimamente para cada registro: o que foi extraído e de onde, quão confiável é cada campo, o resultado da validação contra a base de referência, e o que precisa de revisão humana (e por quê). Como você estrutura isso — e o que decide incluir além disso — será parte da avaliação do Case.

## Entregáveis
- Repositório com o código (instruções de execução no README).
- README com as decisões de arquitetura, incluindo explicitamente o que você decidiu não fazer e por quê (trade-offs).
- A saída gerada sobre o lote (os JSONs + o relatório de exceções).
## O que fornecemos
- `documentos/`: lote de avisos de eventos corporativos (PDFs).
- `golden_records.csv`:  base de referência de emissores/ISIN/ticker para validação.
## Regras e prazo
- Prazo: entrega até segunda-feira, por e-mail/link de repositório.
- Uso de IDEs com IA (Cursor, Claude Code, Copilot etc.) é permitido. Haverá uma sessão técnica ao vivo (45 min) após a entrega, em que pediremos para você estender e depurar o próprio código; saiba defender suas decisões.
## Observações
O lote é sintético, mas modelado em avisos reais de companhias abertas brasileiras (estrutura, terminologia e datas no padrão B3/CVM). Os nomes de empresas, CNPJs e ISINs são fictícios. Premissas que você precisar assumir (ex.: o que conta como "baixa confiança", quais regras de coerência aplicar) devem ser documentadas — parte do que avaliamos é seu critério, não apenas o resultado.