#!/usr/bin/env bash
#
# Do clone ao relatório, num comando.
#
#   ./run.sh                    lote de exemplo, saída em out/
#   ./run.sh --engine linear    o pipeline linear da v1, para comparar
#   ./run.sh --docs meus_pdfs/ --golden minha_base.csv --out saida/
#
# Exige chave de API: as modalidades de modelo fazem parte do desenho, e um lote
# sem elas mede outra coisa. Sem chave o script para e diz qual variável falta.
#
#   export ANTHROPIC_API_KEY=...   # ou CLAUDE_API_KEY
#   export OPENAI_API_KEY=...      # com AS_MODEL=openai:gpt-4.1
#   export GOOGLE_API_KEY=...      # com AS_MODEL=google-gla:gemini-2.5-flash
#
# Qualquer opção do CLI passa direto. O que este script faz além disso é o que
# um clone novo precisa: criar o venv, instalar, conferir a chave e dizer onde
# a saída ficou.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"
DOCS_PADRAO="Case AI Dev - Envio/documents"
GOLDEN_PADRAO="Case AI Dev - Envio/golden_records/golden records.csv"
OUT_PADRAO="out"

msg() { printf '\033[1;34m▸\033[0m %s\n' "$1"; }
erro() { printf '\033[1;31m✗\033[0m %s\n' "$1" >&2; exit 1; }

# `--help` é pergunta sobre as opções, não pedido para processar um lote: sai
# antes de montar venv ou anunciar onde a saída ficaria.
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      sed -n '3,10p' "$0" | sed 's/^# \{0,1\}//'
      [ -x "$VENV/bin/python" ] && PYTHONPATH=src "$VENV/bin/python" -m asset_servicing.cli run --help
      exit 0
      ;;
  esac
done

# --- 1. Python -------------------------------------------------------------
# 3.12 não é capricho: o código usa `StrEnum` e o operador morsa em compreensão.
command -v python3 >/dev/null || erro "python3 não encontrado."
python3 - <<'EOF' || erro "Python 3.12+ é necessário (o código usa StrEnum)."
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
EOF

# --- 2. venv e dependências ------------------------------------------------
# O carimbo evita reinstalar a cada execução, e refaz sozinho quando o
# requirements muda — um `pip install` de ~5,7GB não pode rodar por engano.
if [ ! -x "$PY" ]; then
  msg "criando $VENV"
  python3 -m venv "$VENV"
fi

CARIMBO="$VENV/.requirements.sha"
ATUAL="$(sha256sum requirements.txt | cut -d' ' -f1)"
if [ "$(cat "$CARIMBO" 2>/dev/null || true)" != "$ATUAL" ]; then
  msg "instalando dependências (a primeira vez baixa ~5GB: docling, torch, modelos de OCR)"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
  echo "$ATUAL" > "$CARIMBO"
else
  msg "dependências já instaladas"
fi

# --- 3. a chave, que aqui é requisito --------------------------------------
# Quem sabe qual variável cada provider precisa é o próprio projeto
# (`agents._PROVIDER_KEYS`), então a checagem é feita lá dentro em vez de
# duplicar a tabela em bash — duas listas divergem no dia em que um provider é
# acrescentado só numa delas.
MODELO=""
ANTERIOR=""
OFFLINE=0
for arg in "$@"; do
  [ "$ANTERIOR" = "--model" ] && MODELO="$arg"
  [ "$arg" = "--offline" ] && OFFLINE=1
  ANTERIOR="$arg"
done

# `--offline` é um pedido explícito pelo caminho determinístico: quem o passa
# sabe o que quer, e cobrar chave de quem já disse que não vai usar modelo seria
# a checagem contradizendo a opção.
[ "$OFFLINE" = "1" ] && msg "--offline: caminho determinístico, sem chamada de modelo"
[ "$OFFLINE" = "1" ] || PYTHONPATH=src "$PY" - "$MODELO" <<'EOF' || exit 1
import sys
from asset_servicing.llm.agents import DEFAULT_MODEL, _PROVIDER_KEYS, load_dotenv, model_available

load_dotenv()  # também trata CLAUDE_API_KEY como alias de ANTHROPIC_API_KEY
modelo = (sys.argv[1] if len(sys.argv) > 1 else "") or DEFAULT_MODEL
provider = modelo.split(":", 1)[0]

if model_available(modelo):
    print(f"\033[1;34m▸\033[0m modelo: {modelo}")
    raise SystemExit(0)

variaveis = _PROVIDER_KEYS.get(provider)
print(f"\033[1;31m✗\033[0m falta a chave de API para o provider '{provider}' ({modelo}).",
      file=sys.stderr)
if variaveis:
    print(f"  export {variaveis[0]}=...   (ou uma linha {variaveis[0]}=... no .env)",
          file=sys.stderr)
    if len(variaveis) > 1:
        print(f"  aliases aceitos: {', '.join(variaveis[1:])}", file=sys.stderr)
else:
    print(f"  provider desconhecido. Conhecidos: {', '.join(sorted(_PROVIDER_KEYS))}",
          file=sys.stderr)
print("  outros providers:  AS_MODEL=openai:gpt-4.1  ·  AS_MODEL=google-gla:gemini-2.5-flash",
      file=sys.stderr)
raise SystemExit(1)
EOF

# --- 4. o lote -------------------------------------------------------------
# Os padrões só entram se o usuário não passou os dele; qualquer outra opção do
# CLI (--offline, --engine, --workers, --trace) atravessa sem tradução.
ARGS=("$@")
tem() { case " ${ARGS[*]-} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
tem --docs   || ARGS+=(--docs "$DOCS_PADRAO")
tem --golden || ARGS+=(--golden "$GOLDEN_PADRAO")
tem --out    || ARGS+=(--out "$OUT_PADRAO")

msg "processando o lote (~3min: são três leituras por documento)"
PYTHONPATH=src "$PY" -m asset_servicing.cli run "${ARGS[@]}"

# --- 5. onde a saída ficou -------------------------------------------------
SAIDA="$OUT_PADRAO"
for i in "${!ARGS[@]}"; do
  [ "${ARGS[$i]}" = "--out" ] && SAIDA="${ARGS[$((i + 1))]}"
done
echo
msg "pronto — abra primeiro:"
echo "  $SAIDA/exceptions_report.pdf   o relatório do operador, com o recorte ao lado de cada valor"
echo "  $SAIDA/viewer.html             a página com todo campo desenhado sobre ela"
echo "  $SAIDA/run_summary.json        STP, exception rate e os motivos por documento"
