"""Stage 7a — o que sustenta cada campo.

Havia aqui um score: `confiança = min(leitura, modelo) × validação`, com um piso
por corroboração por cima. Três números de estatuto epistêmico diferente —
medido, declarado e herdado — multiplicados até virarem um só, que a triagem
comparava com um limiar por classe de campo. Nada disso sobreviveu, e por uma
razão que a própria tabela de limiares denunciava: cruzada com os quatro pesos
de leitura possíveis, ela só produzia uma decisão — `baixa` reprova em campo
crítico e de identidade, todo o resto passa. Dezesseis células para repetir o
que a lista de quem leu já dizia, com dois dígitos de precisão que a evidência
não tem.

O que restou são fatos, e eles não se somam entre si:

* **quem leu** — `reading_level` e `corroborating_kinds`, medidos pelo consenso
* **a base de referência** — `corroborated_keys`, medida contra um registro
  externo escrito sem referência a este documento
* **as classes de campo** — quais campos quebram algo downstream se vierem
  errados; hoje sem limiar, só a classificação

As duas primeiras respondem perguntas diferentes — *conseguimos ler isto?* e
*este ativo é quem o aviso diz que é?* — e por isso continuam separadas até o
relatório. Um ISIN lido só pelo OCR continua lido só pelo OCR depois de bater
com a base; um ISIN lido por três mecanismos continua ausente da base se a base
não o tiver. Colapsar as duas num escalar foi exatamente o que o piso por
corroboração tentou fazer, e o resultado foi um `max(conf, min(source_score,
0.97))` capado pelo próprio `source_score`: no doc 07 as quatro chaves de
identidade casaram na base e o número não se mexeu um milésimo.
"""

from __future__ import annotations

from ..models import ExtractedField, ValidationResult
from ..config import load_config


def load_thresholds(config_dir: str | None = None) -> dict:
    return load_config("thresholds", config_dir)


def field_class(name: str, thresholds: dict) -> str:
    for class_name, spec in (thresholds.get("field_classes") or {}).items():
        if name in (spec.get("fields") or []):
            return class_name
    return "default"


def is_critical(name: str, thresholds: dict) -> bool:
    """Critical means a wrong value breaks something downstream.

    Financial terms and identity qualify; descriptive context does not. This is
    what stops the pipeline from treating a mis-read listing segment as
    seriously as a mis-read ISIN.
    """
    return field_class(name, thresholds) in ("financial_critical", "identity")


def cutting_tools(validations: list[ValidationResult]) -> dict[str, list[str]]:
    """Quais ferramentas reprovaram cada campo.

    Era um fator multiplicativo por status (PASS 1,0 · WARN 0,6 · FAIL 0,3) que
    descia até a confiança do campo. O fator não dizia nada que o achado da
    ferramenta já não dissesse, e dizia *pior*: um campo lido com perfeição pelos
    três mecanismos aparecia com 0,29 porque a cronologia do documento se
    contradiz — número que manda o revisor abrir o recorte, que é justamente onde
    não está o problema.

    O nome de quem reprovou fica, porque esse sim é acionável.
    """
    culprits: dict[str, list[str]] = {}
    for validation in validations:
        if validation.status.value not in ("warn", "fail"):
            continue
        for name in validation.fields_affected:
            culprits.setdefault(name, []).append(validation.tool)
    return culprits


def corroborated_keys(validations: list[ValidationResult]) -> set[str]:
    """Fields an independent registry confirmed.

    Só `MATCH`, que exige duas chaves independentes caindo na mesma linha da
    base. Um `WEAK_MATCH` ou um `MISMATCH` já vêm com o próprio código de motivo
    e não confirmam coisa alguma.
    """
    keys: set[str] = set()
    for validation in validations:
        if validation.tool == "lookup_golden_record" and validation.detail.get("status") == "MATCH":
            keys.update(validation.detail.get("agreeing_keys") or [])
    return keys


def note_reference_matches(
    fields: dict[str, ExtractedField], validations: list[ValidationResult]
) -> None:
    """Anota, campo a campo, o que a base de referência confirmou.

    Roda nos dois motores, no lugar onde `score()` rodava. Não decide nada: a
    confirmação externa é um segundo fato sobre o campo, ao lado de quem o leu,
    e é a triagem — não uma multiplicação — que decide o que fazer com cada um.
    """
    for name in corroborated_keys(validations):
        field = fields.get(name)
        if field is not None and field.value is not None:
            field.notes.append("valor confirmado na base de referência")
