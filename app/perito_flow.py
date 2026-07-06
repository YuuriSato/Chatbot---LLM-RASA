from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

import ollama
from langgraph.graph import END, StateGraph


DEFAULT_MODEL = "codex-dental"
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11435")
DEFAULT_OPTIONS = {
    "temperature": 0.1,
    "num_predict": 2048,
}
INTERNAL_MARKER_PATTERN = re.compile(r"<unused94>\s*thought.*?<unused95>", re.DOTALL)


class CodexDentalState(TypedDict, total=False):
    image_path: str
    model: str
    ollama_host: str
    anatomic_description: str
    pathology_detection: str
    final_report: str


def get_client(state: CodexDentalState) -> ollama.Client:
    return ollama.Client(host=state.get("ollama_host", DEFAULT_OLLAMA_HOST))


def clean_model_response(response: object) -> str:
    text = str(response)
    if "<unused95>" in text:
        text = text.rsplit("<unused95>", 1)[-1]
    text = INTERNAL_MARKER_PATTERN.sub("", text)
    text = text.replace("<unused94>", "").replace("<unused95>", "")
    return text.strip()


def describe_anatomy(state: CodexDentalState) -> dict[str, str]:
    response = get_client(state).generate(
        model=state.get("model", DEFAULT_MODEL),
        prompt=(
            "Faca um mapeamento anatomico desta radiografia odontologica. "
            "Identifique regioes visiveis da arcada, dentes presentes quando possivel, "
            "condicoes osseas gerais, restauracoes/aparelhos visiveis e limitacoes da imagem. "
            "Nao conclua patologias nesta etapa; apenas descreva achados anatomicos observaveis. "
            "Nao escreva pensamento, raciocinio interno ou marcadores especiais."
        ),
        images=[state["image_path"]],
        options=DEFAULT_OPTIONS,
    )
    return {"anatomic_description": clean_model_response(response["response"])}


def detect_issues(state: CodexDentalState) -> dict[str, str]:
    prompt = f"""
Mapa anatomico:
{state["anatomic_description"]}

Tarefa:
Busque especificamente sinais visuais de caries, perda ossea, dentes inclusos,
alteracoes periapicais, abscessos aparentes, desalinhamentos e integridade de raizes.
Descreva achados por quadrante/regiao quando a imagem permitir.
Se nao houver evidencia visual suficiente, diga isso explicitamente.
Nao escreva pensamento, raciocinio interno ou marcadores especiais.
""".strip()

    response = get_client(state).generate(
        model=state.get("model", DEFAULT_MODEL),
        prompt=prompt,
        images=[state["image_path"]],
        options=DEFAULT_OPTIONS,
    )
    return {"pathology_detection": clean_model_response(response["response"])}


def compile_report(state: CodexDentalState) -> dict[str, str]:
    prompt = f"""
Consolide a analise radiologica odontologica em um relatorio tecnico.

Descricao anatomica:
{state["anatomic_description"]}

Achados suspeitos:
{state["pathology_detection"]}

Inclua:
1. Resumo dos achados.
2. Sinais que exigem avaliacao profissional.
3. Se ha necessidade de atendimento urgente, usando "SIM", "NAO" ou "INDETERMINADO".
4. Aviso de que isto nao substitui laudo clinico/radiologico.
Nao escreva pensamento, raciocinio interno ou marcadores especiais.
""".strip()

    response = get_client(state).generate(
        model=state.get("model", DEFAULT_MODEL),
        prompt=prompt,
        options=DEFAULT_OPTIONS,
    )
    return {"final_report": clean_model_response(response["response"])}


def build_workflow():
    workflow = StateGraph(CodexDentalState)

    workflow.add_node("mapeamento", describe_anatomy)
    workflow.add_node("diagnostico", detect_issues)
    workflow.add_node("relatorio", compile_report)

    workflow.set_entry_point("mapeamento")
    workflow.add_edge("mapeamento", "diagnostico")
    workflow.add_edge("diagnostico", "relatorio")
    workflow.add_edge("relatorio", END)

    return workflow.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisa raios-X odontologicos usando Ollama + MedGemma 1.5.",
    )
    parser.add_argument(
        "image_path",
        help="Caminho da imagem a ser analisada.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Modelo Ollama a usar. Padrao: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Host do Ollama. Padrao: {DEFAULT_OLLAMA_HOST}.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime apenas o resultado final em JSON.",
    )
    return parser.parse_args()


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def validate_image_path(image_path: str) -> Path:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {path}")
    if not path.is_file():
        raise ValueError(f"O caminho informado nao e um arquivo: {path}")
    return path


def main() -> int:
    configure_stdio()
    args = parse_args()

    try:
        image_path = validate_image_path(args.image_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2

    app = build_workflow()
    input_data: CodexDentalState = {
        "image_path": str(image_path),
        "model": args.model,
        "ollama_host": args.ollama_host,
    }
    final_state: CodexDentalState = dict(input_data)

    try:
        for output in app.stream(input_data):
            node_name, node_output = next(iter(output.items()))
            final_state.update(node_output)

            if not args.json:
                print(f"\n[{node_name}]")
                for key, value in node_output.items():
                    print(f"{key}: {value}")

    except ollama.ResponseError as exc:
        print(f"Erro do Ollama: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Erro inesperado: {exc}", file=sys.stderr)
        return 1

    if args.json:
        result = {
            "anatomic_description": final_state.get("anatomic_description", ""),
            "pathology_detection": final_state.get("pathology_detection", ""),
            "final_report": final_state.get("final_report", ""),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
