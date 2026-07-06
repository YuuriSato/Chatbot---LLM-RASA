# Perito Visual Local

Projeto local para analisar imagens odontologicas/intraorais e classificar a integridade visual:

- `REAL`
- `ALTERADA_MANUALMENTE`
- `ALTERADA_DIGITALMENTE`
- `IA_GERADA_EDITADA`
- `INDETERMINADO`

O sistema usa Ollama local com `codex-dental:latest`, baseado em `medgemma1.5:4b`.
Antes da LLM, a aplicacao roda uma pericia local com hash, dHash perceptual,
ELA simples, nitidez, ruido e comparacao opcional com a imagem original.

## Estrutura

```text
app/                     Aplicacao web e CLI
models/                  Modelfiles do Ollama
data/                    Dataset e JSONL de treino
runtime/                 Historico, uploads, cache e outputs gerados
reports/                 Relatorios CSV/Markdown/JSON
scripts/training/        Scripts de dataset, LoRA e merge
scripts/download/        Scripts auxiliares de download Hugging Face
docs/                    Documentacao complementar
logs/                    Logs antigos de treino/download/web
tests/                   Imagens de teste locais
```

## Servico Web

Subir a pagina:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11435"
$env:VISION_MODEL = "codex-dental:latest"
.\.venv\Scripts\python.exe .\app\web_alteracao.py
```

Acesse:

```text
http://localhost:9090
```

Na tela, envie a imagem suspeita. Se tiver a foto original, envie tambem no
campo opcional "Imagem original opcional"; isso melhora bastante a deteccao de
edicoes localizadas por IA.

Arquivos persistidos pela UI:

```text
runtime/uploads/
runtime/analysis_cache/
runtime/analysis_history.json
runtime/integrity_calibration.json
```

O historico salva a imagem suspeita, a original opcional, o score local,
as evidencias locais e a fonte do veredito.

## Ollama

Baixar modelo base:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama pull medgemma1.5:4b
```

Criar/recriar persona:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama create codex-dental -f .\models\Modelfile-Codex-Dental
```

## CLI

```powershell
.\.venv\Scripts\python.exe .\app\perito_flow.py .\tests\test_dental.png --json
```

## Dataset

Estrutura esperada:

```text
data/dataset_dental/
  real/
  alterada_manualmente/
  alterada_digitalmente/
  ia/
  indeterminado/
```

Gerar JSONL:

```powershell
.\.venv\Scripts\python.exe .\scripts\training\prepare_training_jsonl.py
```

O arquivo padrao gerado fica em:

```text
data/codex_training.jsonl
```

## Fine-Tuning

O fine-tuning do MedGemma 4B em CPU nao foi viavel nesta maquina. Para GPU/Colab, veja:

```text
docs/FINE_TUNING.md
```
