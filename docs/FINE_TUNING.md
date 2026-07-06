# Fine-Tuning LoRA do MedGemma

Este guia fica como referencia. Na maquina atual, o fine-tuning em CPU do MedGemma 1.5 4B nao foi viavel; use GPU/Colab para treino real.

## Dataset

Organize as imagens em:

```text
data/dataset_dental/
  real/
  alterada_manualmente/
  alterada_digitalmente/
  ia/
  indeterminado/
```

Gere o JSONL:

```powershell
.\.venv\Scripts\python.exe .\scripts\training\prepare_training_jsonl.py
```

Saida padrao:

```text
data/codex_training.jsonl
```

## Dependencias

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-finetune-cpu.txt
```

Para Hugging Face:

```powershell
.\.venv\Scripts\hf.exe auth login
```

## Treino GPU/Colab

Com GPU, use os mesmos caminhos do projeto:

```bash
python scripts/training/finetune_lora_medgemma.py \
  --model-id google/medgemma-1.5-4b-it \
  --dataset-jsonl data/codex_training.jsonl \
  --image-root data/dataset_dental \
  --output-dir runtime/outputs/medgemma-dental-lora \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 512
```

## Treino CPU Experimental

Nao recomendado para esta maquina, mas o comando existe:

```powershell
.\.venv\Scripts\python.exe .\scripts\training\finetune_lora_medgemma.py `
  --model-id google/medgemma-1.5-4b-it `
  --dataset-jsonl data/codex_training.jsonl `
  --image-root data/dataset_dental `
  --output-dir runtime/outputs/medgemma-dental-lora `
  --cpu `
  --cpu-dtype bfloat16 `
  --max-length 512
```

## Merge/Exportacao

```bash
python scripts/training/merge_lora.py \
  --base google/medgemma-1.5-4b-it \
  --adapter runtime/outputs/medgemma-dental-lora \
  --output runtime/outputs/medgemma-dental-merged
```

Conversao para GGUF/Ollama depende de suporte atualizado do `llama.cpp` para a arquitetura usada.

## Observacoes

- Ollama e usado para inferencia; o treino deve ocorrer fora dele.
- A qualidade depende do dataset: use pares reais/falsos parecidos com seu caso.
- Para detectar IA em fotos intraorais, inclua exemplos reais com reflexos/saliva/metal e falsos com dentes reconstruidos por IA.
