# Perito Visual Local

Projeto local para analisar imagens odontologicas/intraorais e classificar a integridade visual:

- `REAL`
- `ALTERADA_MANUALMENTE`
- `ALTERADA_DIGITALMENTE`
- `IA_GERADA_EDITADA`
- `INDETERMINADO`

O sistema usa pericia local rapida e, quando o checkbox de LLM detalhada esta
marcado, usa Gemini por padrao. Ollama local com `codex-dental:latest` continua
disponivel como alternativa.
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

## Como subir no localhost

Execute os comandos abaixo na raiz do projeto:

```powershell
cd C:\Okta7\Chatbot_ImageVision
```

Crie e ative o ambiente virtual, caso ainda nao exista:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Configure a chave do Gemini antes de subir a aplicacao web. Para definir apenas
no terminal atual:

```powershell
$env:GEMINI_API_KEY = "sua-chave-do-gemini"
$env:LLM_PROVIDER = "gemini"
$env:GEMINI_MODEL = "gemini-2.5-flash"
```

Para deixar salvo no Windows para os proximos terminais:

```powershell
[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "sua-chave-do-gemini", "User")
[Environment]::SetEnvironmentVariable("LLM_PROVIDER", "gemini", "User")
[Environment]::SetEnvironmentVariable("GEMINI_MODEL", "gemini-2.5-flash", "User")
```

Depois de criar ou alterar essas variaveis, feche e abra o terminal novamente
ou reinicie o servidor web. O processo do Python so enxerga variaveis que
existiam quando ele foi iniciado.

Suba a aplicacao web local:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11435"
$env:VISION_MODEL = "codex-dental:latest"
$env:WEB_PORT = "9090"
python .\app\web_alteracao.py
```

Acesse:

```text
http://localhost:9090
```

Se alterar `WEB_PORT`, troque tambem a porta no navegador. Exemplo:

```powershell
$env:WEB_PORT = "8080"
python .\app\web_alteracao.py
```

```text
http://localhost:8080
```

Na tela, envie a imagem suspeita. Se tiver a foto original, envie tambem no
campo opcional "Imagem original opcional"; isso melhora bastante a deteccao de
edicoes localizadas por IA.

O checkbox "Usar LLM detalhada via Gemini" envia a imagem otimizada e as
evidencias locais para o Gemini. Sem o checkbox, a resposta usa apenas a
pericia local rapida. Para voltar ao Ollama local no checkbox, defina:

```powershell
$env:LLM_PROVIDER = "ollama"
```

Arquivos persistidos pela UI:

```text
runtime/uploads/
runtime/analysis_cache/
runtime/analysis_history.json
runtime/integrity_calibration.json
```

O historico salva a imagem suspeita, a original opcional, o score local,
as evidencias locais e a fonte do veredito. Na interface web, ele e carregado
em abas com paginacao, para nao transferir/renderizar todos os registros de uma
vez.

## Arquitetura Integridade Primeiro

O sistema atua como barreira de seguranca visual antes de qualquer leitura
clinica. Ele nao emite diagnostico odontologico; primeiro verifica se a imagem
parece integra, modificada ou insuficiente para uma pericia confiavel.

Fluxo de decisao:

1. Calibracao local: hash SHA256 e dHash perceptual reconhecem exemplos ja
   ensinados como `REAL`, `MODIFICADO` ou `IA_GERADA_EDITADA`.
2. Qualidade da imagem: largura, altura, megapixels, menor lado, nitidez global
   e blur classificam a imagem como `boa`, `limitada` ou `insuficiente`.
3. Pericia local: ELA, ruido, nitidez regional, marcadores coloridos e
   comparacao opcional com original geram o score tecnico.
4. Invalidacao por integridade: linhas verdes, setas, textos, borroes, recortes
   ou diferencas fortes contra a original invalidam a imagem como `MODIFICADO`
   ou `IA_GERADA_EDITADA`, sem depender da LLM.
5. Desempate por Gemini: quando o score local fica intermediario e o checkbox
   esta marcado, o Gemini recebe as evidencias locais para procurar texturas
   nao biologicas, dentes/raizes fundidos e densidade radiografica incoerente.
6. Auditoria CRAG: quando a LLM detalhada roda, uma segunda chamada interna
   pode revisar o laudo inicial contra ELA, ruido, nitidez, qualidade, MOQAM e
   comparacao com original. Se houver conflito entre achado visual aparente e
   evidencia forense local, o resultado `REAL` e bloqueado.

MOQAM / Escaneamento por Receptividade:

- Micro-escala: a LLM revisa pequenas regioes de pixel/textura/borda, incluindo
  areas de desmineralizacao aparente, carie/lesao apenas como achado visual e
  transicoes que possam indicar reconstrucao por IA.
- Macro-escala: a LLM revisa continuidade biologica de mandibula, arcada,
  proteses, implantes, raizes e separacao entre dentes.
- Conflito de escala: se a macro-escala parecer coerente, mas a micro-escala
  mostrar alteracao localizada forte, a imagem nao deve ser classificada como
  `REAL`. Implantes e metais nao devem mascarar pequenas edicoes vizinhas.

Se a qualidade for insuficiente e nao houver fraude visual forte, o resultado
sera `INDETERMINADO` com `TIPO: QUALIDADE_INSUFICIENTE`. Baixa resolucao nao
prova alteracao; ela limita a confiabilidade da pericia.

Auditoria CRAG / Self-Reflective RAG:

- O laudo inicial do Gemini e tratado como hipotese, nao como conclusao final.
- A auditoria reavalia achados aparentes como carie/lesao, dentes fundidos,
  densidade incoerente ou textura suspeita contra as metricas locais.
- Se a auditoria indicar que um achado clinico aparente pode ser artefato de
  pixel/ruido/ELA, o sistema retorna `INDETERMINADO` ou `MODIFICADO` conforme o
  score local. O sistema continua sem emitir diagnostico odontologico.

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

## Comparacao de Raios-X por Paciente

O script abaixo compara exames odontologicos de atendimentos diferentes e estima
se parecem pertencer ao mesmo paciente. Ele nao emite diagnostico odontologico.

Manifest CSV aceito:

```csv
image_path,patient_name,dentist,appointment_date,appointment_id
.\runtime\uploads\exame_01.jpg,Paciente A,Dentista Y,2026-06-01,1
.\runtime\uploads\exame_02.jpg,Paciente A,Dentista X,2026-07-01,2
```

Tambem sao aceitos cabecalhos em portugues como `caminho_arquivo`,
`nome_paciente`, `dentista`, `data_atendimento` e `numero_atendimento`.

Executar com manifest:

```powershell
.\.venv\Scripts\python.exe .\scripts\compare_dental_xrays.py --manifest .\data\xrays_manifest.csv --output .\reports\xray_patient_comparison.csv --excel .\reports\xray_patient_comparison.xlsx
```

Executar direto com uma pasta de imagens:

```powershell
.\.venv\Scripts\python.exe .\scripts\compare_dental_xrays.py --image-dir .\tests --output .\reports\xray_patient_comparison.csv
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
