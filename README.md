# RuleBook Assistant

**RuleBook Assistant** is a LangChain-based Retrieval-Augmented Generation (RAG) system that answers questions about TTRPG rulebooks such as *Dungeons & Dragons*, *Monopoly*, or others you add. It supports multiple query translation strategies, dynamic vector backends (Chroma, Pinecone), and game-specific namespaces for flexible expansion.

---

## Quickstart

### 1. Clone the Repository and Set Up Environment

```bash
git clone https://github.com/your-name/rulebook-assistant.git
cd rulebook-assistant
```

#### (a) Create a Conda Environment

```bash
conda create -n ruleBookAssistant python=3.11 -y
conda activate ruleBookAssistant
```

#### (b) Install the Project

```bash
pip install -e .
```
> All required dependencies are listed in `pyproject.toml`.

---

### 2. Prepare Your Environment

- Add your API keys in `config/keys.json`:

```json
{
  "OPENAI_API_KEY": "your-openai-key",
  "LANGCHAIN_API_KEY": "your-langsmith-key",
  "PINECONE_API_KEY": "your-pinecone-key",
  "PINECONE_ENV": "your-pinecone-environment",
  "PINECONE_INDEX_NAME": "rulebook-assistant"
}
```

- Ensure tracing is enabled by default (`LANGCHAIN_TRACING=true`) in `config/config.py`.

---

### 3. Add Your PDFs

Organize rulebooks by game name:

```
data/
└── raw/
    ├── dnd/
    │   ├── DMG.pdf
    │   ├── PHB.pdf
    │   └── MM.pdf
    └── monopoly/
        └── Monopoly-Guide.pdf
```
- Ensure `config/supported_games.json` contains all the games you have added.
- __Note__: The game folder name must match the game's `abbr` seen in the above json.
---

## Index Rulebooks

You must index each game before querying it. Use the CLI tool:

```bash
python scripts/index.py --game dnd --target pinecone
```
> Include `--batch_size` to set batch size for indexing (default: 200)

Available options:

- `--game` – Name of the folder in `data/raw/`, also used as the Pinecone namespace.
- `--target` – Choose between `pinecone` or `chroma`.
- `--batch_size` – to set batch size for indexing (default: 200).

Example:

```bash
python scripts/index.py --game monopoly --target chroma
```

---

## Ask Questions

Once indexed, you can query the rulebooks:

```bash
python main.py \
  --question "What are good plot hooks for a beginner campaign?" \
  --strategy multi_query \
  --target pinecone \
  --namespace dnd
```

Arguments:

- `-q, --question` – User question
- `-s, --strategy` – Query translation strategy:
  - `passthrough`, `multi_query`, `rag_fusion`, `hyde`, `step_back`, `decompose`
- `-t, --target` – `pinecone` or `chroma`
- `-n, --namespace` – Game name (must match indexed folder)

---

## Strategies

The assistant supports several RAG strategies:

- **Multi Query** – Rephrases and expands the original query
- **RAG Fusion** – Aggregates results from multiple subqueries
- **HyDE** – Uses hypothetical answers to guide retrieval
- **Step Back** – Queries broader context before zooming in
- **Decomposition** – Breaks complex queries into simpler ones

---

## Development Tips

- Reindex anytime PDFs are updated
- Namespace = game name
- `data/vectorstore/{namespace}` is used for local Chroma
- Pinecone is used remotely with persistent namespaces
- `pip install -e .` is needed only once per environment

---

## Project Structure

```
src/
└── rag/
    ├── indexing.py
    ├── query_translation.py
    ├── query_construction.py
    ├── retrieval.py
    ├── generation.py
    └── tracing.py
scripts/
└── index.py
main.py
config/
└── keys.json
```

---

## Future Enhancements

- API or Web UI
- Automatic subfolder detection
- Scheduled re-indexing
- Hybrid vectorstore support

---
