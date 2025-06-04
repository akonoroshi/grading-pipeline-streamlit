# grading-pipeline

## Setup

Install dependencies by running

```bash
pip install -r reqirememts.txt
```

or

```bash
conda env create -f environment.yml
```

Then, run the streamlit app

```bash
streamlit run app.py
```

### Use local LLMs

If you want to run LLMs locally (e.g., deepseek-r1), you can use Ollama with this app. To do so,

1. [Download](https://ollama.ai/download) and install Ollama onto the available supported platforms (including Windows Subsystem for Linux)
2. Fetch available LLM model via `ollama pull <name-of-model>`

### Use APIs

If you want to call APIs (e.g., OpenAI models), you must store your API key in an environment variable like

```bash
export OPENAI_API_KEY="your-api-key"
```
