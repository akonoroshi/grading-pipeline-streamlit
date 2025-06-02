# grading-pipeline

## Setup
Install dependencies by running
```
pip install -r reqirememts.txt
```

or

```
conda env create -f environment.yml
```

Then, run the streamlit app
```
streamlit run app.py
```

If you want to run LLMs locally (e.g., deepseek-r1), you can use Ollama with this app. To do so,

1. [Download](https://ollama.ai/download) and install Ollama onto the available supported platforms (including Windows Subsystem for Linux)
2. Fetch available LLM model via `ollama pull <name-of-model>`