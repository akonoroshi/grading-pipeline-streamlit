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

We also need `poppler` to convert pdf to images. Poppler is very easy to install by following the instructions [on their website](https://poppler.freedesktop.org/). Or choose one of the following option:

### MacOS with homebrew

```bash
brew install poppler
```

### Debian/Ubuntu

```bash
sudo apt-get install -y poppler-utils
```

## Enabling LLMs

You can use your local LLMs or external APIs. We currently support OpenAI APIs, DeepSeek APIs (chat only), DeepSeek r1 through Ollama, and qwen2.5vl though Ollama. You can add more LLMs by updating `get_model` in `llm_utils.py`.

### Use local LLMs

If you want to run LLMs locally (e.g., deepseek-r1), you can use Ollama or llama.cpp with this app. Note that you need llama.cpp for RAG.

#### llama.cpp

1. Follow the instriction in [llama.cpp Quick start](https://github.com/ggml-org/llama.cpp/tree/master?tab=readme-ov-file#quick-start) to install llama.cpp.
2. Launch a llama-server via `llama-server -hf <hugging-face model>`.

#### Ollama

1. [Download](https://ollama.ai/download) and install Ollama onto the available supported platforms (including Windows Subsystem for Linux)
2. Fetch available LLM model via `ollama pull <name-of-model>`

### Use APIs

If you want to call APIs (e.g., OpenAI models), you must store your API key in an environment variable like

```bash
export OPENAI_API_KEY="your-api-key"
```

## Running the app

First, create a folder called "problems" at the same directory as README. Inside the "problems" folder, create a folfer for each problem (you may create more subdirectories if you want) and place the images (problem diagrams and choices) in the folder. Then, edit `domain_information.py` to update the details of the problems, the path to your textbook, your subject domain, and a preferred name of your course. The keys of the variable `PROBLEMS` must be the same as the name of the folder of each problem. Finally, you can simply run the streamlit app to test it out:

```bash
streamlit run app.py
```
