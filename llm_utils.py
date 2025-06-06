import re
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def get_model(model_name: str):
    temperature = 0
    max_tokens = 2048
    max_retries = 3
    if model_name == "deepseek-chat":
        llm = ChatDeepSeek(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    elif model_name == "deepseek-r1":
        llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    elif "gpt" in model_name or re.search(r"o\d", model_name):
        if re.search(r"o\d", model_name):
            temperature = 1
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    else:
        raise NotImplementedError(f"Model {model_name} is not supported.")
    return llm