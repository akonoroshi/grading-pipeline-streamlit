from typing import Dict
import re
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from grading_pipeline import GradingSystem

class Grade(BaseModel):
    """
    Represents a single grade for an assignment.
    """
    justification: str = Field(..., description="Justification for the score")
    score: float = Field(..., description="Score awarded for the assignment")

class GradingSystemLLM(GradingSystem):
    """
    Main grading system that handles the grading process using semantic similarity
    and grammar checking.
    """
    def __init__(self, model_name: str = "deepseek-chat"):
        super().__init__()
        temperature = 0
        max_tokens = 1024
        max_retries = 3
        if "deepseek" in model_name:
            llm = ChatDeepSeek(
                model_name=model_name,
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
        self.llm = llm.with_structured_output(Grade)
    
    def _get_score(self, item: Dict, assignment_text: str):
        critetia_text = item['criteria']
        if item['description'] and item['criteria'] != item['description']:
            critetia_text += f"\n{item['description']}"
        for label in item['labels']:
            critetia_text += f"\n - {label['label']}: {label['description']}"
        messages = [
            ("system", f"You are a grading assistant. Your task is to evaluate the student's assignment based on the following criteria on a scale of 0-{item['points']}:\n{critetia_text}"),
            ("human", assignment_text)
        ]
        output = self.llm.invoke(messages)

        results = {
            'description': item['description'],
            'max_points': item['points'],
            'justification': output.justification,
            'score': output.score,
            'word_count': len(assignment_text.split())
        }

        return self._add_labels(output.score, item, results)