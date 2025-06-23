import tempfile
import base64
import os
from typing import List
import requests
from numpy import argsort
from scipy.special import softmax
from pdf2image import convert_from_path
from grading_utils import PROBLEMS, TEXT_PATH
from llm_utils import get_model

SERVER_URL  = "http://127.0.0.1:8080/v1/chat/completions"

# prepare messages: one system message + one image message per file + the text prompt
base_messages = [
    {"role": "system", "content": "I will give you a multiple-choice physics problem. The first image is the system diagram of the problem, the second image contains the choices, and the third image is a page in a textbook that may help you answer. Output only the letter (a-e) of the correct answer without any justifications."},
]

def image_content(image_path: str) -> List[dict]:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}

class TextRetriever:
    def __init__(self,
                 model_name: str,
                 text_path: str,
                 top_logprobs: int = 15):
        self.model = get_model(
            model_name,
            max_tokens=1,
            temperature=0,
            server_url=SERVER_URL
            ).bind(logprobs=True, top_logprobs=top_logprobs)
        self.text_path = text_path

    def get_choice_logprob(self, problem: dict, text_image) -> List[dict]:
        # clone the base messages and append the forced‐completion prompt
        user_content = []
        for img_path in problem["images"]:
            user_content.append(image_content(img_path))
        user_content.append(image_content(text_image))
        user_content.append({"type": "text", "text": problem["problem"]})
        msgs = base_messages + [{
            "role": "user",
            "content": user_content
        }]

        response = self.model.invoke(msgs)
        return response.response_metadata["logprobs"]["content"][0]["top_logprobs"]

    def get_prob(self, result: List[dict],
                choices: List[str] = ["a", "b", "c", "d", "e"]):
        logprobs = {}
        for item in result:
            if item["token".lower()] in choices and item["token".lower()] not in logprobs:
                logprobs[item["token"].lower()] = item["logprob"]
        prob_list = []
        for c in choices:
            if c not in logprobs:
                logprobs[c] = float("-inf")
            prob_list.append(logprobs[c])
        return softmax(prob_list)
    
    def retrieve(self, problem: dict, k=5) -> List[str]:
        scores = []
        correct_idx = problem["choices"].index(problem["answer"])
        with tempfile.TemporaryDirectory() as path:
            text_images = convert_from_path(
                TEXT_PATH,
                thread_count=os.cpu_count() - 1,
                output_folder=path,
                paths_only=True,
                dpi=300,
                fmt="png"
            )
            for text_image in text_images:
                res = self.get_choice_logprob(problem, text_image)
                prob = self.get_prob(res, problem["choices"])
                scores.append(prob[correct_idx])
                print("Correct Probability:", prob[correct_idx])
        ranked = argsort(scores)[::-1] + 1
        return ranked[:5]

if __name__ == "__main__":
    model_name  = ["qwen2.5-vl", "gpt-4.1-nano"][1]
    retriever = TextRetriever(model_name, TEXT_PATH)
    for problem in PROBLEMS:
        res = retriever.retrieve(problem, 5)
        print("Top 5 ranked pages:", res)
