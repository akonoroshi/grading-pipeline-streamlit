import base64
from typing import List
from llm_utils import get_model
from grading_utils import PROBLEMS

def image_content(image_path: str) -> List[dict]:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}

if __name__ == "__main__":
    llm = get_model("qwen2.5vl")
    problem = PROBLEMS[0]
    user_content = [{"type": "text","text": problem["problem"]}]
    for img in problem["images"]:
        user_content.append(image_content(img))
    msgs = [
        {
            "role": "system",
            "content": "I will give you a multiple-choice physics problem. The first image is the system diagram of the problem, and the second image contains the choices. First, describe the diagram of the given problem and gather information that can help solve it. Then, provide your solution with the correct answer choice (a-e) at the end."
        },
        {
            "role": "user",
            "content": user_content
        }
        ]
    print(llm.invoke(msgs).content)