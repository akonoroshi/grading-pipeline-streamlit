import torch
from grading_pipeline_similarity import GradingSystemSimilarity
from grading_pipeline_llm import GradingSystemLLM
from grading_pipeline_dummy import GradingSystemDummy

PROBLEMS = [
    {
        "problem": "For a truss or a frame, each connection to the ground provides up to two force components and one moment depending on the type of support used. For the given structure, what are the correct reaction forces at the support A and D? A is a hinge support, and D is a roller support.",
        "images": [
            "../Updated Report and Collected Dataset_SBU/Assigned Questions and Collected Response/Cycle 1/Practice1 Problem.jpg",
            "../Updated Report and Collected Dataset_SBU/Assigned Questions and Collected Response/Cycle 1/Practice1 Choices.jpg"
            ]
        ,
        "choices": ["a", "b", "c", "d", "e"],
        "answer": "e"
    }
]
TEXT_PATH = "../samples/StaticsEngineeringMechanicsRCHibbelerbook12th.pdf"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_grading_system(method: str):
    """
    Initialize grading system and grade assignment
    """
    if "test-chat" in method:
        coefficient = 0.7
        if "low" in method:
            coefficient = 0.4
        elif "high" in method:
            coefficient = 0.95
        grading_system = GradingSystemDummy(coefficient=coefficient)
    elif method == "similarity":
        device = get_device()
        grading_system = GradingSystemSimilarity(device=device)

    else:
        grading_system = GradingSystemLLM(model_name=method)
    return grading_system
