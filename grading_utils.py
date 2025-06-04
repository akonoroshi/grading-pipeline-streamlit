import torch
from grading_pipeline_similarity import GradingSystemSimilarity
from grading_pipeline_llm import GradingSystemLLM
from grading_pipeline_dummy import GradingSystemDummy

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
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        grading_system = GradingSystemSimilarity(device=device)

    else:
        grading_system = GradingSystemLLM(model_name=method)
    return grading_system
