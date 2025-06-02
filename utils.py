import torch
from grading_pipeline_similarity import GradingSystemSimilarity
from grading_pipeline_llm import GradingSystemLLM
from grading_pipeline_dummy import GradingSystemDummy

def get_grades(method: str, final_assignment_path: str, rubric_path: str):
    """
    Initialize grading system and grade assignment
    """
    if method == "test-chat":
        grading_system = GradingSystemDummy()
    elif method == "similarity":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        grading_system = GradingSystemSimilarity(device=device)

    else:
        grading_system = GradingSystemLLM(model_name=method)
    results = grading_system.grade_assignment(final_assignment_path, rubric_path)

    return results
