import pickle
import os
import re
import tempfile
import copy
import json
from pdf2image import convert_from_path
from domain_information import PROBLEMS, TEXT_PATH
from document_processor import RubricProcessor

base_messsage = [{
    "role": "system",
    "content": """I have generic rubrics to evaluate student written rationales for their choice to any multiple-choice problems. Tailor them to the given problem by incorporating knowledge needed to solve it. 
I will give you the problem, example rationale that received a full mark, rubrics, and seven images. The first image is the system diagram of the problem, the second image contains the choices, and the remaining images are textbook pages that may contain knowledge helpful to solve the problem."""
}]
processor = RubricProcessor()
text = processor.process_document("./rubrics/domain-specific rubrics.docx")
for problem_name ,problem in PROBLEMS.items():
    pkl_path = os.path.join("problems", problem_name, "pages.pkl")
    with open(pkl_path, 'rb') as f:
        pages = pickle.load(f)
    json_path = os.path.join("problems", problem_name, "description.json")
    with open(json_path, "r") as f:
        description = json.load(f)
    
    
    page_images = []
    modified_text = ""
    with tempfile.TemporaryDirectory() as path:
        for page in pages:
            page_images.extend(convert_from_path(
                TEXT_PATH,
                thread_count=os.cpu_count() - 1,
                output_folder=path,
                dpi=300,
                fmt="png",
                first_page=page,
                last_page=page,
                paths_only=True
            ))
        user_content = processor.get_user_content(problem, page_images)

        for pattern in processor.section_patterns:
            sections = re.split(pattern, text)
            if len(sections) > 1:
                for i, section in enumerate(sections):
                    print("Section", section)
                    if not section.strip():
                        continue
                    user_content_rubric = copy.deepcopy(user_content)
                    user_content_rubric[0]["text"] += f"\nExample rationale: {description["rationale"]}\nRubric: {section}"
                    messages = [{
                        "role": "user",
                        "content": user_content_rubric
                    }]
                    response = processor.llm.invoke(messages)
                    print("Response:", response)
                    modified_text += f"{i+1}. {response}\n"
        """
        user_content[0]["text"] += f"\nRubric: {text}"
        messages = base_messsage + [{
                        "role": "user",
                        "content": user_content
                    }]
        response = processor.llm.invoke(messages)
        print("Response:", response.rubrics)
        modified_text = response.rubrics
        """

    with open(os.path.join("problems", problem_name, "rubrics.txt"), "w") as f:
        f.write(modified_text)
