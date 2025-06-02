from typing import Dict
from grading_pipeline import GradingSystem

class GradingSystemDummy(GradingSystem):
    """
    Main grading system that handles the grading process using semantic similarity
    and grammar checking.
    """
    
    def _get_score(self, item: Dict, assignment_text: str):
        results = {
            'description': item['description'],
            'max_points': item['points'],
            'justification': "Test justification",
            'score': item['points'] * 0.7,
            'word_count': len(assignment_text.split())
        }

        return self._add_labels(item['points'] * 0.7, item, results)