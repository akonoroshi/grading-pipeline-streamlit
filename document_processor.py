from typing import Dict, List
import os
import re
from pypdf import PdfReader
from docx import Document

class DocumentProcessor:
    """
    Base class for processing documents.
    """

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extracts text from a PDF file.
        Args:
            file_path: Path to the PDF file
        Returns:
            Extracted text as a string
        """
        text = ""
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extracts text from a DOCX file.
        Args:
            file_path: Path to the DOCX file
        Returns:
            Extracted text as a string
        """
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def process_document(self, file_path: str) -> str:
        """
        Processes a document based on its file type.
        Args:
            file_path: Path to the document
        Returns:
            Extracted text as a string
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        if file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        raise ValueError("Unsupported file format")
    

class AssignmentProcessor(DocumentProcessor):
    """
    Handles the processing of different document types (PDF and DOCX)
    and text extraction from these documents.
    """

class RubricProcessor(DocumentProcessor):
    """
    Handles the processing and extraction of grading criteria from rubric files.
    """

    def _extract_points(self, section: str):
        """
        Extract points using various patterns.
        Args:
            section: The section of text to search for points
        Returns:
            Extracted points as a float
        """
        points_patterns = [
            r'(?:Points:|points:?)\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:Points|points)',
            r'(?:Worth|worth|Value|value):\s*(\d+(?:\.\d+)?)',
            r'\((\d+(?:\.\d+)?)\s*pts?\)',
            r'\[(\d+(?:\.\d+)?)\s*pts?\]',
            r'(\d+(?:\.\d+)?)\s*(?:marks|Marks|point|Point)'
        ]
        
        for p_pattern in points_patterns:
            points_match = re.search(p_pattern, section, re.IGNORECASE)
            if points_match:
                return float(points_match.group(1))
        return None
    
    def _extract_range(self, line: str):
        """
        Extracts a range of points from a line.
        Args:
            line: The line of text to search for a range
        Returns:
            Extracted range as a tuple (min, max)
        """
        range_patterns = [
            r'(?:Points:|points:?)\s*(\d+)\s*-\s*(\d+)',
            r'(\d+)\s*-\s*(\d+)\s*(?:Points|points)',
            r'(?:Worth|worth|Value|value):\s*(\d+)\s*-\s*(\d+)',
            r'\((\d+)\s*-\s*(\d+)\s*pts?\)',
            r'\[(\d+)\s*-\s*(\d+)\s*pts?\]',
            r'(\d+)\s*-\s*(\d+)\s*(?:marks|Marks|point|Point)'
        ]
        
        for r_pattern in range_patterns:
            match = re.search(r_pattern, line)
            if match:
                return (float(match.group(1)), float(match.group(2))), re.split(r_pattern, line)
        return None, None
    
    def _format_criteria(self, criteria_line: str, description_lines: List[str], points: float, labels: List[Dict]) -> Dict:
        criteria = re.sub(r'^(?:\d+[\.\)]\s*|\*\s*|\-\s*|\•\s*)', '', criteria_line)
        criteria = criteria.strip(':').strip()
        
        description = ' '.join(description_lines).strip()
        if not description:
            description = criteria
        
        return {
            'criteria': criteria,
            'description': description,
            'points': points,
            'labels': labels
        }
    
    def extract_rubric(self, file_path: str) -> List[Dict]:
        """
        Extracts rubric details from a file.
        Args:
            file_path: Path to the rubric file
        Returns:
            List of dictionaries containing criteria, descriptions, and points
        """
        try:
            # Extract text
            text = self.process_document(file_path)
            
            print(text)
            if not text.strip():
                raise ValueError("No text could be extracted from the rubric PDF")
            
            # Define patterns for identifying rubric sections
            section_patterns = [
                r'\n(?=Criteria:|CRITERIA:|Criterion:|CRITERION:)',
                r'\n(?=\d+\.|\d+\))',
                r'\n(?=[A-Z][^a-z]+:)',
                r'\n(?=\*|\-|\•)',
                r'\n\n(?=[A-Z][^\.]+(?:\.|:))'
            ]
            
            # Process rubric sections
            rubric_items = []
            for pattern in section_patterns:
                sections = re.split(pattern, text)
                if len(sections) > 1:
                    for section in sections:
                        if not section.strip():
                            continue
                        
                        # Extract points using various patterns
                        points = self._extract_points(section)
                        
                        if points is not None:
                            # Extract criteria and description
                            lines = [line.strip() for line in section.split('\n') if line.strip()]
                            
                            criteria_line = None
                            description_lines = []
                            capture_description = False
                            labels = []
                            sub_criteria_lines = []
                            sub_description_lines = []
                            capture_sub_description = False
                            sub_points = []
                            sub_labels = []
                            
                            for line in lines:
                                sub_range, splitted_range = self._extract_range(line)
                                if sub_range is not None:
                                    # If a line contains a range, it is a label
                                    label_item = {
                                            'label': splitted_range[0].strip('(').strip(),
                                            'description': splitted_range[-1].strip(')').strip(':').strip(),
                                            'min': sub_range[0],
                                            'max': sub_range[1]
                                    }
                                    if capture_sub_description:
                                        sub_labels[-1].append(label_item)
                                    else:
                                        labels.append(label_item)
                                else:
                                    sub_point = self._extract_points(line)
                                    if sub_point is not None:
                                        # If a line contains points, it is a criteria line or a sub-criterion
                                        if criteria_line is None: # Criteria line
                                            criteria_line = line
                                            capture_description = True
                                        else: #Sub-criterion
                                            sub_criteria_lines.append(line)
                                            capture_sub_description = True
                                            sub_description_lines.append([])
                                            sub_labels.append([])
                                            sub_points.append(sub_point)
                                    elif capture_sub_description:
                                        sub_description_lines[-1].append(line)
                                    elif capture_description:
                                        description_lines.append(line)
                            
                            if criteria_line:
                                criteria_item = self._format_criteria(criteria_line, description_lines, points, labels)
                                criteria_item['sub_criteria'] = []
                                for sub_criteria_line, sub_description, sub_point, sub_label in zip(sub_criteria_lines, sub_description_lines, sub_points, sub_labels):
                                    sub_criteria_item = self._format_criteria(sub_criteria_line, sub_description, sub_point, sub_label)
                                    criteria_item['sub_criteria'].append(sub_criteria_item)
                                rubric_items.append(criteria_item)
                    
                    if rubric_items:
                        break
            
            if not rubric_items:
                raise ValueError("No valid criteria found in rubric")
            
            print("Extracted Rubric Items:", rubric_items)
            return rubric_items
            
        except Exception as e:
            raise Exception(f"Error processing rubric: {str(e)}")