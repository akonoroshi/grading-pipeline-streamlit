import tempfile
import os
import streamlit as st
from docx import Document
from speech_input import get_speech_input
from utils import get_grades
from chatbot import response_generator

#float_init(theme=True, include_unstable_primary=False)

def save_uploaded_file(uploaded_file):
    """
    Saves an uploaded file to a temporary location.
    
    Args:
        uploaded_file: The file uploaded through Streamlit's file uploader
    
    Returns:
        str: Path to the saved temporary file
        None if there was an error saving the file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def save_speech_to_docx(speech_text: str) -> str:
    """
    Converts speech text to a DOCX file.
    
    Args:
        speech_text: The text to be saved in the DOCX file
    
    Returns:
        str: Path to the saved temporary DOCX file
        None if there was an error saving the file
    """
    try:
        # Create a new document
        doc = Document()
        doc.add_paragraph(speech_text)
        
        # Save to temporary file
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.docx').name
        doc.save(temp_path)
        return temp_path
    except Exception as e:
        st.error(f"Error saving speech to file: {e}")
        return None

def write_scores(criterion, details, prefix=""):
    st.write(f"{prefix}**Criterion:** {criterion}")
    st.write(f"{prefix}**Description:** {details['description']}")
    st.write(f"{prefix}**Max Points:** {details['max_points']:.1f}")
    if 'similarity' in details:
        st.write(f"{prefix}**Similarity Score:** {details['similarity']:.2f}")
    st.write(f"{prefix}**Points Earned:** {details['score']:.2f}")
    if 'label' in details:
        st.write(f"{prefix}**Label:** {details['label']}")
    if 'justification' in details:
        st.write(f"{prefix}**Explanation:** {details['justification']}")

def write_results(results):
    with st.container(height=300):
        st.header("Grading Results")
        st.write(f"Final Grade: {results['final_grade']*100:.1f}%")
        st.write(f"Total Points Earned: {results['total_points_earned']:.1f}/{results['total_points_possible']:.1f}")
        
        # Display detailed criteria breakdown
        st.subheader("Criteria Breakdown:")
        for criterion, details in results['criteria_scores'].items():
            st.write("\n---")
            write_scores(criterion, details)
            if len(details['sub_scores']) > 0:
                st.write("\n**Sub-Criteria Breakdown:**")
                for sub_criterion, sub_details in details['sub_scores'].items():
                    write_scores(sub_criterion, sub_details, "    ")
            
            # Display grammar feedback if available
            if 'feedback' in details:
                st.write("\n**Grammar Analysis:**")
                stats = details['feedback']['statistics']
                st.write(f"- Words: {stats['word_count']}")
                st.write(f"- Sentences: {stats['sentence_count']}")
                if details['feedback']['errors']:
                    st.write("\n**Detailed Error Feedback:**")
                    for error in details['feedback']['errors']:
                        st.write(f"- {error['message']}")
                        if 'suggestion' in error:
                            st.write(f"  Suggestion: {error['suggestion']}")

def chatbox():
    """
    User interface for the chat feature.
    """
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Create a container for the chatbox
    height = 500
    with st.container(height=height):
        c = st.container(height=height-90)
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            c.chat_message(message["role"]).markdown(message["content"])
        
        # Accept user input
        #messages = st.container(height=100)
        if prompt := st.chat_input("Say something"):
            c.chat_message("user").markdown(prompt)
            st.session_state['messages'].append({"role": "user", "content": prompt})
            response = response_generator()
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            c.chat_message("assistant").markdown(response)

        
def main():
    """
    Main function that runs the Streamlit interface.
    Handles:
    1. File uploads
    2. Voice input
    3. Grading process
    4. Results display
    """
    st.title("Assignment Grading System")
    
    # Initialize session state for storing speech text and file path
    if 'speech_text' not in st.session_state:
        st.session_state.speech_text = None
    if 'speech_file_path' not in st.session_state:
        st.session_state.speech_file_path = None
    
    # Input method selection
    input_method = st.radio(
        "Choose input method for assignment",
        ["File Upload", "Voice Input"]
    )
    
    assignment_path = None
    
    # Handle file upload input
    if input_method == "File Upload":
        assignment_file = st.file_uploader(
            "Upload Assignment (PDF or DOCX)", 
            type=['pdf', 'docx']
        )
        if assignment_file:
            assignment_path = save_uploaded_file(assignment_file)
    # Handle voice input
    else:
        st.write("Click the button below and speak your answer")
        if st.button("Start Recording"):
            with st.spinner("Listening..."):
                speech_text = get_speech_input()
                if speech_text:
                    st.session_state.speech_text = speech_text
                    st.write("Transcribed text:")
                    st.write(speech_text)
                    st.session_state.speech_file_path = save_speech_to_docx(speech_text)
                else:
                    st.error("No speech detected or could not transcribe audio")
        
        # Display transcribed text if it exists
        if st.session_state.speech_text:
            st.write("Current transcribed text:")
            st.write(st.session_state.speech_text)
            assignment_path = st.session_state.speech_file_path
    
    # Rubric upload (always file upload)
    rubric_file = st.file_uploader(
        "Upload Rubric (DOCX only)", 
        type=['docx']
    )

    method = st.selectbox(
        "Select Grading Method",
        ("test-chat", "similarity", "deepseek-chat", "gpt-4.1-nano", "o4-mini"))
    
    # Grade for the first time or re-grade
    if 'results' not in st.session_state:
        st.session_state.results = []
        st.session_state.graded = False
    if st.button("Grade Assignment"):
        # Validate inputs
        if (input_method == "File Upload" and not assignment_path) or \
           (input_method == "Voice Input" and not st.session_state.speech_file_path):
            st.error("Please provide an assignment (via file upload or voice input)")
            return
        if not rubric_file:
            st.error("Please upload a rubric file")
            return
            
        try:
            with st.spinner("Processing..."):
                # Save rubric file
                rubric_path = save_uploaded_file(rubric_file)
                
                if not rubric_path:
                    st.error("Error saving rubric file")
                    return
                
                # Use the appropriate path based on input method
                final_assignment_path = assignment_path if input_method == "File Upload" \
                    else st.session_state.speech_file_path
                
                # Initialize grading system and grade assignment
                results = get_grades(method, final_assignment_path, rubric_path)
                
                # Cleanup temporary files
                if final_assignment_path:
                    os.unlink(final_assignment_path)
                if rubric_path:
                    os.unlink(rubric_path)
                
                # Clear session state after successful grading
                st.session_state.speech_text = None
                st.session_state.speech_file_path = None
                st.session_state.results.append(results)
                st.session_state.graded = True
                
        except Exception as e:
            st.error(f"Error processing files: {e}")

    # Display grades and chatbox if grading has been done
    if st.session_state.graded:
        # Display results and chatbox
        write_results(st.session_state.results[-1])
        chatbox()

if __name__ == "__main__":
    main() 