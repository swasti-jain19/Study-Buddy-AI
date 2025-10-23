# application.py - Study Buddy AI with PDF RAG Support

import os
import sys

# Ensure `src/` is importable when running from project root
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from dotenv import load_dotenv

# Project imports
from src.utils.helpers import *  # QuizManager
from src.generator.question_generator import QuestionGenerator

# RAG import
try:
    from src.utils.rag_handler import RAGPipeline
except Exception:
    RAGPipeline = None

load_dotenv()

# --- Initialize RAG pipeline in session state ---
if 'rag_pipeline' not in st.session_state:
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if groq_api_key and RAGPipeline:
        try:
            st.session_state.rag_pipeline = RAGPipeline(groq_api_key)
        except Exception as e:
            st.session_state.rag_pipeline = None
            st.warning(f"⚠️ Warning initializing RAG pipeline: {e}")
    else:
        st.session_state.rag_pipeline = None

if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = False

if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

if 'pdf_study_mode' not in st.session_state:
    st.session_state.pdf_study_mode = "Ask Questions"


# --- RAG UI (Redesigned) ---
def render_rag_section():
    """Render PDF upload and RAG controls in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("📚 PDF Study Mode")

    if not RAGPipeline:
        st.sidebar.warning("RAG support is not available.")
        return

    if not st.session_state.rag_pipeline:
        st.sidebar.warning("⚠️ GROQ_API_KEY not configured.")
        return

    # Step 1: Enable PDF mode
    enable_pdf_mode = st.sidebar.checkbox(
        "📄 Enable PDF Study Mode",
        value=st.session_state.rag_enabled,
        help="Enable to upload and study from PDF documents"
    )
    st.session_state.rag_enabled = enable_pdf_mode

    # Step 2: Show upload only if enabled
    if enable_pdf_mode:
        st.sidebar.markdown("### Upload PDF")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload study materials (notes, textbooks, papers)"
        )

        if uploaded_file:
            if uploaded_file.name not in st.session_state.uploaded_docs:
                with st.spinner("📄 Processing PDF... This may take a moment."):
                    try:
                        result = st.session_state.rag_pipeline.process_pdf(uploaded_file)
                        st.sidebar.success(result)
                        st.session_state.uploaded_docs.append(uploaded_file.name)
                        st.sidebar.info("✅ PDF loaded! Select a mode below.")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error: {str(e)}")
                        if "memory" in str(e).lower():
                            st.sidebar.warning("💡 Try a smaller PDF")
            else:
                st.sidebar.success(f"✅ '{uploaded_file.name}' loaded")

        # Show loaded documents
        if st.session_state.uploaded_docs:
            st.sidebar.info(f"📄 {len(st.session_state.uploaded_docs)} document(s) loaded")
            
            # Clear documents button
            if st.sidebar.button("🗑️ Clear All Documents"):
                st.session_state.uploaded_docs = []
                st.session_state.rag_enabled = False
                groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
                if groq_api_key and RAGPipeline:
                    try:
                        st.session_state.rag_pipeline = RAGPipeline(groq_api_key)
                    except Exception:
                        st.session_state.rag_pipeline = None
                st.rerun()

        # Step 3: Select mode
        if st.session_state.uploaded_docs:
            st.sidebar.markdown("### Study Mode")
            pdf_mode = st.sidebar.radio(
                "What would you like to do?",
                ["📝 Ask Questions", "📋 Summarize PDF", "🎯 Generate Quiz"],
                help="Choose how you want to interact with the PDF"
            )
            
            # Store selected mode
            st.session_state.pdf_study_mode = pdf_mode.replace("📝 ", "").replace("📋 ", "").replace("🎯 ", "")
    else:
        # Reset when disabled
        if st.session_state.uploaded_docs:
            st.sidebar.info("💡 Enable PDF mode above to use uploaded documents")


# --- Main app ---
def main():
    st.set_page_config(page_title="Study Buddy AI", page_icon="🎓")

    # Initialize session state
    if 'quiz_manager' not in st.session_state:
        st.session_state.quiz_manager = QuizManager()
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False

    st.title("🎓 Study Buddy AI")

    # RAG controls in sidebar
    render_rag_section()

    # Check if PDF mode is active
    if st.session_state.rag_enabled and st.session_state.uploaded_docs:
        # ========== PDF MODE ACTIVE ==========
        st.sidebar.markdown("---")
        
        mode = st.session_state.pdf_study_mode
        
        if mode == "Summarize PDF":
            st.header("📋 PDF Summary")
            st.markdown("Generate a comprehensive summary of your uploaded PDF.")
            
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    result = st.session_state.rag_pipeline.query(
                        "Provide a comprehensive summary of this document. Include main topics, key points, and important concepts.",
                        mode="summarizer"
                    )
                    st.markdown("### Summary")
                    st.write(result["answer"])
                    
                    if result.get("source_documents"):
                        with st.expander("📖 View Source Sections"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.markdown(f"**Section {i}:**")
                                st.text(doc.page_content[:200] + "...")
                                if hasattr(doc, 'metadata'):
                                    st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")
        
        # Around line 95-170 in your application.py
        elif mode == "Generate Quiz":
            st.header("🎯 PDF Quiz Generator")
            st.markdown("Generate interactive quiz questions based on your PDF content.")
            
            col1, col2 = st.columns(2)
            with col1:
                num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)
            with col2:
                difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
            
            # Initialize question history if not exists
            if 'all_quiz_questions' not in st.session_state:
                st.session_state.all_quiz_questions = []
            
            # Show question history status
            if st.session_state.all_quiz_questions:
                total_previous = len(st.session_state.all_quiz_questions)
                st.info(f"📚 Question Bank: {total_previous} questions generated so far")
                
                # Add clear history button
                if st.button("🔄 Clear Question History"):
                    st.session_state.all_quiz_questions = []
                    st.success("Question history cleared! Next quiz will have fresh questions.")
                    st.rerun()
            
            if st.button("Generate Quiz from PDF", type="primary"):
                # Clear previous quiz first
                for key in ['pdf_quiz_questions', 'pdf_quiz_answers', 'pdf_quiz_submitted', 'pdf_quiz_generated']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                with st.spinner(f"Generating {num_questions} unique questions..."):
                    # Pass previous questions to avoid duplicates
                    questions = st.session_state.rag_pipeline.generate_structured_quiz(
                        num_questions, difficulty, previous_questions=st.session_state.all_quiz_questions

                    )

                    
                    if questions:
                        st.session_state.pdf_quiz_questions = questions
                        # Add to history for future duplicate prevention
                        st.session_state.all_quiz_questions.extend(questions)
                        st.session_state.pdf_quiz_answers = {}
                        st.session_state.pdf_quiz_generated = True
                        st.success(f"✅ Generated {len(questions)} unique questions!")
                        st.rerun()
                    else:
                        st.error("Failed to generate quiz. Please try again.")
            
            # Display quiz if generated
            if st.session_state.get('pdf_quiz_generated') and st.session_state.get('pdf_quiz_questions'):
                st.markdown("---")
                st.subheader("📝 Quiz Questions")
                
                questions = st.session_state.pdf_quiz_questions
                
                # Display each question
                for i, q in enumerate(questions, 1):
                    st.markdown(f"**Question {i}:** {q['question']}")
                    
                    # Radio button for answer selection
                    answer_key = f"pdf_q_{i}"
                    selected = st.radio(
                        "Select your answer:",
                        options=q['options'],  # Already formatted as ["A) text", "B) text", ...]
                        key=answer_key,
                        index=None
                    )

                    # Store answer
                    if selected:
                        if 'pdf_quiz_answers' not in st.session_state:
                            st.session_state.pdf_quiz_answers = {}
                        st.session_state.pdf_quiz_answers[i] = selected
                    
                    st.markdown("---")
                
                # Submit button
                if st.button("Submit Quiz", type="primary"):
                    if len(st.session_state.pdf_quiz_answers) == len(questions):
                        # Calculate score
                        correct = 0
                        for i, q in enumerate(questions, 1):
                            if st.session_state.pdf_quiz_answers.get(i) == q['correct_answer']:
                                correct += 1
                        
                        score = (correct / len(questions)) * 100
                        
                        st.session_state.pdf_quiz_submitted = True
                        st.session_state.pdf_quiz_score = score
                        st.session_state.pdf_quiz_correct = correct
                        st.rerun()
                    else:
                        st.warning("Please answer all questions before submitting!")
            
            # Display results if submitted
            if st.session_state.get('pdf_quiz_submitted'):
                st.markdown("---")
                st.header("🎯 Quiz Results")
                
                score = st.session_state.pdf_quiz_score
                correct = st.session_state.pdf_quiz_correct
                total = len(st.session_state.pdf_quiz_questions)
                
                st.metric("Your Score", f"{score:.1f}%", f"{correct}/{total} correct")
                
                st.markdown("---")
                st.subheader("📝 Detailed Results")
                
                # Show detailed results
                for i, q in enumerate(st.session_state.pdf_quiz_questions, 1):
                    user_answer = st.session_state.pdf_quiz_answers.get(i, "Not answered")
                    correct_answer = q['correct_answer']
                    is_correct = user_answer == correct_answer
                    
                    # Question header with status
                    if is_correct:
                        st.success(f"✅ Question {i}: {q['question']}")
                    else:
                        st.error(f"❌ Question {i}: {q['question']}")
                    
                    # Always show both answers (already formatted as "A) text")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Your answer:** {user_answer if user_answer else 'Not answered'}")
                    with col2:
                        st.write(f"**Correct answer:** {correct_answer}")
                    
                    st.markdown("---")
                
                # Reset button
                if st.button("Take Another Quiz"):
                    st.session_state.pdf_quiz_generated = False
                    st.session_state.pdf_quiz_submitted = False
                    st.session_state.pdf_quiz_questions = []
                    st.session_state.pdf_quiz_answers = {}
                    st.rerun()
        
        elif mode == "Ask Questions":
            st.header("💬 Ask Questions from PDF")
            st.markdown("Ask anything about the uploaded document!")
            
            user_input = st.chat_input("Type your question here...")
            
            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = st.session_state.rag_pipeline.query(user_input, mode="explainer")
                        st.write(result["answer"])
                        
                        if result.get("source_documents"):
                            with st.expander("📖 View Sources"):
                                for i, doc in enumerate(result["source_documents"], 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text(doc.page_content[:300] + "...")
                                    if hasattr(doc, 'metadata'):
                                        st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")
                                    st.markdown("---")
    
    else:
        # ========== REGULAR QUIZ MODE (No PDF) ==========
        st.sidebar.markdown("---")
        st.sidebar.header("📝 Regular Quiz Settings")
        
        question_type = st.sidebar.selectbox(
            "Question Type",
            ["Multiple Choice", "Fill in the Blank"],
            index=0
        )
        
        topic = st.sidebar.text_input("Enter Topic", placeholder="e.g., Python, History, Math")
        
        difficulty = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        
        num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=10, value=5)
        
        if st.sidebar.button("Generate Quiz"):
            st.session_state.quiz_submitted = False
            generator = QuestionGenerator()
            success = st.session_state.quiz_manager.generate_questions(
                generator, topic, question_type, difficulty, num_questions
            )
            st.session_state.quiz_generated = success
            st.rerun()
        
        # Display quiz
        if st.session_state.quiz_generated and st.session_state.quiz_manager.questions:
            st.header("📝 Quiz")
            st.session_state.quiz_manager.attempt_quiz()
            
            if st.button("Submit Quiz"):
                st.session_state.quiz_manager.evaluate_quiz()
                st.session_state.quiz_submitted = True
                st.rerun()
        
        # Display results
        if st.session_state.quiz_submitted:
            st.header("🎯 Quiz Results")
            results_df = st.session_state.quiz_manager.generate_result_dataframe()
            
            if not results_df.empty:
                correct_count = results_df["is_correct"].sum()
                total_questions = len(results_df)
                score_percentage = (correct_count / total_questions) * 100
                
                # Score display
                st.metric("Your Score", f"{score_percentage:.1f}%", f"{correct_count}/{total_questions} correct")
                
                st.markdown("---")
                
                # Question by question breakdown
                for _, result in results_df.iterrows():
                    question_num = result['question_number']
                    if result['is_correct']:
                        st.success(f"✅ Question {question_num}: {result['question']}")
                    else:
                        st.error(f"❌ Question {question_num}: {result['question']}")
                        st.write(f"**Your answer:** {result['user_answer']}")
                        st.write(f"**Correct answer:** {result['correct_answer']}")
                    st.markdown("---")
                
                # Save results
                if st.button("💾 Save Results"):
                    saved_file = st.session_state.quiz_manager.save_to_csv()
                    if saved_file:
                        with open(saved_file, 'rb') as f:
                            st.download_button(
                                "📥 Download Results",
                                data=f.read(),
                                file_name=os.path.basename(saved_file),
                                mime='text/csv'
                            )
                    else:
                        st.warning("No results available to save")


if __name__ == "__main__":
    main()