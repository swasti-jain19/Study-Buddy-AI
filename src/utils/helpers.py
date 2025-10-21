import os
import streamlit as st
import pandas as pd
from src.generator.question_generator import QuestionGenerator


def rerun():
    st.session_state['rerun_trigger'] = not st.session_state.get('rerun_trigger', False)


class QuizManager:
    def __init__(self):
        self.questions = []
        self.user_answers = []
        self.results = []

    def generate_questions(self, generator: QuestionGenerator, topic: str, question_type: str, difficulty: str, num_questions: int):
        self.questions = []
        self.user_answers = []
        self.results = []

        # Reset stored answers
        st.session_state.user_answers = {}

        try:
            for _ in range(num_questions):
                if question_type == "Multiple Choice":
                    question = generator.generate_mcq(topic, difficulty.lower())

                    self.questions.append({
                        'type': 'MCQ',
                        'question': question.question,
                        'options': question.options,
                        'correct_answer': question.correct_answer
                    })

                else:
                    question = generator.generate_fill_blank(topic, difficulty.lower())

                    self.questions.append({
                        'type': 'Fill in the blank',
                        'question': question.question,
                        'correct_answer': question.answer
                    })
        except Exception as e:
            st.error(f"Error generating question: {e}")
            return False

        return True

    def attempt_quiz(self):
        # Initialize session_state for user answers if not exists
        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = {}

        for i, q in enumerate(self.questions):
            st.markdown(f"**Question {i + 1}: {q['question']}**")

            if q['type'] == 'MCQ':
                user_answer = st.radio(
                    f"Select an answer for Question {i + 1}",
                    q['options'],
                    index=None,
                    key=f"mcq_{i}"
                )
            else:
                user_answer = st.text_input(
                    f"Fill in the blank for Question {i + 1}",
                    key=f"fill_blank_{i}"
                )

            # Save user selection persistently
            st.session_state.user_answers[i] = user_answer

    def evaluate_quiz(self):
        self.results = []

        # Use answers from session_state, not transient list
        user_answers = st.session_state.get('user_answers', {})

        for i, q in enumerate(self.questions):
            user_ans = user_answers.get(i)
            correct = q['correct_answer']

            result_dict = {
                'question_number': i + 1,
                'question': q['question'],
                'question_type': q['type'],
                'user_answer': user_ans,
                'correct_answer': correct,
                'is_correct': False
            }

            if q['type'] == 'MCQ':
                result_dict['is_correct'] = (user_ans or "").strip() == correct.strip()
            elif q['type'] == 'Fill in the blank':
                result_dict['is_correct'] = (user_ans or "").strip().lower() == correct.lower()
            else:
                st.warning(f"Unknown question type for Question {i + 1}")

            self.results.append(result_dict)

    def generate_result_dataframe(self):
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)

    def save_to_csv(self, filename_prefix="quiz_results"):
        if not self.results:
            st.warning("No results to save !!")
            return None

        df = self.generate_result_dataframe()

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{filename_prefix}_{timestamp}.csv"

        os.makedirs('results', exist_ok=True)
        full_path = os.path.join('results', unique_filename)

        try:
            df.to_csv(full_path, index=False)
            st.success("Results saved successfully!")
            return full_path
        except Exception as e:
            st.error(f"Failed to save results: {e}")
            return None
