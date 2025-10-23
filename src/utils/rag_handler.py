# src/utils/rag_handler.py
import os
import re
import random
from typing import List, Optional
from difflib import SequenceMatcher
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile


class RAGPipeline:
    def __init__(self, groq_api_key: str):
        """Initialize RAG pipeline with Groq API"""
        self.groq_api_key = groq_api_key
        
        # Initialize embeddings model (local, no API needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Groq LLM with updated model
        self.llm = ChatGroq(
            api_key=self.groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.9  # Increased for more variety
        )
        
        self.vector_store = None
        self.qa_chain = None
        self.retriever = None
        
    def process_pdf(self, uploaded_file) -> str:
        """Process uploaded PDF and create vector store"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return f"✅ Successfully processed {len(splits)} chunks from PDF"
            
        except Exception as e:
            return f"❌ Error processing PDF: {str(e)}"
    
    def create_qa_chain(self, mode: str = "explainer"):
        """Create RAG chain based on chat mode using modern LangChain approach"""
        if not self.vector_store:
            raise ValueError("No vector store available. Please upload a PDF first.")
        
        # Define prompts for different modes
        prompts = {
            "explainer": """You are a helpful study assistant. Use the following context to explain the concept in detail.
If you don't know the answer based on the context, say so clearly.

Context: {context}

Question: {question}

Detailed Explanation:""",
            
            "summarizer": """You are a helpful study assistant. Use the following context to provide a comprehensive summary.
Focus on key points, main ideas, and important concepts. Make it clear and well-structured.

Context: {context}

Question: {question}

Summary:""",
            
            "quizzer": """You are a helpful study assistant. Use the following context to generate quiz questions or verify answers.

Context: {context}

Question: {question}

Response:"""
        }
        
        prompt_template = prompts.get(mode, prompts["explainer"])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create retriever with proper configuration for MORE DIVERSITY
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 15,  # Get 15 diverse chunks
                "fetch_k": 50,  # Search through 50 candidates
                "lambda_mult": 0.2  # Prioritize diversity (0=max diversity, 1=max relevance)
            }
        )

        # Format documents helper function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self.qa_chain
    
    def query(self, question: str, mode: str = "explainer") -> dict:
        """Query the RAG system"""
        if not self.vector_store:
            return {
                "answer": "⚠️ Please upload a PDF document first to enable RAG-based answers.",
                "source_documents": []
            }
        
        try:
            # Create or recreate chain based on mode
            self.create_qa_chain(mode)
            
            # Get response
            answer = self.qa_chain.invoke(question)
            
            # Get source documents - try multiple methods for compatibility
            try:
                # Modern method (LangChain 0.1+)
                source_docs = self.retriever.invoke(question)
            except AttributeError:
                # Fallback for older versions
                source_docs = self.vector_store.similarity_search(question, k=3)
            
            return {
                "answer": answer,
                "source_documents": source_docs
            }
        except Exception as e:
            return {
                "answer": f"❌ Error querying RAG system: {str(e)}",
                "source_documents": []
            }
    
    def generate_structured_quiz(self, num_questions: int, difficulty: str, previous_questions: list = None) -> list:
        """Generate quiz with automatic batching for large quizzes"""
        if not self.vector_store:
            return []
        
        # If requesting more than 12 questions, batch it
        if num_questions > 12:
            print(f"⚡ Batching: Generating {num_questions} questions in multiple batches...")
            
            all_questions = []
            batch_size = 10
            batches = (num_questions + batch_size - 1) // batch_size  # Ceiling division
            
            for batch_num in range(batches):
                questions_needed = min(batch_size, num_questions - len(all_questions))
                print(f"\n📦 Batch {batch_num + 1}/{batches}: Generating {questions_needed} questions...")
                
                # Combine previous questions with already generated ones
                combined_previous = (previous_questions or []) + all_questions
                
                # Generate batch
                batch_questions = self._generate_single_batch(
                    questions_needed, 
                    difficulty, 
                    combined_previous
                )
                
                if batch_questions:
                    # Deduplicate within this batch before adding
                    batch_questions = self._deduplicate_questions(batch_questions, combined_previous)
                    all_questions.extend(batch_questions)
                    print(f"✅ Batch {batch_num + 1} complete: {len(batch_questions)} unique questions added (Total: {len(all_questions)})")
                else:
                    print(f"⚠️ Batch {batch_num + 1} failed to generate questions")
            
            print(f"\n🎯 Final Result: {len(all_questions)}/{num_questions} questions generated\n")
            return all_questions[:num_questions]
        
        else:
            # For 12 or fewer questions, use single generation
            return self._generate_single_batch(num_questions, difficulty, previous_questions)
    
    def _generate_single_batch(self, num_questions: int, difficulty: str, previous_questions: list = None) -> list:
        """Generate a single batch of questions"""
        try:
            random_seed = random.randint(1, 10000)
            
            # Build exclusion text
            exclusion_text = ""
            if previous_questions and len(previous_questions) > 0:
                recent_questions = previous_questions[-15:]
                exclusion_text = "\n\n🚫 MANDATORY EXCLUSION - DO NOT REPEAT:\n"
                for i, pq in enumerate(recent_questions, 1):
                    q_text = pq.get('question', '')[:100]
                    exclusion_text += f"{i}. {q_text}...\n"
                exclusion_text += "\n"
            
            # Enhanced prompt with strict diversity requirements
            prompt = f"""Based STRICTLY on the PDF content, generate exactly {num_questions} COMPLETELY UNIQUE questions at {difficulty} difficulty.

🚨 ULTRA-STRICT ANTI-DUPLICATION REQUIREMENTS:
1. Each question MUST cover a COMPLETELY DIFFERENT topic/concept
2. MANDATORY: Cover different CASE STUDIES from the PDF (e.g., one about EV, one about media, one about airline, etc.)
3. MANDATORY: Mix question types - some about frameworks, some about specific cases, some about calculations
4. DO NOT ask multiple questions about the same case study or framework
5. Questions must be AT LEAST 30-50 pages apart in the original PDF
6. AVOID these overused topics: MECE (already covered), basic interview prep questions

{exclusion_text}

SEED: {random_seed}

MANDATORY DIVERSITY CHECKLIST (must follow):
- Question 1: About a FRAMEWORK (Porter's Five Forces, BCG Matrix, SWOT, etc.)
- Question 2: About a SPECIFIC CASE STUDY from first 100 pages
- Question 3: About a CALCULATION/GUESSTIMATE from the PDF
- Question 4: About INDUSTRY ANALYSIS (pick one: media, aviation, retail, manufacturing, etc.)
- Question 5: About a SPECIFIC CASE STUDY from middle 100 pages
- Question 6: About a METHODOLOGY (Waterfall, Agile, SDLC, etc.)
- Question 7: About a SPECIFIC CASE STUDY from last 100 pages
- Question 8: About MARKET ENTRY strategy from a case
- Question 9: About PRICING/PROFITABILITY from a specific case
- Question 10: About GROWTH STRATEGY from a different case
- Continue this pattern: alternate between frameworks, specific cases, calculations, industry analysis

STRICT FORMAT:
Q1: [Question about Porter's Five Forces OR BCG Matrix OR another framework]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
CORRECT_ANSWER: C

Q2: [Question about SPECIFIC case - e.g., luxury yacht case, waterpark case, etc.]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
CORRECT_ANSWER: B

Now generate {num_questions} MAXIMALLY DIVERSE questions. Each must be about a DIFFERENT case/framework/topic."""

            result = self.query(prompt, mode="quizzer")
            quiz_text = result["answer"]
            
            print(f"=== Quiz Output (Seed: {random_seed}) ===")
            
            # Parse the quiz text
            questions = self._parse_quiz_text(quiz_text)
            
            print(f"Parsed: {len(questions)}/{num_questions} questions\n")
            return questions[:num_questions]
            
        except Exception as e:
            print(f"Error in batch generation: {e}")
            return []
    
    def _parse_quiz_text(self, quiz_text: str) -> list:
        """Parse quiz text into structured questions"""
        questions = []
        question_blocks = re.split(r'Q\d+:', quiz_text)
        
        for i, block in enumerate(question_blocks[1:], 1):
            try:
                lines = [line.strip() for line in block.split('\n') if line.strip()]
                
                question_text = ""
                options = {}
                correct_letter = ""
                
                for line in lines:
                    line_upper = line.upper()
                    
                    if 'CORRECT' in line_upper:
                        if ':' in line:
                            answer_part = line.split(':', 1)[1].strip()
                            match = re.search(r'^([ABCD])\b', answer_part.upper())
                            if match:
                                correct_letter = match.group(1)
                        continue
                    
                    if not question_text and not re.match(r'^[ABCD]\)', line):
                        question_text = line.strip()
                    
                    elif re.match(r'^A\)', line, re.IGNORECASE):
                        options['A'] = re.sub(r'^A\)\s*', '', line, flags=re.IGNORECASE).strip()
                    elif re.match(r'^B\)', line, re.IGNORECASE):
                        options['B'] = re.sub(r'^B\)\s*', '', line, flags=re.IGNORECASE).strip()
                    elif re.match(r'^C\)', line, re.IGNORECASE):
                        options['C'] = re.sub(r'^C\)\s*', '', line, flags=re.IGNORECASE).strip()
                    elif re.match(r'^D\)', line, re.IGNORECASE):
                        options['D'] = re.sub(r'^D\)\s*', '', line, flags=re.IGNORECASE).strip()
                
                if question_text and len(options) == 4 and correct_letter in ['A', 'B', 'C', 'D']:
                    questions.append({
                        "question": question_text,
                        "options": [f"{k}) {v}" for k, v in options.items()],
                        "correct_answer": f"{correct_letter}) {options[correct_letter]}"
                    })
            
            except Exception as e:
                continue
        
        return questions
    
    def _deduplicate_questions(self, new_questions: list, existing_questions: list) -> list:
        """Remove duplicates using fuzzy matching + keyword overlap detection"""
        
        def similarity(q1: str, q2: str) -> float:
            """Calculate similarity between two questions (0-1)"""
            return SequenceMatcher(None, q1.lower(), q2.lower()).ratio()
        
        def extract_key_phrases(question: str) -> set:
            """Extract important phrases from question"""
            stop_words = {'what', 'is', 'the', 'a', 'an', 'in', 'of', 'for', 'to', 'and', 'or', 
                          'according', 'mentioned', 'pdf', 'document', 'context', 'used', 'purpose', 
                          'primary', 'main', 'key', 'important', 'following'}
            
            words = question.lower().split()
            phrases = set()
            
            # Single important words (5+ chars)
            for word in words:
                cleaned = word.strip('?,.')
                if len(cleaned) > 4 and cleaned not in stop_words:
                    phrases.add(cleaned)
            
            # Two-word phrases
            for i in range(len(words) - 1):
                w1, w2 = words[i].strip('?,.'), words[i+1].strip('?,.')
                if len(w1) > 3 and len(w2) > 3:
                    phrase = f"{w1} {w2}"
                    if len(phrase) > 8:
                        phrases.add(phrase)
            
            # Three-word phrases
            for i in range(len(words) - 2):
                w1 = words[i].strip('?,.')
                w2 = words[i+1].strip('?,.')
                w3 = words[i+2].strip('?,.')
                if len(w1) > 3 and len(w2) > 3:
                    phrase = f"{w1} {w2} {w3}"
                    if len(phrase) > 12:
                        phrases.add(phrase)
            
            return phrases
        
        def keyword_overlap(q1: str, q2: str) -> float:
            """Calculate keyword overlap between two questions"""
            phrases1 = extract_key_phrases(q1)
            phrases2 = extract_key_phrases(q2)
            
            if not phrases1 or not phrases2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(phrases1 & phrases2)
            union = len(phrases1 | phrases2)
            
            return intersection / union if union > 0 else 0.0
        
        unique_questions = []
        duplicates_removed = 0
        
        for new_q in new_questions:
            is_duplicate = False
            new_text = new_q['question']
            
            # Check against existing questions
            all_to_check = existing_questions + unique_questions
            
            for existing_q in all_to_check:
                existing_text = existing_q.get('question', '')
                
                # Calculate both similarity metrics
                text_sim = similarity(new_text, existing_text)
                keyword_sim = keyword_overlap(new_text, existing_text)
                
                # Duplicate if:
                # 1. Text similarity > 70% OR
                # 2. Keyword overlap > 50%
                if text_sim > 0.70 or keyword_sim > 0.50:
                    print(f"   🗑️  Removing duplicate (text: {text_sim:.2f}, keywords: {keyword_sim:.2f})")
                    print(f"       {new_text[:60]}...")
                    is_duplicate = True
                    duplicates_removed += 1
                    break
            
            if not is_duplicate:
                unique_questions.append(new_q)
        
        if duplicates_removed > 0:
            print(f"   ✂️  Total duplicates removed from batch: {duplicates_removed}")
        
        return unique_questions
    
    def save_vector_store(self, path: str = "vector_store"):
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            return f"✅ Vector store saved to {path}"
        return "⚠️ No vector store to save"
    
    def load_vector_store(self, path: str = "vector_store"):
        """Load vector store from disk"""
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return "✅ Vector store loaded successfully"
        except Exception as e:
            return f"❌ Error loading vector store: {str(e)}"