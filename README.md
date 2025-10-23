# Study Buddy AI 🎓

An intelligent quiz generation and study assistance system powered by LLM (Large Language Models) with RAG (Retrieval Augmented Generation) capabilities.

## Features

- 📝 Interactive Quiz Generation
  - Multiple Choice Questions (MCQ)
  - Fill in the Blank Questions
  - Customizable difficulty levels
  - Automatic scoring and feedback

- 📚 PDF Study Mode
  - Upload and analyze PDF documents
  - Generate questions from PDF content
  - Ask questions about the material
  - Get comprehensive summaries

- 💡 Smart Features
  - RAG-powered responses
  - Automatic duplicate question detection
  - Progress tracking
  - Downloadable quiz results

## Installation

1. Clone the repository:
```sh
git clone <repository-url>
cd study-buddy-ai
```

2. Install dependencies:
```sh
pip install -e .
```

3. Set up environment variables:
Create a `.env` file with:
```
GROQ_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```sh
streamlit run application.py
```

2. Choose your study mode:
   - Regular Quiz Mode
   - PDF Study Mode

3. For PDF Mode:
   - Upload your study material
   - Select study options (Quiz/Summary/Q&A)
   - Get AI-powered assistance

## Project Structure

- `src`: Source code modules
  - `generator/`: Question generation logic
  - `llm/`: LLM integration
  - `models/`: Data models
  - `prompts/`: LLM prompt templates
  - `utils/`: Helper functions
- `results`: Quiz results storage
- `logs`: Application logs

## Requirements

Main dependencies:
- Python 3.7+
- Streamlit
- LangChain
- GROQ LLM
- FAISS for vector storage
- HuggingFace Embeddings

For full list, see `requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License

## Author

Swasti Jain