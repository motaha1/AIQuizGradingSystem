
# AutoGrade-AI

AutoGrade-AI is a Flask-based web application that automatically grades student programming submissions using Azure OpenAI. It accepts quiz files, student submissions (in zip/rar format), and optionally custom grading instructions. The system analyzes and grades each student's code against the quiz, then generates real-time feedback and downloadable reports.

## Features

- Upload quizzes and student submissions through a web interface
- Supports `.docx`, `.txt`, `.html`, and `.java` file types
- Optional grading instructions upload (or AI-generated)
- Uses Azure OpenAI (GPT-4o) for grading and instructions
- Real-time progress updates via Server-Sent Events (SSE)
- Grading results downloadable as `.txt` or `.xlsx` reports
- Stores and manages files using Azure Blob Storage
- Designed for grading multiple students in a batch

## Technologies Used

- Python (Flask)
- Azure OpenAI (GPT-4o)
- Azure Blob Storage
- Pandas and OpenPyXL for Excel generation
- Server-Sent Events (SSE) for real-time updates

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/AutoGrade-AI.git
   ```

2. Create and activate a virtual environment:


3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add the following:

   ```env
   AZURE_OPENAI_ENDPOINT=https://your-openai-endpoint/
   AZURE_OPENAI_API_KEY=your-openai-api-key
   AZURE_STORAGE_CONNECTION_STRING=your-azure-blob-connection-string
   AZURE_STORAGE_CONTAINER_NAME=quiz-uploads
   ```

## Usage

1. Start the Flask application:

   ```bash
   python app.py
   ```

2. Open your browser and go to:

   ```
   http://localhost:5000
   ```

3. Upload:
   - A quiz file (`.docx`, `.txt`)
   - A compressed archive of student submissions (`.zip`)
   - (Optional) Grading instructions

4. Monitor grading progress live.

5. Download the results in text or Excel format.

## File Structure

```
app.py               
templates/
    index.html       
.env                 
requirements.txt     
```

