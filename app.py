import os
import zipfile
import rarfile
import tempfile
import shutil
import json
import pandas as pd
import re
from datetime import datetime
from flask import Flask, request, jsonify, Response, render_template
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from docx import Document
from openai import AzureOpenAI
import threading
import time
import queue
from dotenv import load_dotenv
from io import BytesIO
from azure.storage.blob import BlobServiceClient, BlobClient
import uuid

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production

# Azure OpenAI Configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = "gpt-4.1"
subscription_key = os.getenv('AZURE_OPENAI_API_KEY')
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Azure Blob Storage Configuration
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "quiz-uploads")

# Initialize Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Ensure container exists
try:
    container_client = blob_service_client.get_container_client(container_name)
    container_client.get_container_properties()
except Exception:
    # Container doesn't exist, create it
    blob_service_client.create_container(container_name)

# Global variables for real-time updates
grading_progress = {
    'current_student': '',
    'total_students': 0,
    'processed_students': 0,
    'results': [],
    'status': 'idle',
    'error': None,
    'student_grades': []  # For Excel export
}

# Flag to stop grading process
stop_grading_flag = False

# Flag to prevent multiple grading processes
grading_in_progress = False

# SSE event queue for real-time updates
sse_clients = []


def broadcast_update(data):
    """Broadcast update to all SSE clients"""
    dead_clients = []
    for client_queue in sse_clients:
        try:
            client_queue.put(data, timeout=1)
        except queue.Full:
            dead_clients.append(client_queue)

    # Remove dead clients
    for dead_client in dead_clients:
        sse_clients.remove(dead_client)


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def upload_to_blob(file_stream, blob_name):
    """Upload file stream to Azure Blob Storage"""
    try:
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        file_stream.seek(0)
        blob_client.upload_blob(file_stream, overwrite=True)
        return blob_name
    except Exception as e:
        raise Exception(f"Failed to upload to blob storage: {str(e)}")


def download_from_blob(blob_name):
    """Download file from Azure Blob Storage to memory"""
    try:
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        blob_data = blob_client.download_blob()
        return BytesIO(blob_data.readall())
    except Exception as e:
        raise Exception(f"Failed to download from blob storage: {str(e)}")


def delete_blob(blob_name):
    """Delete blob from Azure Blob Storage"""
    try:
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        blob_client.delete_blob()
    except Exception:
        pass  # Ignore errors when deleting


def read_quiz_file_from_blob(blob_name, file_extension):
    """Read quiz file from blob storage and extract text content"""
    try:
        file_stream = download_from_blob(blob_name)

        if file_extension == '.docx':
            return read_docx_from_stream(file_stream)
        elif file_extension == '.txt':
            file_stream.seek(0)
            return file_stream.read().decode('utf-8', errors='ignore')
        else:
            # For other file types, try to read as text
            file_stream.seek(0)
            return file_stream.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Error reading quiz file: {str(e)}"


def read_docx_from_stream(file_stream):
    """Read docx file directly from file stream"""
    try:
        # Create temporary file in memory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, 'temp_docx.docx')

        try:
            # Write the stream content to the temporary file
            with open(temp_file_path, 'wb') as temp_file:
                file_stream.seek(0)
                temp_file.write(file_stream.read())

            # Read the docx content
            doc = Document(temp_file_path)
            content = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            return content

        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        return f"Error reading docx file: {str(e)}"


def read_code_file(file_content, file_extension):
    """Read code file content and extract text"""
    try:
        if file_extension == '.html':
            soup = BeautifulSoup(file_content, 'html.parser')
            return soup.get_text()
        return file_content
    except Exception as e:
        return f"Error reading code file: {str(e)}"


def extract_archive_from_blob(blob_name):
    """Extract zip or rar file from blob storage and return files structure"""
    try:
        # Download archive from blob
        archive_stream = download_from_blob(blob_name)

        # Create temporary directory for extraction
        temp_dir = tempfile.mkdtemp()
        archive_path = os.path.join(temp_dir, "archive.zip")

        try:
            # Save archive to temporary file
            with open(archive_path, 'wb') as f:
                archive_stream.seek(0)
                f.write(archive_stream.read())

            # Extract archive
            file_extension = blob_name.lower().split('.')[-1]
            extract_to = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_to, exist_ok=True)

            if file_extension == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as archive_ref:
                    archive_ref.extractall(extract_to)
            elif file_extension == 'rar':
                with rarfile.RarFile(archive_path, 'r') as archive_ref:
                    archive_ref.extractall(extract_to)
            else:
                raise Exception(f"Unsupported archive format: {file_extension}")

            # Collect student files structure
            student_files = {}
            for root, dirs, files in os.walk(extract_to):
                for file in files:
                    if file.endswith(('.java', '.html', '.txt')):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, extract_to)

                        # Extract student name from path
                        path_parts = relative_path.split(os.sep)
                        student_name = path_parts[0] if len(path_parts) > 0 else "unknown"

                        if student_name not in student_files:
                            student_files[student_name] = []

                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        student_files[student_name].append({
                            'filename': file,
                            'content': content,
                            'extension': os.path.splitext(file)[1]
                        })

            return student_files

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        raise Exception(f"Error extracting archive from blob: {str(e)}")


def get_grading_distribution(quiz_content, total_mark=10, temperature=0.2):
    """Get grading distribution from GPT"""
    prompt = f"""You are an instructor. Given the following quiz, create a detailed and fair grading distribution out of {total_mark} points. Please break down how marks should be assigned to each part of the answer.
The grading should be additive only — no deductions and no bonus points.
Clearly list how each mark is earned for every part of the answer.

For each part, break it down as follows:

State the total mark for the part.

Break the mark into specific steps.
Each step should specify what is required to earn the corresponding fraction of the mark.

Quiz:
{quiz_content}

Respond with a clear, detailed mark distribution from {total_mark} in bullet points."""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=temperature,
            top_p=1.0,
        )
        time.sleep(10)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Failed to get distribution: {str(e)}"


def extract_grade_from_response(response_text, total_mark=10):
    """Extract numerical grade from GPT response"""
    patterns = [
        rf'(?:grade|score|mark|total)[:\s]*(\d+(?:\.\d+)?)[/\s]*(?:out of\s*)?{total_mark}',
        rf'(\d+(?:\.\d+)?)[/\s]*{total_mark}',
        rf'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*{total_mark}',
        r'final[:\s]*(\d+(?:\.\d+)?)',
        r'total[:\s]*(\d+(?:\.\d+)?)'
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            try:
                grade = float(match.group(1))
                if 0 <= grade <= total_mark:
                    return grade
            except ValueError:
                continue

    return None


def grade_student_code(quiz_content, student_code, grading_instructions, student_name, file_name, total_mark=10, temperature=0.1, grading_rules=None):
    """Grade student code using GPT"""
    
    # Default grading rules if none provided
    if grading_rules is None or len(grading_rules) == 0:
        grading_rules = [
            "Include a short explanation based on the distribution",
            "Be fair",
            "recheck the code again and evaluate it (if you check twice print \"✅I check it twice\")",
            "give a partial mark if needed (not only 0 or full mark)",
            "Be very very Tolerant",
            "Give marks for the attempt if some things are correct.",
            "Don't focus too much on the function name If there is not a very big difference.",
            "Don't focus too much on format of output."
        ]
    
    # Format grading rules as bullet points
    rules_text = "\n".join([f"-{rule}" for rule in grading_rules])
    
    prompt = f"""Grade the following student code based on this quiz question.

Use this grading distribution:
{grading_instructions}

Each grade should:
- Be scored out of **{total_mark}**
- Be clearly labeled with "Grade: X/{total_mark}" or "Score: X/{total_mark}" at the end
{rules_text}

Question:
{quiz_content}

Student Code:
{student_code}

Please end your response with a clear final grade in the format "Final Grade: X/{total_mark}" """

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0, #new
            presence_penalty=0, #new
        )
        #print(f"🔥 API RESPONSE ID: {response.id} - Student: {student_name}, File: {file_name}")
        time.sleep(2)
        
        # Return both content and response ID
        return {
            'content': response.choices[0].message.content.strip(),
            'response_id': response.id
        }
    except Exception as e:
        print(f"❌ API CALL FAILED - Student: {student_name}, File: {file_name}, Error: {str(e)}")
        return {
            'content': f"Grading failed: {str(e)}",
            'response_id': None
        }


def create_excel_data(student_grades):
    """Create Excel data in memory"""
    if not student_grades:
        return None

    # Create DataFrame
    df = pd.DataFrame(student_grades)

    # Sort by student name
    df = df.sort_values('Student Name')

    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write main sheet
        df.to_excel(writer, sheet_name='Grades', index=False)

        # Add summary statistics
        total_mark = df['Total Mark'].iloc[0] if len(df) > 0 and 'Total Mark' in df.columns else 10
        pass_threshold = total_mark * 0.6
        
        summary_data = {
            'Statistic': ['Total Students', 'Average Grade', 'Highest Grade', 'Lowest Grade', f'Pass Rate (≥{pass_threshold}/{total_mark})'],
            'Value': [
                len(df),
                f"{df['Grade'].mean():.2f}/{total_mark}" if len(df) > 0 else "N/A",
                f"{df['Grade'].max():.1f}/{total_mark}" if len(df) > 0 else "N/A",
                f"{df['Grade'].min():.1f}/{total_mark}" if len(df) > 0 else "N/A",
                f"{(df['Grade'] >= pass_threshold).sum()}/{len(df)} ({(df['Grade'] >= pass_threshold).mean() * 100:.1f}%)" if len(
                    df) > 0 else "N/A"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    output.seek(0)
    return output


def process_grading_task(quiz_blob_name, quiz_extension, archive_blob_name, instructions_blob_name=None,
                         instructions_extension=None, temp_instructions=0.2, temp_grading=0.1, total_mark=10, model='gpt-4o', grading_rules=None):
    """Background task to process all student submissions with SSE updates"""
    global grading_progress, deployment, stop_grading_flag, grading_in_progress

    # Check if grading is already in progress
    if grading_in_progress:
        print("WARNING: Grading already in progress, skipping duplicate call")
        return

    # Set grading in progress flag
    grading_in_progress = True
    print(f"DEBUG: Starting grading process for session")

    # Set the deployment model
    deployment = model
    
    # Reset stop flag
    stop_grading_flag = False

    try:
        # Clear any existing data to prevent duplicates
        grading_progress['status'] = 'processing'
        grading_progress['results'] = []
        grading_progress['student_grades'] = []
        grading_progress['error'] = None
        grading_progress['current_student'] = ''
        grading_progress['total_students'] = 0
        grading_progress['processed_students'] = 0

        # Broadcast status update
        broadcast_update({
            'type': 'status',
            'status': 'processing',
            'message': 'Starting grading process...'
        })

        # Read quiz content from blob
        broadcast_update({
            'type': 'status',
            'status': 'processing',
            'message': 'Reading quiz content...'
        })
        quiz_content = read_quiz_file_from_blob(quiz_blob_name, quiz_extension)

        # Get or generate grading instructions
        if instructions_blob_name:
            broadcast_update({
                'type': 'status',
                'status': 'processing',
                'message': 'Reading grading instructions...'
            })
            grading_instructions = read_quiz_file_from_blob(instructions_blob_name, instructions_extension)
        else:
            broadcast_update({
                'type': 'status',
                'status': 'processing',
                'message': 'Generating grading instructions...'
            })
            grading_progress['current_student'] = 'Generating grading instructions...'
            grading_instructions = get_grading_distribution(quiz_content, total_mark, temp_instructions)

        # Add grading instructions to results and broadcast
        instruction_result = {
            'type': 'instructions',
            'content': grading_instructions,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        grading_progress['results'].append(instruction_result)
        broadcast_update({
            'type': 'result',
            'data': instruction_result
        })

        # Extract and process submissions from blob
        broadcast_update({
            'type': 'status',
            'status': 'processing',
            'message': 'Extracting student submissions...'
        })

        student_files = extract_archive_from_blob(archive_blob_name)
        grading_progress['total_students'] = len(student_files)

        # Debug: Log student structure
        print(f"DEBUG: Found {len(student_files)} students:")
        for student_name, files in student_files.items():
            print(f"  {student_name}: {len(files)} files - {[f['filename'] for f in files]}")

        # Broadcast total students count
        broadcast_update({
            'type': 'progress',
            'total_students': len(student_files),
            'processed_students': 0
        })

        student_list = list(student_files.keys())
        print(f"DEBUG: Processing students in order: {student_list}")

        for i, student_name in enumerate(student_list):
            # Check if grading should be stopped
            if stop_grading_flag:
                grading_progress['status'] = 'idle'
                grading_progress['error'] = 'Grading stopped by user'
                broadcast_update({
                    'type': 'status',
                    'status': 'idle',
                    'message': 'Grading stopped by user'
                })
                return
                
            grading_progress['current_student'] = student_name

            # Broadcast current progress BEFORE processing
            broadcast_update({
                'type': 'progress',
                'current_student': student_name,
                'processed_students': i,
                'total_students': len(student_list)
            })

            student_total_grade = 0
            file_count = 0
            all_grades = []  # Track all grades for this student

            # Process each code file for the student
            for file_info in student_files[student_name]:
                # Check if grading should be stopped
                if stop_grading_flag:
                    return
                    
                file_name = file_info['filename']
                file_content = file_info['content']
                file_extension = file_info['extension']

                # Process the code content
                processed_content = read_code_file(file_content, file_extension)

                # Grade the code
                grade_result = grade_student_code(quiz_content, processed_content, grading_instructions,
                                                  student_name, file_name, total_mark, temp_grading, grading_rules)

                # Extract grade for Excel (handle new return format)
                grade_content = grade_result.get('content', grade_result) if isinstance(grade_result, dict) else grade_result
                response_id = grade_result.get('response_id') if isinstance(grade_result, dict) else None
                
                extracted_grade = extract_grade_from_response(grade_content, total_mark)
                if extracted_grade is not None:
                    student_total_grade += extracted_grade
                    file_count += 1
                    all_grades.append(extracted_grade)

                # Create result object with response ID
                result = {
                    'type': 'grade',
                    'student': student_name,
                    'file': file_name,
                    'content': grade_content,
                    'response_id': response_id,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }

                # Add to progress and broadcast
                grading_progress['results'].append(result)
                broadcast_update({
                    'type': 'result',
                    'data': result
                })

            # Add student to Excel data only once per student (after processing all their files)
            if file_count > 0:
                final_grade = student_total_grade / file_count
                
                # Check if student already exists in student_grades (safety check)
                existing_student = next((s for s in grading_progress['student_grades'] if s['Student Name'] == student_name), None)
                if existing_student is None:
                    grading_progress['student_grades'].append({
                        'Student Name': student_name,
                        'Grade': round(final_grade, 2),
                        'Total Mark': total_mark,
                        'Files Processed': file_count,
                        'Individual Grades': ', '.join([str(g) for g in all_grades]) if len(all_grades) > 1 else str(all_grades[0]) if all_grades else 'N/A',
                        'Status': 'Pass' if final_grade >= (total_mark * 0.6) else 'Fail'  # 60% pass rate
                    })
                else:
                    # Log warning if duplicate found
                    print(f"Warning: Duplicate student '{student_name}' found, skipping...")

            # Update processed count AFTER processing each student
            grading_progress['processed_students'] = i + 1

            # Broadcast updated progress
            broadcast_update({
                'type': 'progress',
                'current_student': student_name,
                'processed_students': i + 1,
                'total_students': len(student_list)
            })

        grading_progress['status'] = 'completed'
        grading_progress['current_student'] = 'Grading completed!'

        # Clean up blobs
        delete_blob(quiz_blob_name)
        delete_blob(archive_blob_name)
        if instructions_blob_name:
            delete_blob(instructions_blob_name)

        # Broadcast completion
        broadcast_update({
            'type': 'status',
            'status': 'completed',
            'message': 'Grading completed successfully!'
        })
        broadcast_update({
            'type': 'progress',
            'current_student': 'Grading completed!',
            'processed_students': len(student_list),
            'total_students': len(student_list)
        })

        print(f"DEBUG: Grading process completed successfully")

    except Exception as e:
        grading_progress['status'] = 'error'
        grading_progress['error'] = str(e)

        print(f"DEBUG: Grading process failed with error: {str(e)}")

        # Clean up blobs on error
        try:
            delete_blob(quiz_blob_name)
            delete_blob(archive_blob_name)
            if instructions_blob_name:
                delete_blob(instructions_blob_name)
        except:
            pass

        # Broadcast error
        broadcast_update({
            'type': 'status',
            'status': 'error',
            'message': str(e)
        })
    
    finally:
        # Always reset the grading in progress flag
        grading_in_progress = False
        print(f"DEBUG: Grading process ended, flag reset")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    global grading_progress, stop_grading_flag, grading_in_progress

    try:
        print(f"DEBUG: Upload request received")
        
        # Check if grading is already in progress
        if grading_in_progress:
            return jsonify({'error': 'Grading process already in progress'}), 400

        # Reset progress and stop flag
        stop_grading_flag = False
        grading_progress = {
            'current_student': '',
            'total_students': 0,
            'processed_students': 0,
            'results': [],
            'student_grades': [],
            'status': 'idle',
            'error': None
        }

        print(f"DEBUG: Progress reset, checking files")

        # Check if files are present
        if 'quiz_file' not in request.files or 'archive_file' not in request.files:
            return jsonify({'error': 'Missing required files'}), 400

        quiz_file = request.files['quiz_file']
        archive_file = request.files['archive_file']
        instruction_file = request.files.get('instruction_file')

        if quiz_file.filename == '' or archive_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # Validate file types
        if not allowed_file(quiz_file.filename, {'pdf', 'doc', 'docx', 'txt'}):
            return jsonify({'error': 'Invalid quiz file type'}), 400

        if not allowed_file(archive_file.filename, {'zip', 'rar', '7z'}):
            return jsonify({'error': 'Invalid archive file type'}), 400

        # Generate unique blob names
        session_id = str(uuid.uuid4())
        quiz_extension = os.path.splitext(quiz_file.filename)[1]
        archive_extension = os.path.splitext(archive_file.filename)[1]

        quiz_blob_name = f"{session_id}/quiz{quiz_extension}"
        archive_blob_name = f"{session_id}/archive{archive_extension}"

        # Upload files to blob storage
        upload_to_blob(quiz_file.stream, quiz_blob_name)
        upload_to_blob(archive_file.stream, archive_blob_name)

        # Process instruction file if provided
        instructions_blob_name = None
        instructions_extension = None
        if instruction_file and instruction_file.filename != '':
            if allowed_file(instruction_file.filename, {'pdf', 'doc', 'docx', 'txt'}):
                instructions_extension = os.path.splitext(instruction_file.filename)[1]
                instructions_blob_name = f"{session_id}/instructions{instructions_extension}"
                upload_to_blob(instruction_file.stream, instructions_blob_name)

        # Get temperature and model parameters from form
        temp_instructions = float(request.form.get('temp_instructions', 0.2))
        temp_grading = float(request.form.get('temp_grading', 0.1))
        total_mark = int(request.form.get('total_mark', 10))
        model = request.form.get('model', 'gpt-4o')
        
        # Get grading rules from form (JSON string)
        grading_rules_json = request.form.get('grading_rules', '[]')
        try:
            grading_rules = json.loads(grading_rules_json) if grading_rules_json else []
        except json.JSONDecodeError:
            grading_rules = []

        # Validate temperature values
        temp_instructions = max(0.0, min(1.0, temp_instructions))
        temp_grading = max(0.0, min(1.0, temp_grading))
        
        # Validate total mark
        total_mark = max(1, min(100, total_mark))

        # Validate model
        if model not in ['gpt-4o', 'gpt-4.1']:
            model = 'gpt-4o'

        # Start background processing
        print(f"DEBUG: Starting background thread for grading")
        thread = threading.Thread(
            target=process_grading_task,
            args=(quiz_blob_name, quiz_extension, archive_blob_name, instructions_blob_name, instructions_extension, temp_instructions, temp_grading, total_mark, model, grading_rules)
        )
        thread.daemon = True
        thread.start()
        print(f"DEBUG: Background thread started")

        return jsonify({'message': 'Files uploaded successfully, grading started'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_grading():
    """Stop the grading process"""
    global stop_grading_flag, grading_progress, grading_in_progress
    
    print(f"DEBUG: Stop grading requested")
    stop_grading_flag = True
    grading_progress['status'] = 'idle'
    grading_progress['error'] = 'Grading stopped by user'
    
    # Reset the grading in progress flag
    grading_in_progress = False
    
    # Broadcast stop message to all SSE clients
    broadcast_update({
        'type': 'status',
        'status': 'idle',
        'message': 'Grading stopped by user'
    })
    
    return jsonify({'message': 'Grading process stopped'}), 200


@app.route('/events')
def events():
    """SSE endpoint for real-time updates"""

    def event_stream():
        # Create a queue for this client
        client_queue = queue.Queue(maxsize=50)
        sse_clients.append(client_queue)

        try:
            # Send initial state
            yield f"data: {json.dumps({'type': 'status', 'status': grading_progress['status'], 'message': 'Connected'})}\n\n"

            # Send existing results if any
            for result in grading_progress['results']:
                yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"

            # Send current progress
            if grading_progress['total_students'] > 0:
                yield f"data: {json.dumps({'type': 'progress', 'current_student': grading_progress['current_student'], 'processed_students': grading_progress['processed_students'], 'total_students': grading_progress['total_students']})}\n\n"

            # Listen for new events
            while True:
                try:
                    # Wait for new events with timeout
                    event_data = client_queue.get(timeout=30)
                    yield f"data: {json.dumps(event_data)}\n\n"
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

        except Exception as e:
            print(f"SSE client disconnected: {e}")
        finally:
            # Remove client from list when disconnected
            if client_queue in sse_clients:
                sse_clients.remove(client_queue)

    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )


@app.route('/download/text')
def download_text():
    """Download results as text file"""
    try:
        if not grading_progress['results']:
            return jsonify({'error': 'No results available'}), 400

        # Generate text content
        text_content = "Quiz Grading Results\n"
        text_content += "=" * 50 + "\n\n"

        for result in grading_progress['results']:
            if result['type'] == 'instructions':
                text_content += f"Grading Instructions:\n{result['content']}\n\n"
                text_content += "=" * 50 + "\n\n"
            elif result['type'] == 'grade':
                text_content += f"Student: {result['student']}, File: {result['file']}\n"
                text_content += f"Timestamp: {result['timestamp']}\n"
                text_content += f"Grade Result:\n{result['content']}\n"
                text_content += "-" * 50 + "\n\n"

        return Response(
            text_content,
            mimetype='text/plain',
            headers={
                'Content-Disposition': f'attachment; filename=grading_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/excel')
def download_excel():
    """Download results as Excel file"""
    try:
        if not grading_progress['student_grades']:
            return jsonify({'error': 'No grades available for Excel export'}), 400

        excel_data = create_excel_data(grading_progress['student_grades'])
        if excel_data is None:
            return jsonify({'error': 'Failed to create Excel file'}), 500

        return Response(
            excel_data.getvalue(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                'Content-Disposition': f'attachment; filename=student_grades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Keep the old endpoints for backward compatibility
@app.route('/progress')
def get_progress():
    return jsonify(grading_progress)


@app.route('/results')
def get_results():
    return jsonify(grading_progress['results'])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)