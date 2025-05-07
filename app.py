# app.py
import os
import uuid
import json # For saving and loading prompts
import flask
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import torch
import whisper # Using openai-whisper
import torchaudio
import google.generativeai as genai
import requests # For sending requests to Discord webhook

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
PROMPTS_FILE = 'prompts.json' # File to store custom prompts
API_KEY_FILE = 'gemini_api_key.txt' # File to store the Gemini API Key

MAX_CONTENT_LENGTH_MB = 500 
FLASK_MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH_MB * 1024 * 1024
WHISPER_MODEL_SIZE = "base"

# --- Load Gemini API Key ---
# Priority:
# 1. Environment variable (GEMINI_API_KEY)
# 2. Local file (gemini_api_key.txt)
# 3. Placeholder
def load_gemini_api_key():
    """Loads the Gemini API key from environment variable or local file."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print(f"Loaded Gemini API key from environment variable.")
        return api_key
    
    if os.path.exists(API_KEY_FILE):
        try:
            with open(API_KEY_FILE, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if api_key:
                print(f"Loaded Gemini API key from {API_KEY_FILE}.")
                return api_key
            else:
                print(f"Warning: {API_KEY_FILE} is empty.")
        except IOError as e:
            print(f"Warning: Could not read {API_KEY_FILE}: {e}")
    
    print(f"Warning: Gemini API key not found in environment variable or {API_KEY_FILE}.")
    print("Please set the GEMINI_API_KEY environment variable or create a file named")
    print(f"{API_KEY_FILE} in the application directory containing your API key.")
    print("Using placeholder key 'YOUR_GEMINI_API_KEY' - Summarization will likely fail.")
    return "YOUR_GEMINI_API_KEY" # Fallback placeholder

GEMINI_API_KEY = load_gemini_api_key()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = FLASK_MAX_CONTENT_LENGTH
app.secret_key = 'super secret key for flash messages' # Change this in production

# --- Device Selection for Whisper ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    if hasattr(torch.version, 'hip') and torch.cuda.is_available():
        print("ROCm (AMD GPU) detected. Using device: cuda for Whisper.")
    elif torch.cuda.is_available():
        print("CUDA (NVIDIA GPU) detected. Using device: cuda for Whisper.")
else:
    print("No GPU detected or ROCm/CUDA not available. Using device: cpu for Whisper.")

# --- Prompt Management Functions ---
def load_prompts():
    """Loads prompts from the JSON file."""
    if not os.path.exists(PROMPTS_FILE):
        # Create a default prompts file if it doesn't exist
        default_prompts = [
            {"name": "Default Concise Summary", "text": "Please provide a concise summary of the following audio transcript:\n\nTranscript:\n{transcript}\n\nSummary:"},
            {"name": "Key Action Items", "text": "From the following transcript, please extract the key action items, decisions made, and responsible individuals if mentioned. Format as a list:\n\nTranscript:\n{transcript}\n\nKey Points:"},
            {"name": "Main Topics Discussed", "text": "What are the main topics discussed in the following transcript? List them clearly.\n\nTranscript:\n{transcript}\n\nMain Topics:"}
        ]
        save_prompts(default_prompts)
        return default_prompts
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading prompts: {e}. Returning empty list.")
        return []

def save_prompts(prompts):
    """Saves prompts to the JSON file."""
    try:
        with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=4)
    except IOError as e:
        print(f"Error saving prompts: {e}")

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    """Ensures the upload folder exists."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

# --- Transcription Function ---
def transcribe_audio(audio_path):
    """
    Transcribes the audio file using Whisper.
    """
    print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on {DEVICE}...")
    try:
        model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
        print("Whisper model loaded.")
        print(f"Starting transcription for: {audio_path}")
        result = model.transcribe(audio_path, fp16=(DEVICE == "cuda"))
        print("Transcription complete.")
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

# --- Summarization Function (using Gemini) ---
def summarize_with_gemini(transcript_text, full_prompt_text):
    """
    Summarizes the given transcript using the provided full prompt text with Gemini.
    The full_prompt_text should include a placeholder like {transcript}
    which will be replaced by the actual transcript_text.
    """
    print("Attempting to summarize with Gemini...")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY": # Check against the placeholder
        print("GEMINI_API_KEY not configured correctly. Please check environment variable or gemini_api_key.txt.")
        return "Gemini summarization is not configured. Please set your API key."

    if "{transcript}" not in full_prompt_text:
        # Fallback if the placeholder is missing, though UI should guide user
        print("Warning: '{transcript}' placeholder missing in custom prompt. Appending transcript.")
        final_prompt = f"{full_prompt_text}\n\nTranscript:\n{transcript_text}\n\nSummary:"
    else:
        final_prompt = full_prompt_text.replace("{transcript}", transcript_text)
    
    print(f"Final prompt for Gemini:\n{final_prompt[:300]}...") # Log start of prompt

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using "gemini-1.5-pro-latest" as it's a more recent and generally well-supported identifier.
        model_name = "gemini-1.5-pro-latest" 
        print(f"Using Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        print("Sending request to Gemini API...")
        response = model.generate_content(final_prompt)
        print("Received response from Gemini API.")
        
        if response and hasattr(response, 'text') and response.text:
            return response.text
        elif response and hasattr(response, 'parts'):
            all_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            if all_text:
                return all_text
            else:
                print(f"Gemini API response parts did not contain text. Response: {response}")
                return "Gemini API returned a response, but it contained no text."
        else:
            print(f"Gemini API returned an unexpected response structure: {response}")
            return "Gemini API returned an empty or invalid response."

    except Exception as e:
        print(f"Error during Gemini summarization: {e}")
        return f"An error occurred while trying to summarize with Gemini: {str(e)}"

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Serves the main upload page and loads prompts."""
    ensure_upload_folder() 
    prompts = load_prompts()
    return render_template('index.html', max_upload_mb=MAX_CONTENT_LENGTH_MB, saved_prompts=prompts)

@app.route('/transcribe', methods=['POST'])
def transcribe_and_summarize_route():
    """Handles file upload, transcription, and summarization with custom prompts."""
    ensure_upload_folder()

    if 'audioFile' not in request.files:
        flash('No file part in the request. Please select a file.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['audioFile']
    if file.filename == '':
        flash('No file selected. Please choose an audio file to upload.', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = str(uuid.uuid4()) + "." + file_extension
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        original_filename = file.filename

        selected_prompt_name = request.form.get('saved_prompt_selection')
        custom_prompt_text_input = request.form.get('custom_prompt_text', '').strip()
        save_new_prompt_flag = request.form.get('save_new_prompt_checkbox')
        new_prompt_name_input = request.form.get('new_prompt_name', 'My Custom Prompt').strip()
        
        final_summary_prompt_text = ""
        prompts = load_prompts()

        if custom_prompt_text_input:
            final_summary_prompt_text = custom_prompt_text_input
            if save_new_prompt_flag and new_prompt_name_input:
                if any(p['name'] == new_prompt_name_input for p in prompts):
                    flash(f"A prompt with the name '{new_prompt_name_input}' already exists. Not saving.", 'warning')
                else:
                    prompts.append({"name": new_prompt_name_input, "text": custom_prompt_text_input})
                    save_prompts(prompts)
                    flash(f"New prompt '{new_prompt_name_input}' saved successfully!", 'success')
        elif selected_prompt_name:
            selected_prompt_obj = next((p for p in prompts if p['name'] == selected_prompt_name), None)
            if selected_prompt_obj:
                final_summary_prompt_text = selected_prompt_obj['text']
            else:
                flash(f"Selected saved prompt '{selected_prompt_name}' not found. Using default.", 'error')
                final_summary_prompt_text = prompts[0]['text'] if prompts else "Summarize this: {transcript}"
        else:
            flash("No prompt selected or entered. Using a default summary prompt.", 'warning')
            final_summary_prompt_text = prompts[0]['text'] if prompts else "Please summarize the following: {transcript}"
        
        if "{transcript}" not in final_summary_prompt_text:
            flash("The selected or entered prompt is missing the '{transcript}' placeholder. Please edit the prompt to include it.", "error")
            return redirect(url_for('index'))

        try:
            file.save(filepath)
            flash(f"File '{original_filename}' uploaded successfully. Processing...", 'info')
            print(f"File saved to: {filepath}")

            transcript = transcribe_audio(filepath)
            if not transcript:
                flash("Transcription failed or returned no content.", 'error')
                if os.path.exists(filepath): os.remove(filepath)
                return redirect(url_for('index'))
            print("Transcription successful.")

            summary = summarize_with_gemini(transcript, final_summary_prompt_text)
            print("Summarization process finished.")

            return render_template('results.html', 
                                   transcript=transcript, 
                                   summary=summary, 
                                   filename=original_filename,
                                   filename_json=json.dumps(original_filename), # Pass filename as JSON for JS
                                   used_prompt=final_summary_prompt_text.replace("{transcript}", "[Transcript was inserted here]"))

        except Exception as e:
            flash(f"An error occurred: {str(e)}", 'error')
            print(f"Error in /transcribe route: {e}")
            return redirect(url_for('index'))
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Cleaned up uploaded file: {filepath}")
                except OSError as e_del:
                    print(f"Error deleting file {filepath} during cleanup: {e_del}")
    else:
        flash(f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}", 'error')
        return redirect(url_for('index'))

@app.route('/send_to_discord', methods=['POST'])
def send_to_discord_route():
    """Receives summary and webhook URL, then sends to Discord."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid request data."}), 400

    webhook_url = data.get('webhook_url')
    summary_text = data.get('summary_text')
    filename = data.get('filename', 'Audio File') # Optional: include filename

    if not webhook_url or not summary_text:
        return jsonify({"status": "error", "message": "Webhook URL and summary text are required."}), 400

    # Discord message limit is 2000 characters. Truncate if necessary.
    max_len = 1900 # Leave some room for formatting
    if len(summary_text) > max_len:
        summary_text = summary_text[:max_len] + "... (summary truncated)"
    
    discord_payload = {
        "content": f"**Summary for: {filename}**\n\n>>> {summary_text}"
    }

    try:
        response = requests.post(webhook_url, json=discord_payload, timeout=10)
        response.raise_for_status() 
        return jsonify({"status": "success", "message": "Summary sent to Discord successfully!"})
    except requests.exceptions.RequestException as e:
        print(f"Error sending to Discord: {e}")
        error_message = str(e)
        # Attempt to get status code if response object exists
        status_code = response.status_code if 'response' in locals() and response is not None else None

        if "No schema supplied" in error_message or "Invalid URL" in error_message:
             error_message = "Invalid Discord Webhook URL provided."
        elif status_code == 404:
             error_message = "Discord Webhook URL not found (404)."
        elif status_code == 401:
             error_message = "Unauthorized - Check your Discord Webhook URL (401)."
        elif status_code == 403:
             error_message = "Forbidden - Check your Discord Webhook permissions (403)."
        # Add more specific error messages based on common Discord webhook errors if needed

        return jsonify({"status": "error", "message": f"Failed to send summary to Discord: {error_message}"}), 500
    except Exception as e:
        print(f"Unexpected error sending to Discord: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred while sending to Discord."}), 500


if __name__ == '__main__':
    print("Starting Flask app...")
    print("To run for development: flask run --host 0.0.0.0 --port 5000")
    print("For Waitress: waitress-serve --host 0.0.0.0 --port 5000 app:app")
    app.run(debug=True, host='0.0.0.0', port=5000)
