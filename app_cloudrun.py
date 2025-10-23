from flask import Flask, request, jsonify, render_template
import requests
import PyPDF2
import os
import json
from datetime import datetime

app = Flask(__name__)

PROJECT_ID = "gen-lang-client-0035881252"
LOCATION = "us-central1"

# Your fine-tuned model endpoint
TUNED_MODEL_ENDPOINT = "gemini-2.5-flash"  # Replace with your tuned model ID after checking console

def get_access_token():
    """Get access token for Cloud Run"""
    try:
        import google.auth
        from google.auth.transport.requests import Request
        
        credentials, _ = google.auth.default()
        credentials.refresh(Request())
        return credentials.token
    except:
        # Local fallback
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'print-access-token'], 
                              capture_output=True, text=True)
        return result.stdout.strip()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_claim(claim_text):
    """Analyze using fine-tuned model"""
    token = get_access_token()
    
    url = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{TUNED_MODEL_ENDPOINT}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "role": "user",
            "parts": [{"text": claim_text[:3000]}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        result = response.json()
        
        if 'candidates' in result:
            text = result['candidates'][0]['content']['parts'][0]['text']
            text = text.replace('```json', '').replace('```', '').strip()
            return json.loads(text)
        
        return {"error": "No response"}
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    # Save to /tmp (Cloud Run has writable /tmp)
    filepath = os.path.join('/tmp', file.filename)
    file.save(filepath)
    
    # Extract and analyze
    claim_text = extract_text_from_pdf(filepath)
    
    if "Error" in claim_text:
        return jsonify({"error": claim_text}), 500
    
    result = analyze_claim(claim_text)
    result['filename'] = file.filename
    result['processed_at'] = datetime.now().isoformat()
    result['model'] = 'fine-tuned-gemini-2.5-flash'
    
    # Cleanup
    try:
        os.remove(filepath)
    except:
        pass
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)