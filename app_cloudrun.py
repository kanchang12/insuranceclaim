from flask import Flask, request, jsonify, render_template
import PyPDF2
import os
import json
from datetime import datetime
import logging
import tempfile
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

ENDPOINT = "projects/451954006366/locations/us-central1/endpoints/7356989531611987968"

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            return "Error: No text could be extracted from PDF"
        
        return text
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        return f"Error: {str(e)}"

def analyze_claim(claim_text):
    """Analyze insurance claim using Gemini model"""
    try:
        client = genai.Client(vertexai=True)
        
        # Updated prompt to be generic for all insurance types
        prompt = f"""Analyze this insurance claim document and assess the risk level:

{claim_text[:1500]}

Evaluate the claim based on:
- Document completeness and clarity
- Claim amount reasonableness
- Supporting evidence quality
- Red flags or inconsistencies

Return ONLY a valid JSON object with no additional text or formatting."""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ]
        
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=512,
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "risk_score": types.Schema(
                        type=types.Type.INTEGER, 
                        description="Risk score from 0-100 (0=low risk, 100=high risk)"
                    ),
                    "recommendation": types.Schema(
                        type=types.Type.STRING, 
                        description="One of: APPROVE, REVIEW, or DENY"
                    ),
                    "reasoning": types.Schema(
                        type=types.Type.STRING, 
                        description="Detailed explanation for the score and recommendation"
                    )
                },
                required=["risk_score", "recommendation", "reasoning"]
            )
        )
        
        logging.info("Calling Gemini API for claim analysis...")
        response = client.models.generate_content(
            model=ENDPOINT,
            contents=contents,
            config=config
        )
        
        text = response.text.strip()
        logging.info(f"Received response from API: {text[:100]}...")
        
        try:
            parsed = json.loads(text)
            
            # Validate and sanitize the response
            risk_score = int(parsed.get('risk_score', 50))
            risk_score = max(0, min(100, risk_score))  # Clamp between 0-100
            
            recommendation = parsed.get('recommendation', 'REVIEW').upper()
            if recommendation not in ['APPROVE', 'REVIEW', 'DENY']:
                recommendation = 'REVIEW'
            
            reasoning = str(parsed.get('reasoning', 'No reasoning provided'))[:1000]
            
            return {
                "risk_score": risk_score,
                "recommendation": recommendation,
                "reasoning": reasoning
            }
        
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}. Raw text: {text[:500]}")
            return {
                "risk_score": 50,
                "recommendation": "REVIEW",
                "reasoning": f"Unable to parse model response. Please review manually."
            }
        
    except Exception as e:
        logging.error(f"Error during claim analysis: {e}", exc_info=True)
        return {
            "error": str(e),
            "risk_score": 50,
            "recommendation": "REVIEW",
            "reasoning": f"System error occurred: {str(e)}"
        }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded insurance claim PDF"""
    filepath = None
    try:
        # Validate file upload
        if 'file' not in request.files:
            logging.warning("No file provided in request")
            return jsonify({
                "error": "No file uploaded", 
                "risk_score": 0, 
                "recommendation": "ERROR", 
                "reasoning": "Please upload a PDF file"
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "error": "Empty filename", 
                "risk_score": 0, 
                "recommendation": "ERROR", 
                "reasoning": "No file selected"
            }), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                "error": "Invalid file type", 
                "risk_score": 0, 
                "recommendation": "ERROR", 
                "reasoning": "Only PDF files are supported"
            }), 400
        
        # Create secure filepath
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"{datetime.now().timestamp()}_{filename}")
        
        logging.info(f"Processing file: {filename}")
        file.save(filepath)
        
        # Extract text from PDF
        claim_text = extract_text_from_pdf(filepath)
        
        if claim_text.startswith("Error"):
            logging.error(f"PDF extraction failed: {claim_text}")
            return jsonify({
                "error": "PDF processing failed", 
                "risk_score": 0, 
                "recommendation": "ERROR", 
                "reasoning": claim_text
            }), 200
        
        # Analyze the claim
        result = analyze_claim(claim_text)
        result['filename'] = filename
        result['processed_at'] = datetime.now().isoformat()
        result['model'] = 'fine-tuned-gemini'
        
        logging.info(f"Analysis complete: {result['recommendation']} (score: {result['risk_score']})")
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Unexpected error in /analyze: {e}", exc_info=True)
        return jsonify({
            "error": str(e), 
            "risk_score": 0, 
            "recommendation": "ERROR", 
            "reasoning": f"System error: {str(e)}"
        }), 500
    
    finally:
        # Always clean up temp file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logging.info(f"Cleaned up temp file: {filepath}")
            except Exception as cleanup_error:
                logging.warning(f"Could not remove temp file {filepath}: {cleanup_error}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
