from flask import Flask, render_template, request, redirect, url_for
import os
import base64
from openai import AzureOpenAI
import pdfplumber
import docx


# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

endpoint = os.getenv("ENDPOINT_URL", "")  
deployment = os.getenv("DEPLOYMENT_NAME", "")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "") 

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to call Azure OpenAI API
def summarize_resume(text):
    client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key, 
    api_version="2024-05-01-preview",
    )
    chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "summarize the resume in 3 sentences.c" + text
            }
        ]
    }
    ]    
    messages = chat_prompt  
    completion = client.chat.completions.create(  
    model=deployment,
    messages=messages,
    max_tokens=800,  
    temperature=0.7,  
    top_p=0.95,  
    frequency_penalty=0,  
    presence_penalty=0,
    stop=None,  
    stream=False
    )
    return completion.choices[0].message.content

# Flask routes
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["resume"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Extract text based on file type
            if file.filename.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            elif file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            else:
                return "Unsupported file format", 400

            # Get summary from Azure OpenAI
            summary = summarize_resume(text)
            return render_template("upload.html", summary=summary)

    return render_template("upload.html", summary=None)

if __name__ == "__main__":
    app.run(debug=True)
