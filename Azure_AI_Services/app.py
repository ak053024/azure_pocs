from flask import Flask, request, render_template
import mimetypes
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.ai.translation.text import TextTranslationClient
from azure.core.exceptions import HttpResponseError
import azure.cognitiveservices.speech as speechsdk

app = Flask(__name__)

# Load Azure API Keys (Replace with actual keys)
AZURE_VISION_KEY = ""
AZURE_VISION_ENDPOINT = ""
AZURE_TEXT_ANALYTICS_KEY = ""
AZURE_TEXT_ANALYTICS_ENDPOINT = ""
AZURE_TRANSLATOR_KEY = ""
AZURE_TRANSLATOR_ENDPOINT = ""
AZURE_SPEECH_KEY = ""
AZURE_SPEECH_REGION = ""
AZURE_DOCUMENT_INTELLIGENCE_KEY = ""
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = ""

vision_client = ImageAnalysisClient(endpoint=AZURE_VISION_ENDPOINT,credential=AzureKeyCredential(AZURE_VISION_KEY))
document_client = DocumentIntelligenceClient(endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY))
text_analytics_client = TextAnalyticsClient(AZURE_TEXT_ANALYTICS_ENDPOINT, AzureKeyCredential(AZURE_TEXT_ANALYTICS_KEY))
translator_client = TextTranslationClient(endpoint=AZURE_TRANSLATOR_ENDPOINT,credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY), region=AZURE_SPEECH_REGION)

# Function to detect file type
def get_file_type(file_path):
    try:
        mime_type, _ = mimetypes.guess_file_type(file_path)
        return mime_type
    except HttpResponseError as exception:
        if exception.error is not None:
            print(f"Error Code: {exception.error.code}")
            print(f"get_file_type - Message: {exception.error.message}")
        raise

# Function to extract text from an image
def extract_text_from_image(image_path):
    try:
        # Load image to analyze into a 'bytes' object
        with open(image_path, "rb") as f:
            image_data = f.read()
        # Extract text (OCR) from an image stream. This will be a synchronously (blocking) call.
        response = vision_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ]
        )
        extracted_text = "\n".join([line.text for page in response.read.blocks for line in page.lines])
        return extracted_text
    except HttpResponseError as exception:
        if exception.error is not None:
            print(f"Error Code: {exception.error.code}")
            print(f"extract_text_from_image - Message: {exception.error.message}")
        raise

# Function to extract text from documents (PDF)
def extract_text_from_document(file_path):
    try:
        with open(file_path, "rb") as file:
            poller = document_client.begin_analyze_document("prebuilt-layout", body=file)
            result = poller.result()
    
            extracted_text = " ".join([line.content for page in result.pages for line in page.lines])
            return extracted_text
    except HttpResponseError as exception:
        if exception.error is not None:
            print(f"Error Code: {exception.error.code}")
            print(f"extract_text_from_document - Message: {exception.error.message}")
        raise

# Function to summarize text
def summarize_text(text):
    try:
        response = text_analytics_client.extract_key_phrases([text])
        result_text = "\n".join("Key phrases in article #{}: {}".format(idx + 1, ", ".join(doc.key_phrases))
            for idx, doc in enumerate(response) if not doc.is_error)
        print(result_text)
        return result_text
    except HttpResponseError as exception:
        if exception.error is not None:
            print(f"Error Code: {exception.error.code}")
            print(f"summarize_text - Message: {exception.error.message}")
        raise

# Function to translate text
def translate_text(text, target_language):
    try:
        response = translator_client.translate(body=[text], to_language=["fr"], from_language="en")
        return response[0].translations[0].text
    except HttpResponseError as exception:
        if exception.error is not None:
            print(f"Error Code: {exception.error.code}")
            print(f"Message: {exception.error.message}")
        raise

# Function to convert text to speech
def text_to_speech(text):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesizer.speak_text_async(text)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def process_file():
    file = request.files["file"]
    function = request.form["function"]
    
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    file_type = get_file_type(file_path)
    if file_type.startswith("image"):
        text = extract_text_from_image(file_path)
    elif file_type in ["application/pdf"]:
        text = extract_text_from_document(file_path)
    else:
        return "Unsupported file type"

    if function == "summarize":
        result = summarize_text(text)
    elif function == "translate":
        result = translate_text(text, "")
    elif function == "tts":
        text_to_speech(text)
        result = "Text converted to speech"
    else:
        result = text
    
    return result
    
if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000, debug=True)
    
    