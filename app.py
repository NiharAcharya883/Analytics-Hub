from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.serving import WSGIRequestHandler
WSGIRequestHandler.protocol_version = "HTTP/1.1"  # Keeps connections alive

# LANGUAGE TRANSLATOR LIBRARIES
from googletrans import Translator, LANGUAGES
import fitz  # PyMuPDF for PDFs
from docx import Document
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from flask import jsonify, send_file, flash
translator = Translator()

# FAKE NEWS DETECTION LIBRARIES
import pickle
import re
import string

# CONTENT PLAGIARISM DETECTION LIBRARIES
import os
import re
import requests
import bs4
import spacy
import PyPDF2
from flask import Flask, render_template, request, jsonify
from googlesearch import search
from summa import summarizer
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename


# TEXT SUMMARIZATION LIBRARIES

# SENTIMENT ANALYSIS LIBRARIES

app = Flask(__name__)

app.config['TIMEOUT'] = 60  # Increase request timeout to 60 seconds
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disables caching

nlp = spacy.load("en_core_web_sm")
local_model_path = r"C:\Users\nihar\Desktop\8th Sem\AI Tools\all-MiniLM-L6-v2"
model = SentenceTransformer(local_model_path)

# Set plagiarism_upload folder
UPLOAD_FOLDER = "plagiarism_uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pdf"}

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/about')
def about():
    return render_template('About.html')  # Example, replace with about.html if needed

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Example, replace with contact.html if needed


# LANGUAGE TRANSLATOR
# Function to extract text from uploaded files
def doc2text(file_path):
    if file_path.endswith('.pdf'):
        with fitz.open(file_path) as doc:
            text = ''.join(page.get_text() for page in doc)
        return text
    elif file_path.endswith(('.docx', '.DOCX', '.Docx')):
        document = Document(file_path)
        return '\n'.join(paragraph.text for paragraph in document.paragraphs)
    elif file_path.endswith(('.html', '.HTML')):
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'lxml')
        return soup.get_text(separator='\n', strip=True)
    elif file_path.endswith(('.jpg', '.jpeg', '.png')):
        return pytesseract.image_to_string(Image.open(file_path))
    return None

# Route for Translator Page
@app.route('/translator')
def translator_page():
    return render_template('Translator.html', languages=LANGUAGES)

# API Route for Text Translation
@app.route('/translate', methods=['POST'])
def translator_2():
    languages = LANGUAGES
    text = request.form.get('text')
    language = request.form.get('language','en')

    if not text:
        return jsonify({'error': 'Please enter text to translate'}), 400

    if len(text) > 15000:
        return jsonify({'error': 'Character limit exceeded (Max: 15,000)'}), 400
    print(text)
    print(language)
    print(type(translator))
    print(len(text))
    try:
        translated_text = translator.translate(text, dest=language).text
        return jsonify({'translated_text': translated_text})
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

# API Route for File Upload Translation
@app.route('/translate_file', methods=['POST'])
def translate_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    language = request.form.get('language')

    if not file or file.filename == '':
        return jsonify({'error': 'Please select a file to translate'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    extracted_text = doc2text(file_path)

    if not extracted_text:
        return jsonify({'error': 'Unsupported file format'}), 400

    if len(extracted_text) > 15000:
        return jsonify({'error': 'Character limit exceeded (Max: 15,000)'}), 400


    translated_text = translator.translate(extracted_text, dest=language).text
    print(translated_text)

    try:
        # Save translated text as PDF using ReportLab
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'translated.pdf')

        # Create a PDF Canvas
        c = canvas.Canvas(pdf_path, pagesize=letter)

        # Register and Set a Unicode Font
        font_path = "DejaVuSans.ttf"  # Path to a proper Unicode font (must be available)
        pdfmetrics.registerFont(TTFont("DejaVu", font_path))  # Register the font
        c.setFont("DejaVu", 12)  # Use the registered font

        # Wrap long text properly
        text_object = c.beginText(40, 750)  # Starting position
        text_object.setFont("DejaVu", 12)

        # Manually wrap long text lines
        max_width = 80  # Max characters per line
        lines = [translated_text[i:i+max_width] for i in range(0, len(translated_text), max_width)]

        for line in lines:
            text_object.textLine(line)

        c.drawText(text_object)
        c.showPage()
        c.save()

        return jsonify({'download_url': '/download_translated_pdf'})

    except Exception as e:
        flash(f"Error generating PDF: {str(e)}", "error")
        return jsonify({'error': "Failed to generate PDF. Please try again."})

# Route to Download Translated PDF
@app.route('/download_translated_pdf')
def download_translated_pdf():
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'translated.pdf'), as_attachment=True)
    except Exception as e:
        flash(f"Error downloading PDF: {str(e)}", "error")
        return jsonify({'error': "Failed to download PDF. Please try again."})












# FAKE NEWS DETECTION
# Load the trained model and vectorizer
model_path = 'static/FakeNewsmodel.pkl'
vectorizer_path = 'static/FakeNewsvectorizer.pkl'

with open(model_path, 'rb') as model_file:
    classifier = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text preprocessing function
def preprocess(text):  
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'<.*?>+', '', text)  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  
    text = re.sub(r'\n', ' ', text)  
    text = re.sub(r'\w*\d\w*', '', text)  
    text = re.sub(r'\W', ' ', text)  
    return text.strip()

# Route for Fake News Detection
@app.route("/fake-news", methods=['GET', 'POST'])
def fake_news():
    if request.method == "POST":
        user_input = request.form.get("news_text", "").strip()
        if not user_input:
            return jsonify({"error": "Please enter news text to check."})
        
        processed_text = preprocess(user_input)
        user_input_vectorized = vectorizer.transform([processed_text])
        prediction = classifier.predict(user_input_vectorized)
        result = "Fake News" if prediction[0] == 1 else "Real News"
        return jsonify({"result": result})

    return render_template("FakeNews.html")


# CONTENT PLAGIARISM DETECTOR

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_text_from_pdf(pdf_path):
#     """Extracts text from a PDF file."""
#     try:
#         with open(pdf_path, "rb") as file:
#             reader = PyPDF2.PdfReader(file)
#             text = ""
#             for page in reader.pages:
#                 text += page.extract_text() + "\n"
#         return text.strip()
#     except Exception as e:
#         print(f"Error extracting text from PDF: {e}")
#         return ""

def preprocess_text(text):
    """Cleans text by removing special characters and extra spaces."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.!?]', '', text)
    return text

def extract_key_sentences_textrank(text, num_sentences=3):
    """Extracts important sentences using TextRank."""
    summary = summarizer.summarize(text, ratio=0.3)
    if not summary.strip():
        return text
    doc = nlp(summary)
    sentences = [sent.text.strip() for sent in doc.sents]
    return " ".join(sentences[:num_sentences])

def extract_key_sentences_embeddings(text, num_sentences=3):
    """Finds unique & important sentences using embeddings."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    if not sentences:
        return text
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    sentence_scores = sentence_embeddings.mean(dim=1).tolist()
    sorted_sentences = [sent for _, sent in sorted(zip(sentence_scores, sentences), reverse=True)]
    return " ".join(sorted_sentences[:num_sentences])

def extract_key_sentences(text):
    """Choose method based on text length."""
    if len(text.split()) < 100:
        return extract_key_sentences_textrank(text, num_sentences=3)
    return extract_key_sentences_embeddings(text, num_sentences=5)

def google_search(query, num_results=10):
    """Fetches top search results from Google."""
    return list(search(query, num_results=num_results))

def preprocess_web_text(text):
    """Cleans web-scraped text while keeping structure."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[“”‘’]', '', text)
    text = re.sub(r'[^a-z0-9.,!?;:\'\"()\s]', '', text)
    return text

def scrape_website(url):
    """Scrapes text from a webpage."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return preprocess_web_text(text)
    except Exception:
        return ""

def calculate_similarity(input_text, website_text):
    """Computes similarity score using SBERT."""
    if not website_text:
        return 0
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    website_embedding = model.encode(website_text, convert_to_tensor=True)
    return round(util.pytorch_cos_sim(input_embedding, website_embedding).item() * 100, 2)

def check_plagiarism(input_text):
    """Checks plagiarism and returns top 3 results."""
    input_text = preprocess_text(input_text)
    key_sentences = extract_key_sentences(input_text)
    urls = google_search(key_sentences)
    for i in urls:
        print(i)
    similarities = {}
    for url in urls:
        web_text = scrape_website(url)
        similarity = calculate_similarity(input_text, web_text)
        similarities[url] = similarity
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    return sorted_results

@app.route("/plagiarism", methods=["GET", "POST"])
def plagiarism():
    if request.method == "POST":
        input_text = request.form.get("text", "").strip()

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        results = check_plagiarism(input_text)
        return jsonify(results)

    return render_template("plagiarism.html")




if __name__ == '__main__':
    app.run(debug=True)