from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import requests
from bs4 import BeautifulSoup
import os
import re
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import requests
import sys
from bs4 import BeautifulSoup
import re
import asyncio
from script import process_privacy_policy, process_Terms#, analyze_text_pipeline_pP, analyze_text_pipeline_Terms, replace_multiple_newlines, chunkify_flags, process_sentences_with_bert
from html import unescape
import csv
import html
import torch
from flask_socketio import SocketIO, emit
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import asyncio
# from get_embeddings import get_text_embeddings_from_list  # Ensure this is correctly implemented
import openai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import random
from flask_session import Session
import redis
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
import stripe
import json
from flask import send_from_directory

from datetime import timedelta
app = Flask(__name__)
socketio = SocketIO(app)

app.secret_key = os.urandom(24)
stripe.api_key = os.getenv("STRIPE_API_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")

app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_REDIS'] = redis.StrictRedis(host='localhost', port=6379)  # Replace with your Redis host if needed

Session(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simulated in-memory user storage (email -> {password, is_premium})
USER_FILE = 'users.json'

# Load users from file
def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as file:
            return json.load(file)
    return {}

# Save users to file

# Initialize users by loading from file
users = load_users()

class User(UserMixin):
    def __init__(self, email):
        self.id = email

    @property
    def is_premium(self):
        return users[self.id]["is_premium"]

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# Route for Login Page

@app.route('/sitemap.xml')
def serve_sitemap():
    return send_from_directory(app.root_path, 'sitemap.xml')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember = 'remember' in request.form
        users = load_users()  # Load users from file

        if email in users and check_password_hash(users[email]["password"], password):
            user = User(email)
            
            login_user(user, remember=True)
            
            # Check if the user is premium
            if user.is_premium:
                return redirect(url_for('home'))  # Redirect to homepage if premium

            return redirect(url_for('home'))  # Redirect after login
        else:
            flash('Invalid email or password')

    return render_template('login.html')



def save_users():
    with open('users.json', 'w') as file:
        json.dump(users, file)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if email in users:
            flash('Email already exists. Please log in.')
            return redirect(url_for('login'))
        
        users[email] = {"password": generate_password_hash(password), "is_premium": False}
        save_users()  # Save the user to the JSON file
        
        user = User(email)
        login_user(user)
        flash('Account created successfully.')
        return redirect(url_for('payment'))
    
    return render_template('signup.html')

@app.route('/confirm-premium', methods=['POST'])
def confirm_premium():
    data = request.json
    email = data.get('email')

    if email not in users:
        return jsonify({'error': 'User not found'}), 400

    # Mark the user as premium
    users[email]['is_premium'] = True
    save_users()

    # Log the user in
    user = User(email)
    login_user(user, remember=True)

    return jsonify({'message': 'User upgraded to premium'})
# Home route for showing the payment form (login required)
@app.route('/payment')
@login_required
def payment():
    return render_template('payment_form.html', is_premium=current_user.is_premium)

# Create Payment Intent (for Stripe Elements)
@app.route('/create-payment-intent', methods=['POST'])
@login_required
def create_payment():
    try:
        intent = stripe.PaymentIntent.create(
            amount=600,  # $10 in cents
            currency='usd',
            payment_method_types=['card'],
        )
        return jsonify({'clientSecret': intent['client_secret']})
    except Exception as e:
        return jsonify(error=str(e)), 403

# Route for payment success (after Stripe payment succeeds)
@app.route('/payment-success', methods=['POST'])
@login_required
def payment_success():
    users[current_user.id]["is_premium"] = True
    save_users()  # Now this works without any arguments
    return jsonify({'message': 'User premium status updated'})
# Route for Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('home'))



def get_random_quote():
    with open('quotes.txt', 'r') as file:
        quotes = file.readlines()
    return random.choice(quotes).strip()


def load_definitions(filepath):
    definitions = {}
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:  # Check if the row has at least two columns
                print(f"Skipping incomplete row: {row}")
                continue
            key = row[0].strip().lower()  # Normalize the key to lowercase
            definitions[key] = row[1].strip()
    return definitions

def underline_terms_pP(text):
    definitions = load_definitions('vocab_pP.csv')
    terms = definitions
    if not isinstance(text, (str, bytes)):
        print(f"Warning: Non-string input received: {text}")
        return text  # or handle the error differently as needed

    # Sort terms by length in descending order to prevent partial matching issues
    sorted_terms = sorted(terms.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_terms)) + r')\b', re.IGNORECASE)

    # Function to add HTML tags
    def replace(match):
        original_word = match.group(0)
        word = original_word.lower()
        if word in terms:
            # Escape quotes and special characters in the definition to avoid breaking HTML
            definition = html.escape(terms[word])
            return f"<span class='highlight' data-definition=\"{definition}\">{original_word}</span>"
        return original_word

    return pattern.sub(replace, text)

def underline_terms_Terms(text):
    definitions = load_definitions('vocab_Terms.csv')
    terms = definitions
    if not isinstance(text, (str, bytes)):
        print(f"Warning: Non-string input received: {text}")
        return text  # or handle the error differently as needed

    # Sort terms by length in descending order to prevent partial matching issues
    sorted_terms = sorted(terms.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_terms)) + r')\b', re.IGNORECASE)

    # Function to add HTML tags
    def replace(match):
        original_word = match.group(0)
        word = original_word.lower()
        if word in terms:
            # Escape quotes and special characters in the definition to avoid breaking HTML
            definition = html.escape(terms[word])
            return f"<span class='highlight' data-definition=\"{definition}\">{original_word}</span>"
        return original_word

    return pattern.sub(replace, text)

def read_phrases(file_path):
    with open(file_path, 'r') as file:
        phrases = [line.strip() for line in file.readlines()]
    return phrases

def highlight_phrases(sentence):
    with open('red_lines.txt', 'r') as file:
        phrases = [line.rstrip('\n') for line in file.readlines()]

    # Sort phrases by length in descending order
    sorted_phrases = sorted(phrases, key=len, reverse=True)
    
    for phrase in sorted_phrases:
        pattern = re.escape(phrase)  # Escape special characters in phrase
        
        # Add \b at the beginning and end to match whole words
        sentence = re.sub(f"(?i)({pattern})", r'<span class="red_highlight">\1</span>', sentence)
    return sentence

def highlight_phrases_pP(sentence):
    with open('red_lines_pP.txt', 'r') as file:
        phrases = [line.rstrip('\n') for line in file.readlines()]

    # Sort phrases by length in descending order
    sorted_phrases = sorted(phrases, key=len, reverse=True)
    
    for phrase in sorted_phrases:
        pattern = re.escape(phrase)  # Escape special characters in phrase
        
        # Add \b at the beginning and end to match whole words
        sentence = re.sub(f"(?i)({pattern})", r'<span class="red_highlight">\1</span>', sentence)
    return sentence

def insert_markers_around_flagged_sentences(input_text, flagged_sentences):
    """
    Inserts 'MARKER' before and after flagged sentences in the input text,
    considering only alphabetical characters and ignoring punctuation and whitespace.
    """
    # Create normalized text and mapping from normalized index to original index
    normalized_text, mapping = create_index_mapping(input_text)
    
    # Collect all the insertions to make
    insertions = []
    
    # Process each flagged sentence
    for sentence in flagged_sentences:
        # Normalize the flagged sentence
        normalized_sentence, _ = create_index_mapping(sentence)
        
        # Find all non-overlapping matches of the normalized sentence in the normalized text
        start = 0
        while True:
            start = normalized_text.find(normalized_sentence, start)
            if start == -1:
                break
            
            # Calculate the start and end indices in the original text
            original_start = mapping[start]
            original_end = mapping[start + len(normalized_sentence) - 1] + 1

            # Schedule the insertions
            insertions.append((original_end, "</div>"))
            insertions.append((original_start, "<div class='bolds'>"))
            
            # Move start index beyond the current match to prevent overlapping
            start += len(normalized_sentence)
    
    # Apply the insertions in reverse order by sorting them by position descending
    insertions.sort(reverse=True, key=lambda x: x[0])
    output_text = list(input_text)
    for pos, text in insertions:
        output_text.insert(pos, text)

    return ''.join(output_text)

def create_index_mapping(text):
    """
    Create a normalized version of the text that contains only alphabetical characters,
    and a mapping list where each index corresponds to the position of the character in the original text.
    """
    normalized_text = []
    mapping = []
    for index, char in enumerate(text):
        if char.isalpha():
            normalized_text.append(char.lower())
            mapping.append(index)
    return ''.join(normalized_text), mapping

def clean_text(text):

    pattern = r'[^a-zA-Z0-9\s\.,;:!?@#%$+=\'"-]'
    text = re.sub(r'(?<=[a-zA-Z])\.(?=[A-Z])', '. ', text)
    # Use re.sub() to replace unwanted characters with an empty string
    text = re.sub(pattern, '', text)

    return text

def is_url(string):
    print("doodoo")
    return string.startswith('http://') or string.startswith('https://')

def get_text_by_link(link, file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['link'] == link:
                return row['text']
    return None

def scrape_url(url):
    print("BBL Drizzy")
    csv.field_size_limit(sys.maxsize)
    file_path = 'links_and_texts.csv'
    cached_text = get_text_by_link(url, file_path)
    if cached_text:
        print("Using cached text from CSV")
        return clean_text(cached_text)

    try:
        response = requests.get(url, timeout=4)
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator='', strip=False)
        cleaned_text = clean_text(text_content)
        if ".docx" in url:
           return " For docs files, copy and paste the whole text into the search bar."
        if ".otd" in url:
           return " For .otd files, copy and paste the whole text into the search bar."
        if ".rtf" in url:
           return " For .rtf files, copy and paste the whole text into the search bar."
        if ".md" in url:
           return " For .md files, copy and paste the whole text into the search bar."
        if ".docs" in url:
           return " For docs files, copy and paste the whole text into the search bar."
        if ".pdf" in url:
           return " For PDF files, copy and paste the whole text into the search bar."
        if url.lower().endswith('.pdf'):
           return " For PDF files, copy and paste the whole text into the search bar."
        if url.lower().startswith('https://docs'):
           return " For docs files, copy and paste the whole text into the search bar."
        if url.lower().startswith('http://docs'):
           return " For docs files, copy and paste the whole text into the search bar."
        if len(cleaned_text) < 7000:
           x = scraper_api_scraping(url)
           if len(x) < 1000:
              return " We didn't find much text on that webpage, paste the entire text into the search bar instead."
           return scraper_api_scraping(url)

        return cleaned_text
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while trying to scrape the URL: {e}")
        x = scraper_api_scraping(url)
        if len(x) < 1000:
           return " We didn't find much text on that webpage, paste the entire text into the search bar instead."
        if len(x) > 1000:
           return x
        return " We didn't find much text on that webpage, paste the entire text into the search bar instead."
    

def scraper_api_scraping(url):
    # Create a session object
    print("USING SCRAPER API")
    with requests.Session() as session:
        # Set up the ScraperAPI URL with the API key and target URL
        api_url = f'http://api.scraperapi.com'
        params = {
            'api_key': "e9339f2b9adaf1a90d475e0ca7549b1f",
            'url': url
        }

        try:
            # Send the GET request using the session
            response = session.get(api_url, params=params, timeout=17)
            response.raise_for_status()  # Check for HTTP errors

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            text =  soup.get_text(strip=True)
            return text

        except requests.exceptions.Timeout:
            print(f"A timeout error occurred")
            return "Copy and Paste the full contract into the search bar, the Link could not be scraped." 

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return "Copy and Paste the full contract into the search bar, the Link could not be scraped." 

# def scrape_url(url):
#     # Set up Selenium WebDriver
#     service = Service(ChromeDriverManager().install())
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--disable-gpu')
#     driver = webdriver.Chrome(service=service, options=options)
    
#     # Open the webpage
#     driver.get(url)
#     time.sleep(1)  # Wait for JavaScript to render the content

#     # Get the page source and close the browser
#     page_source = driver.page_source
#     driver.quit()

#     # Parse the page source with BeautifulSoup
#     soup = BeautifulSoup(page_source, 'html.parser')
#     text = soup.get_text()
#     cleaned_text = clean_text(text)
#     if len(cleaned_text) < 2000:
#         return "ERROR: Link could not be scraped. No worries, just copy and paste the entire contract in the search bar. (Cmd + A to select all, Cmd + C to copy)"

#     return cleaned_text

def split_long_sentences(text, max_length=120):
    """
    Splits text into sentences, ensuring none exceeds the specified maximum length.
    Splits are made at a period followed by a space, '. ', or a semicolon ';'.
    """
    # Split the text into sentences first
    initial_sentences = re.split(r'(?<=[.?!])[\s\n]+(?=[A-Z0-9"\(])', text)

    # Process each sentence to ensure it doesn't exceed the maximum length
    processed_sentences = []
    for sentence in initial_sentences:
        # If the sentence is too long, split it further, specifically looking for '. ' to split.
        if len(sentence) > max_length:
            # Split by period followed by a space, keeping the delimiter in place
            sub_sentences = re.split(r'(?<=[.?!;])\s+(?=[A-Z])', sentence)
            processed_sentences.extend(sub_sentences)
        else:
            processed_sentences.append(sentence)
    
    return processed_sentences

def replace_abbreviations(text):
    # Define a dictionary for replacements
    replacements = {
        # ... [your existing replacements] ...
        'e.g.': 'eg',
        'etc.': 'etc',
        'viz.': 'viz',
        'vs.': 'vs',
        'Mr.': 'Mr',
        'Mrs.': 'Mrs',
        'Dr.': 'Dr',
        'Prof.': 'Prof',
        'Jan.': 'Jan',
        'Feb.': 'Feb',
        'Mar.': 'Mar',
        'Apr.': 'Apr',
        'Jun.': 'Jun',
        'Jul.': 'Jul',
        'Aug.': 'Aug',
        'Sep.': 'Sep',
        'Sept.': 'Sept',
        'Oct.': 'Oct',
        'Nov.': 'Nov',
        'Dec.': 'Dec',
        'a.m.': 'am',
        'p.m.': 'pm',
        'Ph.D.': 'PhD',
        'M.D.': 'MD',
        'B.A.': 'BA',
        'M.A.': 'MA',
        'cf.' : 'cf',
        '(e.g.': 'eg',
        'D.D.S.': 'DDS',
        'R.N.': 'RN',
        'P.S.': 'PS',
        'U.S.': 'US',
    }

    # Split the text into words
    words = text.split()

    # Iterate through words and replace abbreviations
    replaced_words = [replacements.get(word, word) for word in words]

    # Join the words back into a string
    replaced_text = ' '.join(replaced_words)

    # Split long sentences after potential abbreviations are handled
 #    replaced_text = split_long_sentences(replaced_text)

    return replaced_text

def index_sentences_with_formatting(text, max_length=1000):
    """
    Processes the input text into sentences and prints the index before each sentence,
    while preserving the exact formatting of the input text.
    """
    sentences = split_long_sentences(text, max_length)
    
    # Create a list to store the parts of the indexed text
    indexed_text_parts = []
    last_pos = 0
    index_and_sentence = []
    for index, sentence in enumerate(sentences, start=1):
        # Find the sentence in the original text starting from last_pos to handle duplicates
        start_pos = text.find(sentence, last_pos)
        if start_pos == -1:
            continue
        end_pos = start_pos + len(sentence)
        
        # Append the part of the text before the current sentence
        indexed_text_parts.append(text[last_pos:start_pos])
        
        # Append the indexed sentence
        indexed_text_parts.append(f"[{index}] {sentence}")
        index_and_sentence.append((-1,text[last_pos:start_pos]))
        index_and_sentence.append((index,sentence))
        # Update the last_pos to the end of the current sentence
        last_pos = end_pos

    # Add any remaining part of the text
    indexed_text_parts.append(text[last_pos:])
    
    # Join the parts into a single text
    return ("".join(indexed_text_parts)), index_and_sentence

# def remove_duplicates(sentences):
    seen = set()
    unique_sentences = []

    for sentence in sentences:
        normalized_sentence = sentence.lower().strip()
        if normalized_sentence not in seen:
            seen.add(normalized_sentence)
        else:
            continue  # Skip adding the sentence if it's a duplicate
        unique_sentences.append(sentence)
    
    return unique_sentences

def remove_duplicates(trouples):
    seen = set()
    unique_tuples = []

    for read_sentence, origin_sentence, number in trouples:
        normalized_sentence = origin_sentence.lower().strip()
        if normalized_sentence not in seen:
            seen.add(normalized_sentence)
            unique_tuples.append((read_sentence, origin_sentence, number))
    
    return unique_tuples


@app.route('/explain', methods=['POST'])
def explain():
    data = request.json
    print('Received data:', data)
    sentence = data.get('sentence')
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant that reads Terms of Service and Privacy Policies and explains dangerous clauses regarding user rights and privacy, to the everyday person."},
                {"role": "user", "content": f"Summarize this sentence for the everyday person using common language, emphasize why it may be alarming . Maximum 300 tokens: {sentence}"},
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Split the response into explanation and title using the word "STOP"
        full_response = response['choices'][0]['message']['content'].strip()
        explanation = full_response.strip()
      
        
        return jsonify({"explanation": explanation})
    except Exception as e:
        print('Error while getting explanation and title:', str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/title', methods=['POST'])
def title():
    data = request.json
    print('Received data:', data)
    sentence = data.get('sentence')
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant that reads Terms of Service and Privacy Policies and explains dangerous sentences to the everyday person."},
                {"role": "user", "content": f"Give a title for this sentence in 3-7 words. Don't provide anything else other than the title, no quotes or special characters. Make the titles very specific to the the sentence, not generic at all. ALWAYS speak in third person. If needed, speak to the User directly. Use SIMPLE wording everyone can understand, and make the sentences slightly alarming to the user.: {sentence}"},
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )

        # Split the response into explanation and title using the word "STOP"
        full_response = response['choices'][0]['message']['content'].strip()
    
        title = full_response.strip()
        title = full_response.strip().replace('"', '').replace("'", "")        
        return jsonify({"title": title + ':'})
    except Exception as e:
        print('Error while getting title:', str(e))
        return jsonify({"title": "..." + ''})
#        return jsonify({"error": str(e)}), 500


def process_text(input_text):
    sentences = replace_abbreviations(input_text)
    # sentences = remove_duplicates(sentences)
    print("ITS WORKINGGGGGGGGGGGG")
    print(len(input_text))
    
    return [sentence for sentence in sentences if len(sentence) > 0]


@app.route('/')

def home():
    with open('quotes.txt', 'r') as file:
        quotes = file.readlines()
        quote = random.choice(quotes).strip()
        if '–' in quote:
            parts = quote.split('–')
        else:
            parts = quote.split('-')
        quote_text = parts[0].strip()
        quote_author = parts[1].strip()
    return render_template('index.html', quote_text=quote_text, quote_author=quote_author)
  # Assuming your HTML file is named index.html


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how')
def how_page():
    return render_template('how_page.html')


def match_flagged_with_original(flagged_sentences, original_sentences):
    """
    Matches each flagged sentence with the original sentences and finds the index of the original sentences that match.
    
    :param flagged_sentences: A list of flagged sentences.
    :param original_sentences: A list of original sentences.
    :return: A list of tuples where each tuple contains a flagged sentence and a list of indexes of the original sentences that match it.
    """
    matches = []
    # Normalize sentences for comparison
    normalized_originals = [sentence.lower().strip() for sentence in original_sentences]
    
    for flagged_sentence in flagged_sentences:
        # Extract core text from the flagged sentence, excluding any leading or trailing ellipses
        flagged_core_text = flagged_sentence.strip(' .').lower()
        matched_indexes = []

        # Find the original sentence index(es) that contain the core text of the flagged sentence
        for index, original in enumerate(normalized_originals):
            if flagged_core_text in original:
                matched_indexes.append(index)
        
        matches.append((matched_indexes))
    
    return matches

@app.route('/results')
def results():
    with open('quotes.txt', 'r') as file:
        quotes = file.readlines()
        quote = random.choice(quotes).strip()
        if '–' in quote:
            parts = quote.split('–')
        else:
            parts = quote.split('-')
        quote_text = parts[0].strip()
        quote_author = parts[1].strip()
    return render_template('results.html', quote_text=quote_text, quote_author=quote_author)


@app.route('/process_privacy', methods=['POST'])
def process_input_pP():
    print("processing PP")
    input_text = request.form['inputText']
    if input_text.startswith(" For docs"):
        return jsonify({'status': 'error', 'message': input_text})
    if input_text.startswith(" For .pdf"):
        return jsonify({'status': 'error', 'message': input_text})    
    if is_url(input_text):
        input_text = scrape_url(input_text)
        if input_text.startswith("Copy"):
           return jsonify({'status': 'error', 'message': input_text})    
    original_sentences = process_text(input_text)
    input_text = replace_abbreviations(input_text)
    z, y = index_sentences_with_formatting(input_text)
    x = [f[1] for f in y]
    if len(x) > 500:
        x = x[:500]
    x = [i for i in x if len(i) < 1200]
    # Process terms and store flags in session or a temporary storage
    flags1 = process_privacy_policy(x)
    flags1 = sorted(flags1, key=lambda x: x[2], reverse=True)
    modified_flags = []
    for sentence, original_sentence, probability in flags1:
        sentence_highlighted = highlight_phrases_pP(sentence)
        print('highlighting phrases')
        underlined_sentence = underline_terms_pP(sentence_highlighted)
        original_sentence_highlighted = highlight_phrases_pP(original_sentence)
        print('highlighting phrases originals')
        original_underlined_sentence = underline_terms_pP(original_sentence_highlighted)

        modified_tuple = (underlined_sentence, original_underlined_sentence, probability)

        modified_flags.append(modified_tuple)

    flags = modified_flags
    

    # Store in session for simplicity

    # Redirect to the results page
    session['flags'] = flags

    # Redirect to the function that will render the results
    return jsonify({'status': 'success'})


@app.route('/process_terms', methods=['POST'])
def process_input_Terms():
    print("processing Terms")
    input_text = request.form['inputText']
    if is_url(input_text):
        input_text = scrape_url(input_text)
        if input_text.startswith(" We didn"):
            return jsonify({'status': 'error', 'message': input_text})     
        if input_text.startswith("Link"):
            return jsonify({'status': 'error', 'message': input_text}) 
        if input_text.startswith("Copy"):
            return jsonify({'status': 'error', 'message': input_text})    
        if input_text.startswith(" For .pdf"):
            return jsonify({'status': 'error', 'message': input_text})
        if input_text.startswith(" For docs"):
            return jsonify({'status': 'error', 'message': input_text})
    original_sentences = process_text(input_text)
    input_text = replace_abbreviations(input_text)
    z, y = index_sentences_with_formatting(input_text)
    x = [f[1] for f in y]
    if len(x) > 500:
        x = x[:500]

    x = [i for i in x if len(i) < 1200]
    # Process terms and store flags in session or a temporary storage
    flags = process_Terms(x)
    flags = sorted(flags, key=lambda x: x[2], reverse=True)
   
    modified_flags = []
    for sentence, original_sentence, probability in flags:
        sentence_highlighted = highlight_phrases(sentence)
        underlined_sentence = underline_terms_Terms(sentence_highlighted)
        original_sentence_highlighted = highlight_phrases(original_sentence)
        original_underlined_sentence = underline_terms_Terms(original_sentence_highlighted)

        modified_tuple = (underlined_sentence, original_underlined_sentence, probability)

        modified_flags.append(modified_tuple)

    flags = modified_flags

    session['flags'] = flags

    # Redirect to the function that will render the results
    return jsonify({'status': 'success'})

@app.route('/show_results', methods=['GET'])
def show_results():
    # Retrieve flags from session
    flags = session.get('flags', [])

    if hasattr(current_user, 'is_premium'):
        is_premium = current_user.is_premium
    else:
        is_premium = False 

    # Render the results page with the flags
    return render_template('results.html', flags=flags, is_premium=is_premium)





@app.errorhandler(500)
def error():
    return jsonify({'status': 'error', 'message': "Their was something fishy with that link, copy and paste the policy into the searchbar instead."})






if __name__ == '__main__':
    pass
