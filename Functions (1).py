import pandas as pd
import gspread
import gspread_dataframe as gd
import streamlit as st
import numpy as np
import re
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import praw
from datetime import datetime, timezone
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import langid
from llama_cpp import Llama
import torch
import shap
import torch.nn.functional as F

# Looking for existing vader lexicon for sentiment analysis. Else, we are just downloading it.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# This is the function used to obtain VADER sentiment scores
vader_analyzer = SentimentIntensityAnalyzer()

# This is the function to connect to the google sheets and load data
@st.cache_data(ttl="1h", max_entries=1)
def load_data_from_gsheet(sheet_name, worksheet_name):
    try:
        # Connecting to google sheets usually need a long set of credentials. THese usually come in a JSON file. 
        # But, we cannot share that JSON file publicly. So, we just stored these in an .env file and are reading from there.
        creds_json = {
            "type": os.environ.get("GCP_TYPE"),
            "project_id": os.environ.get("GCP_PROJECT_ID"),
            "private_key_id": os.environ.get("GCP_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("GCP_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.environ.get("GCP_CLIENT_EMAIL"),
            "client_id": os.environ.get("GCP_CLIENT_ID"),
            "auth_uri": os.environ.get("GCP_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": os.environ.get("GCP_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": os.environ.get("GCP_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
            "client_x509_cert_url": os.environ.get("GCP_CLIENT_X509_CERT_URL")
        }
        
        # Connecting to the google service
        google_handle = gspread.service_account_from_dict(creds_json)
        # Opening the google sheet
        sheet_handle = google_handle.open(sheet_name)
        # Opening the worksheet 
        worksheet = sheet_handle.worksheet(worksheet_name)
        
        # Converting the worksheet to a pandas dataframe
        df = gd.get_as_dataframe(worksheet)
        # The dataset has author, id, prediction_score as well. These are initially created to eliminate duplicates. We dont want to load that personal information
        df = df.dropna(subset=['id', 'created_utc', 'clean_text', 'sentiment_score'])
        # Chanding the date to pandas datetime format
        df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
        # Remove the bad rows immediately here
        df = df.dropna(subset=['created_utc'])
        # Ensuring sentiment score is numeric
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
        return df
    except Exception as e:
        # If there is any error, we are returning an empty dataframe and display a message
        st.error(f"Error loading data from Google Sheets: {e}")
        return pd.DataFrame()
    
@st.cache_resource
def load_topic_modeling_models():

    # Loading a sentence transformers to convert text to numbers.
    # We need this since KMeans cannot cluster texts
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
    filename = "Phi-3-mini-4k-instruct-q4.gguf"
    # Loading a GGUF file enables us to run the model on CPU
    try:
        # Loading the LLM from gguf file
        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=4096,   # token size
            n_threads=os.cpu_count() - 2, # Using all cores available except for one
            verbose=False
        )
        return embedder, llm
    except Exception as e:
        st.error(f"Failed to load GGUF model: {e}")
        return None, None

# This is the function to create z-scores and rolling averages for sentiment over time
def analyze_sentiment_over_time(df, classification_filter=None):
    if df.empty:
        # If the dataframe is empty, we just return an empty dataframe
        return pd.DataFrame()
    
    # If the classification filter is provided, we filter the dataframe
    if classification_filter:
        df_filtered = df[df['prediction_label'].isin(classification_filter)].copy()
    else:
        # Else we just retuen the complete dataframe
        df_filtered = df.copy()

    # After filtering, the dataframe might get empty. In that case, we just return an empty dataframe
    if df_filtered.empty: 
        return pd.DataFrame()
    
    # We are creating a new dataframe by grouping entire data to daily level. We calculate the mean of the sentiment score for each day.
    df_daily = df_filtered.set_index('created_utc')['sentiment_score'].resample('D').mean().to_frame(name='distress_score').dropna()
    # Right now, the sentiment scores -ve indicates distress. So, we flip the scores as well
    df_daily['distress_score'] = -df_daily['distress_score']
    # We also calculate the total number of posts for each day and append it to the new daily dataframe
    df_daily['daily_volume'] = (df_filtered.set_index('created_utc').resample('D').size())
    # We calculate distress energy to capture both high volume and high distress score.
    df_daily['distress_energy'] = df_daily['distress_score'] * df_daily['daily_volume']

    WINDOW = 7

    # We are creating rolling mean and std deviation for sentiment and volume over a 7-day window
    df_daily['distress_mean'] = df_daily['distress_score'].rolling(WINDOW, min_periods=1).mean()
    df_daily['distress_std'] = df_daily['distress_score'].rolling(WINDOW, min_periods=1).std()
    df_daily['distress_z'] = (df_daily['distress_score'] - df_daily['distress_mean']) / df_daily['distress_std']

    # rolling mean/std for volume
    df_daily['volume_mean'] = df_daily['daily_volume'].rolling(WINDOW, min_periods=1).mean()
    df_daily['volume_std'] = df_daily['daily_volume'].rolling(WINDOW, min_periods=1).std()
    df_daily['volume_z'] = (df_daily['daily_volume'] - df_daily['volume_mean']) / df_daily['volume_std']    
    

    # I am also calculating a multiplication of both volume and sentiment z-scores to identify days with both high volume and high sentiment deviation
    df_daily['energy_mean'] = df_daily['distress_energy'].rolling(WINDOW, min_periods=1).mean()
    df_daily['energy_std'] = df_daily['distress_energy'].rolling(WINDOW, min_periods=1).mean().std()
    df_daily['energy_z'] = (df_daily['distress_energy'] - df_daily['energy_mean']) / df_daily['energy_std']  

    return df_daily.reset_index()


## After running the topic modeling for multiple times, we identified these words are popping up in the top-15 words.
# These words doesn't add any meaning to our analysis. So, we are adding them to the list of stop words as well
extra_stopwords = set("""
don didn doesnt wasnt wouldnt couldnt shouldnt mightnt mustnt wont cant isnt arent wasnt havent ive youre
thing things something anything everything nothing maybe someone anyone everyone nobody people
like just really know think want got get gonna feel feels feeling felt make makes made try trying tried sure need 
see say said even much still back way well one also still never lot even much every anything anywhere anyone nobody somebody everybody
life day days time years months today tonight tomorrow yesterday now then always sometimes often usually
mental health help issue issues problem problems okay ok good bad better worse
feel feelings felt feeling really kinda sorta probably maybe maybe honestly literally actually
anxiety anxious depression depressed
""".split())

stopwords = list(text.ENGLISH_STOP_WORDS.union(extra_stopwords))

# Cleaning the text further more because, SLM doesn't need to know about emojis and punctuations. Also, lowering the text will reduce the vector size significantly.
def clean_text_for_topic_model(txt):
    if not isinstance(txt, str): 
        return ""
    txt = txt.lower()
    # Remove anything apart from lower case letters and spaces
    txt = re.sub(r"[^a-z\s]", " ", txt)
    # Trimming extra spaces
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# Running LLM to extract reasons
def get_top_reasons_for_day(daily_posts, selected_labels, embedder, llm, n_clusters=2):
    # If there are no posts, it will return a message
    if not daily_posts: 
        return ["No posts to analyze."]
    
    # We apply additional cleaning to remove any punctuations, emojis etc
    cleaned_texts = [clean_text_for_topic_model(t) for t in daily_posts]
    # Next we use the embedder to encode the text
    embeddings = embedder.encode(cleaned_texts, show_progress_bar=False)
    
    # Determining the number of clusters.
    # If there are less than 0 posts, we stop the exec
    if len(cleaned_texts)==0:
        return ["Not enough data."]
    # If there are less text than the user given cluster size, we just set the cluster size to the length of posts
    elif len(cleaned_texts)<n_clusters:
        n_clusters = len(cleaned_texts)
    # else, we just leave the cluster size to the user input

    # We run KMeans clustering to cluster the posts
    kmeans = KMeans(n_clusters=n_clusters, random_state=264, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # This will take all the text and find dictionary of 1000 most common words. 
    # A vector will be formed for each post based on the frequency of words in a post. The index of each word in the vector is always same since these are dependent on the dictionary we already formed
    # we are taking 1-grams (single words) and 2-grams (2 words at a time to not miss the context)
    vectorizer = CountVectorizer(stop_words=stopwords, max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(cleaned_texts)

    # It saves the actual words (like "exam", "fail", "money") into a list.
    # The vector only knows " a particular Column has a count of n." You but we need these terms list to know that Column is actually the word "professor." etc.
    terms = np.array(vectorizer.get_feature_names_out())

    top_reasons = []
    
    # Combining if multiple labels are selected
    label_str = ", ".join(selected_labels)
    
    # Next we extract top-15 words in the post based on the frequency
    for cluster_id in range(n_clusters):
        # isolating the cluster
        idx = np.where(cluster_labels == cluster_id)[0]
        # If there are no words, we just continue
        if len(idx) == 0: continue
            
        cluster_word_counts = X[idx, :].sum(axis=0)
        top_keyword_indices = np.argsort(np.array(cluster_word_counts).flatten())[-15:]
        top_keywords = terms[top_keyword_indices][::-1] 
        # Combining all the keywords into a string after extraction
        top_keywords_str = ", ".join(top_keywords)

        # Constructs the structured LLM prompt by defining an 'Expert Analyst' persona and injecting extracted cluster keywords.
        # Enforces a strict 'Title/Explanation' output schema to ensure the generated analysis can be reliably parsed for the UI.
        messages = [
            {
                "role": "system",
                "content": f"You are an expert psychological analyst. The text data belongs to the class: \"{label_str}\"."
            },
            {
                "role": "user",
                "content": f"""Analyze the following keywords extracted from a cluster of Reddit posts:
                "{top_keywords_str}"

                Based *only* on these keywords, identify the single most likely root cause of distress.
                Do NOT provide examples. Provide a concise Theme Title and a 2-sentence Explanation.

                Format your response exactly like this:
                Theme Title: [Insert Title]
                Explanation: [Insert Explanation]"""
            }
        ]
        
        #Prompting the SLM to generate output.
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=150,    # Restricting the token size will give us limited output
            temperature=0.1,   # Lower temp will make the model less creative and prompt the model to use words in the actual posts
            top_p=0.95,
            repeat_penalty=1.1 # We will peanize the model for any repeated words and reasoning
        )
        
        response_text = output['choices'][0]['message']['content']
        
        # Remove any extra spaces from the response
        cleaned_response = response_text.strip()
        
        # If the model forgets to type "Theme Title:", we add it back
        if not cleaned_response.startswith("Theme Title:"):
            # Sometimes models just output the title first
            cleaned_response = "Theme Title: " + cleaned_response
        
        # This response is generated for only 1 cluster. We repeat it for other clusters as well
        top_reasons.append(cleaned_response)

    return top_reasons


# Loading the model for our SHAP
@st.cache_resource
def load_shap_model(model_name="NaveenTNS/distilbert-mental-health-classifier"):
    # Tokenizer of our model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Classification model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    # Class names to display at the top of the SHAP (clickable)
    class_names = list(model.config.id2label.values()) if hasattr(model.config, "id2label") else None
    return tokenizer, model, class_names

def run_shap_explanation(text, tokenizer, model, class_names):

    def prediction_function(text_array):
        inputs = tokenizer(text_array.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()

    explainer = shap.Explainer(prediction_function, tokenizer)
    shap_values = explainer([text])

    if class_names:
        shap_values.output_names = class_names

    raw_html = shap.plots.text(shap_values[0], display=False)
    
    custom_css = """
    <style>
        /* Match Streamlit's Font */
        body {
            font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #31333F;
            background-color: #ffffff; /* SHAP needs white bg for contrast */
        }
        
        /* Target the SVG Text elements (the class labels at the top) */
        svg text {
            cursor: pointer !important; /* Force pointer cursor */
            font-weight: 600 !important; /* Make them bolder */
            fill: #555 !important; /* Darker grey for better visibility */
        }
        
        /* Add hover effect to the labels */
        svg text:hover {
            fill: #000 !important;
            font-weight: 800 !important;
            text-decoration: underline;
        }
        
        /* Adjust the main container padding */
        .shap-visual-container {
            padding: 10px;
        }
    </style>
    """
    
    # Combine CSS + HTML
    styled_html = custom_css + raw_html
    
    return styled_html


## Page-2 Data Extraction

def fetch_reddit_posts_live(subreddit_name, client_id, client_secret, user_agent):
    # Converting the subreddit name to lower since all the subreddit names are saved with lower case in API
    subreddit_name = subreddit_name.strip().lower()

    # Loading a reddit handle
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    posts = []
    try:
        # We are retrieveing posts inside try block since the subreddit might not exist and throw an error.
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.new(limit=1000):
            posts.append({
                "id": post.id,
                "title": post.title,
                "text": post.selftext,
                "author": str(post.author) if post.author else "[deleted]",
                "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            })
    
    # We return (data, error) tuple
    except Exception as e:
        return None, f"Failed to fetch subreddit: {e}"

    return pd.DataFrame(posts), None

# Loading classification model
@st.cache_resource
def load_page2_models():
    # Tokenizer
    classifier_tokenizer = AutoTokenizer.from_pretrained("NaveenTNS/distilbert-mental-health-classifier")
    # Classifier Model
    classifier_model = AutoModelForSequenceClassification.from_pretrained("NaveenTNS/distilbert-mental-health-classifier")
    # Classification pipeline
    classifier = pipeline("text-classification", model=classifier_model, tokenizer=classifier_tokenizer)

    return classifier, classifier_tokenizer

# Preprocessing pipeline on live data
def preprocess_text_page2(text: str) -> str:
    
    ## If the text is not a string, we are converting it to string
    if not isinstance(text, str): text = str(text)

    # langid returns a tuple ('en', -54.4)
    lang, score = langid.classify(text)
    
    if len(text) < 3 or lang != 'en':
        return None
    
    ### Removing Gender specific information

    # he/she -> they
    text = re.sub(r'\b(he|she)\b', 'they', text, flags = re.IGNORECASE)
    # him/her -> them
    text = re.sub(r'\b(him|her)\b', 'them', text, flags = re.IGNORECASE)
    # his/hers -> their
    text = re.sub(r'\b(his|hers)\b', 'their', text, flags = re.IGNORECASE)
    # man/woman/boy/girl -> "person"
    text = re.sub(r'\b(man|woman|men|women|boy|girl|gentleman|lady|guy)\b', 'person', text, flags = re.IGNORECASE)
    
    # father/mother -> "parent", son/daughter -> "child"
    text = re.sub(r'\b(father|mother)\b', 'parent', text, flags = re.IGNORECASE)
    text = re.sub(r'\b(son|daughter)\b', 'child', text, flags = re.IGNORECASE)
    text = re.sub(r'\b(husband|wife)\b', 'spouse', text, flags = re.IGNORECASE)
    
    # Removing Nationalities. Replacing with "citizen"
    # I am using top-30 nationalities since they are most common. This can be extended as needed.
    nationality_regex = r'\b(american|indian|british|chinese|russian|german|french|japanese|canadian|australian|mexican|italian|spanish|korean|african|european|asian|arab|irish|scottish|dutch|swiss|swed|norwegian|danish|finnish|polish|ukrainian|brazilian|argentinian)\w*\b'
    text = re.sub(nationality_regex, 'citizen', text, flags = re.IGNORECASE)

    ## We are removing any links
    text = re.sub(r'http\S+|www\.\S+', '', text, flags = re.IGNORECASE)
    ## We are removing the mentions of users and subreddits to make text anonymous
    text = re.sub(r'u/\S+|r/\S+', '', text, flags = re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_sentiment_score_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    # Calculate polarity scores
    scores = vader_analyzer.polarity_scores(text)
    
    # Return the compound score which is normalized between -1 (negative) and +1 (positive). 
    return scores['compound']

def classify_posts_page2(df, classifier):
    # Forming full text by merging title and text
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    # Getting clean text by passing full text through preprocessing pipeline
    df["clean_text"] = df["full_text"].apply(preprocess_text_page2)
    # Removing any null values
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)

    # Classify the posts
    results_class = classifier(df["clean_text"].tolist(), truncation=True)
    df["prediction_label"] = [r["label"] for r in results_class]
    df["prediction_score"] = [r["score"] for r in results_class]

    # Obtain sentiment scores
    df["sentiment_score"] = df["clean_text"].apply(lambda t: 
        find_sentiment_score_text(t)
    )

    return df



