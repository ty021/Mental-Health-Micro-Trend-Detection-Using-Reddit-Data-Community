import pandas as pd
import re
import glob
import os
import torch
import gspread
import gspread_dataframe as gd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langdetect import detect, LangDetectException
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import langid

# Configurations of my google sheet, data directory, and models
service_account_file = "/opt/airflow/config/tough-access-469416-a7-1cb6c85d7b7c.json"
google_sheet_name = "DTSC_5082_Clean_Data"
raw_data_dir_name = "/opt/airflow/RedditData"
classification_model = "NaveenTNS/mental-roberta"


## Data Preprocessing
def preprocess_text(text):
    
    ## If the text is not a string, we are converting it to string
    if not isinstance(text, str): 
        text = str(text)

    try:
        # If text is less than 3 characters (just abbreviations) or not in english language, we are removing them
        lang, score = langid.classify(text)
        if len(text) < 3 or lang != 'en':
            return None
    except LangDetectException:
        ## This exception is possible when the text doesn't have any text and have just emojis. Even in that case, we will return None
        return None
    
    ### Removing Gender specific information

    # he/she - they
    text = re.sub(r'\b(he|she)\b', 'they', text, flags=re.IGNORECASE)
    # him/her - them
    text = re.sub(r'\b(him|her)\b', 'them', text, flags=re.IGNORECASE)
    # his/hers -> their
    text = re.sub(r'\b(his|hers)\b', 'their', text, flags=re.IGNORECASE)
    # man/woman/boy/girl -> "person"
    text = re.sub(r'\b(man|woman|men|women|boy|girl|gentleman|lady|guy)\b', 'person', text, flags=re.IGNORECASE)
    
    # father/mother -> "parent", son/daughter -> "child"
    text = re.sub(r'\b(father|mother)\b', 'parent', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(son|daughter)\b', 'child', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(husband|wife)\b', 'spouse', text, flags=re.IGNORECASE)
    
    # Removing Nationalities. Replacing with "citizen"
    # I am using top-30 nationalities since they are most common. This can be extended as needed.
    nationality_regex = r'\b(american|indian|british|chinese|russian|german|french|japanese|canadian|australian|mexican|italian|spanish|korean|african|european|asian|arab|irish|scottish|dutch|swiss|swed|norwegian|danish|finnish|polish|ukrainian|brazilian|argentinian)\w*\b'
    text = re.sub(nationality_regex, 'citizen', text, flags=re.IGNORECASE)

    ## We are removing any links
    text = re.sub(r'http\S+|www\.\S+', '', text, flags = re.IGNORECASE)
    ## We are removing the mentions of users and subreddits to make text anonymous
    text = re.sub(r'u/\S+|r/\S+', '', text, flags = re.IGNORECASE)
    
    ## We are stripping any extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

## SENTIMENT ANALYSIS USING VADER

# Making sure vader_lexicon is downloaded. if not, we will download it.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

vader_analyzer = SentimentIntensityAnalyzer()

# Vader can handle virtually any length of text, so no chunking is needed unlike classification.
def find_sentiment_score_text(text):

    # If text is not a string, convert to string
    if not isinstance(text, str):
        text = str(text)
    
    # Calculate polarity scores
    ## Vader outputs a dictionary with 'neg', 'neu', 'pos', and 'compound' scores. We use the compound score in this project
    scores = vader_analyzer.polarity_scores(text)
    
    # Return the 'compound' score which is normalized between -1 (negative) and +1 (positive). 
    # The normalization dependes on the emojis, punctuation, and capitalization in the text.
    return scores['compound']


def main():
    
    # We are trying to connect to the google sheet first before any classification task.
    try:
        ## loading the service account file
        print("Authenticating with Google Sheets...")
        gc = gspread.service_account(filename=service_account_file)
        ## opening the google sheet
        sh = gc.open(google_sheet_name)
        print(f"Authenticated and opened Google Sheet: '{google_sheet_name}'")
    except Exception as e:
        print(f"Cannot connect to Google Sheets. Error: {e}")
        return
        
    try:
        ## Loading the classifier model from cache
        ## We will use GPU if available. So, device was being set dynamically. GPU = 0, CPU = -1
        print(f"Loading classifier: {classification_model} from cache...")
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

        ## Initializing the classification pipeline from cached model. local_files_only=True ensures we are loading from cache
        # Tokenizer is to convert text to tokens
        classifier_tokenizer = AutoTokenizer.from_pretrained(classification_model, local_files_only=True)
        # Model is to load the classification pipeline
        classifier_model = AutoModelForSequenceClassification.from_pretrained(classification_model, local_files_only=True)
        # Pipeline is streamline the process of tokenization, classification and output formatting
        classifier = pipeline("text-classification", model=classifier_model, tokenizer=classifier_tokenizer, device=device)
        print("Classifier loaded successfully")
        
    except Exception as e:
        print(f"Cannot load models from cache. Error: {e}")
        return


    ## We are creating a list of all raw data files to process 
    all_raw_files = glob.glob(os.path.join(raw_data_dir_name, "*_posts.csv"))
    print(f"Found {len(all_raw_files)} raw files to process.")

    for raw_file_path in all_raw_files:
        try:
            # Getting the csv file base name and creating a valid worksheet name
            base_name = os.path.basename(raw_file_path)
            worksheet_name = re.sub(r'[^A-Za-z0-9_]', '', base_name.replace('_posts.csv', ''))[:100]
            print(f"Processing: {base_name} -> Worksheet: '{worksheet_name}'")
            
            # First we try to load the worksheet. It it exists, We will load the data from google worksheet and get already-processed post IDs. 
            try:
                worksheet = sh.worksheet(worksheet_name)
                print("  -> Found existing worksheet.")
                old_data = gd.get_as_dataframe(worksheet)
                processed_ids = set(old_data['id'])
                print(f"  -> Found {len(processed_ids)} already-processed posts.")
            
            ## If the worksheet doesn't exist, it will throw an exception. In that case, we will create a new worksheet.
            except Exception as e:
                print(f"  -> Worksheet not found or empty. Will create. Error: {e}")
                worksheet = sh.add_worksheet(title=worksheet_name, rows=100, cols=20)
                old_data = pd.DataFrame()
                processed_ids = set()

            ## Loading the raw data file
            df = pd.read_csv(raw_file_path)
            ## If the dataframe is empty, we will give a log message and skip to the next subreddit
            if df.empty:
                print("  -> Raw data file is empty. Skipping.")
                continue

            # Next, we will filter out posts whcih are already processed (using processed_ids).
            df_new_posts = df[~df['id'].isin(processed_ids)].copy()
            
            ## If there are no new posts to process, we will give a log message and skip to the next subreddit
            if df_new_posts.empty:
                print("  -> No new posts to process. Skipping.")
                continue
            ## Else, we will log the number of new posts to process
            print(f"  -> Found {len(df_new_posts)} new posts to process.")
            
            # First, we will create 'full_text' by combining 'title' and 'text' columns. If it is missing, we will fill with empty string
            df_new_posts['full_text'] = df_new_posts['title'].fillna('') + ' ' + df_new_posts['text'].fillna('')
            ## Out of the new posts, we will drop any duplicate IDs if present
            df_new_posts = df_new_posts.drop_duplicates(subset=['id'])
            
            # We are creating a new column 'clean_text' by applying the preprocess_text function.
            df_new_posts['clean_text'] = df_new_posts['full_text'].apply(preprocess_text)
            
            ## Dropping any posts where 'clean_text' is None
            texts_to_classify = df_new_posts['clean_text'].dropna().tolist()
            if not texts_to_classify:
                print(f"  -> No text to classify in new posts. Skipping.")
                continue
            
            # CLASSIFICATION
            print(f"  -> Classifying {len(texts_to_classify)} posts...")
            results_classify = classifier(texts_to_classify, padding=True, truncation=True)
            
            ## Converting the result to a dataframe and adding 'prediction_label' and 'prediction_score' columns to our df_new_posts
            df_results = pd.DataFrame(results_classify, index=df_new_posts['clean_text'].dropna().index)
            df_new_posts['prediction_label'] = df_results['label']
            df_new_posts['prediction_score'] = df_results['score']
            
            # SENTIMENT ANALYSIS
            print(f"  -> Running sentiment analysis on {len(df_new_posts)} new posts...")
            # We use 'full_text' for sentiment, as it's the raw, un-cleaned text
            df_new_posts['sentiment_score'] = df_new_posts['clean_text'].apply(find_sentiment_score_text)

            # Extracting only the relevant columns to upload. We left out 'title', 'text', and 'full_text'
            final_df_new = df_new_posts[[
                'id', 'created_utc', 'author', 'clean_text', 
                'prediction_label', 'prediction_score', 'sentiment_score'
            ]]
            
            # Concatenating old data with new data
            combined_df = pd.concat([old_data, final_df_new], ignore_index=True)
            
            print(f"  -> Uploading {len(combined_df)} total rows ({len(final_df_new)} new) to '{worksheet_name}'...")
            
            ## Before uploading, we will clear the existing worksheet since we are uploading the entire combined dataframe
            worksheet.clear()
            df_filled = combined_df.fillna("")
            gd.set_with_dataframe(worksheet, df_filled, include_index=False, include_column_header=True)
            print(f"  -> Successfully uploaded data.")
        
        except Exception as e:
            print(f"  -> FAILED to process {base_name}. Error: {e}")

if __name__ == "__main__":
    main()