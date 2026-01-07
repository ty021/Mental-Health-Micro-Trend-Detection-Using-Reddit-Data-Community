import praw
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

load_dotenv()

def get_and_save_reddit_posts(client_id, client_secret, user_agent, subreddit_name, days, csv_filepath):

    # Getting the ids of already saved posts to avoid duplicates
    existing_ids = set()
    file_exists = os.path.exists(csv_filepath)
    
    if file_exists:
        try:
            # Read the existing CSV to get the IDs of posts already saved
            existing_df = pd.read_csv(csv_filepath)
            # Ensure the 'id' column exists before creating the set
            if 'id' in existing_df.columns:
                existing_ids = set(existing_df['id'])
        except pd.errors.EmptyDataError: ## I am focusing on only one error because other errors should not trigger the creation of a new file
            file_exists = False # We are setting that the file doesnt exist

    # Connecting to Reddit API and fetch posts
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    # Creating handle to the subreddit
    subreddit = reddit.subreddit(subreddit_name)
    # We are creating the time value by subtracting 'days' from current time. This is the time after which we want posts
    time_limit = datetime.now(timezone.utc) - timedelta(days=days)

    new_posts = []

    # The reddit API limit is 1000 most recent posts anyway
    for post in subreddit.new(limit=1000):
        # We are creating a pandas datetime object from the post's created_utc timestamp
        post_time = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
        
        # Process post only if it's within the timeframe AND its ID isn't already in our set
        if post_time >= time_limit and post.id not in existing_ids:
            new_posts.append({
                "id": post.id,  # Needed for checking duplicates
                "title": post.title,
                "text": post.selftext,
                "author": str(post.author) if post.author else "[deleted]",  ## We are collecting it just if we need to analyze author behavior later. Because, collecting this later on the fly is not possible
                "created_utc": post_time.isoformat()
            })

    df_new = pd.DataFrame(new_posts)

    # Append to the CSV. The header is written only if the file didn't exist before.
    df_new.to_csv(csv_filepath, mode='a', header=not file_exists, index=False)

## Directory where the data should be stored
output_dir = "/opt/airflow/RedditData"
## List of subreddits to extract data from
mental_subreddits = ["mentalhealth", "Anxiety", "depression", "selfhelp", "SuicideWatch", "bipolar", 
                     "TalkTherapy", "askatherapist", "jobs", "recruitinghell", "cscareerquestions", 
                     "college", "GradSchool", "careerguidance"]

# client_id = os.getenv("CLIENT_ID")
# client_secret = os.getenv("CLIENT_SECRET")
# user_agent = os.getenv("USER_AGENT")
# I was not able to get the env variables working in airflow, so hardcoding for now

client_id = "AW-xfEQ5Ey_DBGJ_DcIWzQ"
client_secret = "eCFxLD8C7i2KOsLaNl8Ptt7Ul9gOzA"
user_agent = "myredditbot/0.1 by NaveenSaiTNS"

# Iterating through each subreddit and extracting posts from the last 1 day since this script will run everyday
for subreddit in mental_subreddits:
    csv_file = f"{subreddit}_posts.csv"
    full_path = os.path.join(output_dir, csv_file)
    get_and_save_reddit_posts(client_id, client_secret, user_agent, subreddit, days=1, csv_filepath=full_path)