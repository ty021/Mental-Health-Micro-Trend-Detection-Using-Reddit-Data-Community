# Importing Required Libraries
import streamlit as st
import pandas as pd
import altair as alt
import Functions as fn
from dotenv import load_dotenv
import os
# Required for rendering SHAP HTML
import streamlit.components.v1 as components  

# Loading environmental variables
load_dotenv()

# SESSION STATE VARIABLES

# The first time the app loads, we set the page to be "dashboard"
# This state variable is the main router for our multipage app
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# Page-1 Variables - These are useful when we switch between pages to retain the state

# First,we track subreddit of the dashboard. By default, it is mentalhealth
if "dashboard_subreddit" not in st.session_state:
    st.session_state.dashboard_subreddit = "mentalhealth"
# This tracks the root cause analysis results. It is none when the app starts
if "rca_results" not in st.session_state:
    st.session_state.rca_results = None
# This tracks the previous subreddit. This is used to see if there is a difference from dashboard_subreddit. If there is, we will just run RCA again
if "prev_subreddit" not in st.session_state:
    st.session_state.prev_subreddit = None

# PAGE-2 state variables
if "df_page2" not in st.session_state:
    st.session_state.df_page2 = None
# This saves the page-2 RCA results
if "page2_rca" not in st.session_state:
    st.session_state.page2_rca = None
# This saves the previous subreddit for page-2
if "page2_prev_subreddit" not in st.session_state:
    st.session_state.page2_prev_subreddit = None
# This saves the current subreddit input for page-2. The text input. By default, it is mentalhealth
if "page2_current_input" not in st.session_state:
    st.session_state.page2_current_input = "mentalhealth"

## Our first page has 14 subreddits. We defined the names of all subreddits here
subreddit_list = [
    "Anxiety","askatherapist","bipolar","careerguidance","college",
    "cscareerquestions","depression","GradSchool","jobs","mentalhealth",
    "recruitinghell","selfhelp","SuicideWatch","TalkTherapy"
]

## Name of our google sheet which holds the cleaned data
sheet_name = "DTSC_5082_Clean_Data"


# SUPPORTING FUNCTIONS

# This function will set the page state variable to the name of the page 'dashboard' or 'page2'
def set_page(p):
    st.session_state.page = p

# Function to Plot Time Series Graphs. We have 6 plots. So, this is creating a lot of redundancy
def plot_anomaly_chart(data, metric_col, lower_band_col, upper_band_col, anomaly_col, z_score_col, title, y_axis_title, line_color):
    # In altair, we pass dataset only with the alt.Chart() function. Then we just reference the column names everywhere else
    # Chart Header
    st.subheader(title)
    
    # Creating the base chart with x-axis and tooltip
    base = alt.Chart(data).encode(
        x=alt.X("created_utc:T", axis=alt.Axis(title="Date")),
        tooltip=["created_utc:T", f"{metric_col}:Q", f"{z_score_col}:Q"]
    )
    
    # THe band area between upper and lower bands. This is shaded light gray
    band = base.mark_area(opacity=0.15, color="lightgray").encode(
        y=alt.Y(f"{lower_band_col}:Q", axis=alt.Axis(title=y_axis_title)), 
        y2=f"{upper_band_col}:Q"
    )
    
    # The main line showing distress score over time
    line = base.mark_line(color=line_color).encode(y=f"{metric_col}:Q")
    
    # The red dots showing anomalies
    spike = base.mark_point(size=110, filled=True).encode(
        y=f"{metric_col}:Q",
        # We use f"datum.{anomaly_col}" so Altair knows which column to check for True/False
        color=alt.condition(f"datum.{anomaly_col}", alt.value("red"), alt.value("gray")),
        opacity=alt.condition(f"datum.{anomaly_col}", alt.value(1), alt.value(0))
    )
    
    # Rendering the combined chart
    st.altair_chart((band + line + spike).interactive(), use_container_width=True)

# Function to format RCA results
def display_rca_results(current_rca):
    # Heading
    st.subheader("Root Cause Analysis")
    # If current_rca state variable was set, it has data and we can show it
    if current_rca:
        # For each theme found
        for i, r in enumerate(current_rca):
            # We are asking the SLM to format the output as "Theme Title: <title> Explanation: <explanation>"
            # If the output contains Explanation:, we split it to get title and explanation separately
            if "Explanation:" in r:
                # Splitting at explanation will give theme title: <title> and <explanation> separately
                title_part, content = r.split("Explanation:", 1)
                # We are removing the "Theme Title:" part to get only the title
                title = title_part.replace("Theme Title:", "").strip()
                explanation = content.strip()
            else:
                # If the output doesn't contain Explanation:, it means the SLM didn't follow instructions properly. So, we just set the theme title to theme-1/2
                # or, there are no posts to analyze
                # and print the rest of the output
                title, explanation = f"Theme {i+1}", r

            # Write title in expander and show explanation inside it
            with st.expander(f"Theme {i+1}: {title}", expanded=(i == 0)):
                st.write(explanation)

## We are defining a function to invoking the load_data_from_gsheet function from Functions.py to load data from google sheet and caching the results for performance
## Caching will not run this function again unless the parameters change
@st.cache_data(show_spinner=False)
def load_data_cached(sheet_name, subreddit):
    return fn.load_data_from_gsheet(sheet_name, subreddit)

# We are defining a function to invoking the get_top_reasons_for_day function from Functions.py to run SLM based RCA and caching the results for performance
@st.cache_data(show_spinner=False)
def run_llm_rca(posts, labels):
    return fn.get_top_reasons_for_day(
        daily_posts=posts,
        selected_labels=labels,
        embedder=embedder,
        llm=llm_pipeline,
        n_clusters=2
    )

# We are loading the models for RCA and SHAP explanation only once here.
# We are saving the models to global variables so that they can be reused in the RCA and SHAP functions
global embedder, llm_pipeline, shap_tokenizer, shap_model, shap_class_names

# For RCA
embedder, llm_pipeline = fn.load_topic_modeling_models()
# For SHAP
shap_tokenizer, shap_model, shap_class_names = fn.load_shap_model()


#  PAGE LAYOUT


# Setting page title and width
st.set_page_config(page_title="Mental Health Micro-Trend Detection", layout="wide")

## The main content is starting 5 rem below the header by default. We will reduce that to 2 rem to make better use of space.
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


## Setting the title of the page
st.markdown(
    "<h1 style='text-align:center;margin-bottom:4rem;'>Mental Health Micro-Trend Detection - Community Level</h1>",
    unsafe_allow_html=True
)
## Dividing the page into 4 columns to create navigation buttons in the 2nd and 3rd columns
c1, c2, c3, c4 = st.columns([1.5, 2, 2, 1.5])
## Creating a button in the 2nd column
with c2:
    st.button(
        "Trend Dashboard",
        # If session state is first page, this button is highlighted
        type="primary" if st.session_state.page == "dashboard" else "secondary",
        use_container_width=True,
        ## If we click this button, we invoke the set_page function to set the page to first page
        on_click=set_page,
        args=("dashboard",)
    )
with c3:
    st.button(
        "Live Reddit Analysis",
        type="primary" if st.session_state.page == "page2" else "secondary",
        use_container_width=True,
        on_click=set_page,
        args=("page2",)
    )

# THis creates a divider line below the navigation buttons
st.divider()


#############    PAGE-1 (USING GOOGLE SHEET DATA)    #############

## Defines the dashboard page layout and logic for the selected subreddit
def dashboard_page(df, selected_subreddit, *, sidebar_status):

    #### SIDEBAR ####
    # A dropdown to select the subreddit
    selected_subreddit = st.sidebar.selectbox(
        "Subreddit",
        options=subreddit_list,
        ## This will select the default subreddit which I set (mentalhealth) in the dashboard_subreddit session state variable
        index=subreddit_list.index(st.session_state.dashboard_subreddit)
    )
    ## Once the user selects a subreddit, we update the session state variable to hold the current subreddit.
    ## This is useful when user makes switches between the pages
    st.session_state.dashboard_subreddit = selected_subreddit

    ## If the dataframe is empty after removing invalid timestamps, we show an error in the log screen to the left top and stop the execution
    if df.empty:
        sidebar_status.error("No valid timestamps in dataset.")
        return

    # Category dropdown
    available_labels = sorted(df["prediction_label"].unique().tolist())
    ## I am removing Normal label since it is not the focus of our analysis
    available_labels = [x for x in available_labels if x != "Normal"]
    selected_label = st.sidebar.selectbox(
        "Category",
        options=available_labels,
        ## Here, I wanted to set Anxiety as the default selected label. If no post was classified to be anxiety in the data, then I am setting it to the first label.
        index=available_labels.index("Anxiety") if "Anxiety" in available_labels else 0
    )

    # anomaly slider (to adjust the z-score value to be considered as an anomaly. Higher means fewer anomalies)
    anomaly_threshold = st.sidebar.slider(
        "Anomaly Sensitivity",
        # Starts from 1.0 to 3.0 with default 2. 0.1 increments
        1.0, 3.0, 2.0, 0.1
    )

    # date range selector (safe defaults)
    ## These are the default dates for the date selector
    start_default = df["created_utc"].min().date()
    end_default = df["created_utc"].max().date()

    # Date selector widget
    selected_dates = st.sidebar.date_input(
        "Select date range",
        value=(start_default, end_default)
    )

    # Checking if the user selected 2 valid dates or not. 
    # If now, we show a warning in the sidebar log and stop execution
    valid_range = isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2
    if not valid_range:
        sidebar_status.warning("Select two valid dates.")
        return
    
    # Unpack selected dates
    start_date, end_date = selected_dates

    st.sidebar.markdown("---")

    # Button to perform RCA. This is to make sure that RCA is not run on every change of the controls above
    ## I want to align this button to center and make it secondary type
    _, col2, _ = st.sidebar.columns([1,6,1])
    with col2:
        run_rca_button = st.button("Find Root Cause", type="primary")

    ## FILTERING POSTS BASED ON THE SELECTED FILTERS

    ## We are not changing the time series analysis since the graph was already interactive
    ## Instead of selecting posts directly, we are creating a mask and updating it everytime use selects a different range.
    mask = (
        (df["created_utc"].dt.date >= start_date) &
        (df["created_utc"].dt.date <= end_date)
    )
    ## Selecting posts according to the mask
    range_df = df[mask]
    ## Then we are selecting the posts which are classified to be the selected_label
    range_df = range_df[range_df["prediction_label"] == selected_label]
    posts_to_analyze = range_df["clean_text"].tolist()


    ## Running the Root Cause Analysis

    ## If the rca_results variable (initially set to None when the website loads) or prev_subreddit not equal to selected_subreddit, we will run the function automatically
    if (st.session_state.rca_results is None) or \
       (st.session_state.prev_subreddit != selected_subreddit):

        if posts_to_analyze:
            sidebar_status.info("Running Topic Modeling")
            ## Storing results to a session variable so that even if we switch pages, the results persist.
            st.session_state.rca_results = run_llm_rca(posts_to_analyze, [selected_label])
            sidebar_status.success("Done")
        else:
            st.session_state.rca_results = []

        ## Once ran successfully, set the prev_subreddit to be currently selected sub_reddit.
        st.session_state.prev_subreddit = selected_subreddit

    ## We can also run RCA when the user clicks the button
    if run_rca_button:
        if posts_to_analyze:
            sidebar_status.info("Running Topic Modeling")
            st.session_state.rca_results = run_llm_rca(posts_to_analyze, [selected_label])
            sidebar_status.success("Done")
        else:
            sidebar_status.warning("No posts found")

    current_rca = st.session_state.rca_results

    ###### TIME SERIES ANALYSIS ######

    ## We are running a function to filter the data according to the classification selected, find distress score and other scores like rolling mean, rolling std, z-score etc.
    df_daily = fn.analyze_sentiment_over_time(
        df[df["prediction_label"] == selected_label],
        [selected_label]
    )

    ## Marking anomalies based on the selected threshold
    df_daily["distress_is_anomaly"] = (df_daily["distress_z"] > anomaly_threshold)
    df_daily["volume_is_anomaly"] = (df_daily["volume_z"] > anomaly_threshold)
    df_daily["energy_is_anomaly"] = (df_daily["energy_z"] > anomaly_threshold)

    ## Calculating upper and lower bands for visualization
    ## These bands are (anomaly_threshold * standard deviations) away from the rolling mean. Usually, the outliers fall outside these bands. This is just for the visualization purpose
    df_daily["distress_upper_band"] = df_daily["distress_mean"] + anomaly_threshold * df_daily["distress_std"]
    df_daily["distress_lower_band"] = df_daily["distress_mean"] - anomaly_threshold * df_daily["distress_std"]

    df_daily["volume_upper_band"] = df_daily["volume_mean"] + anomaly_threshold * df_daily["volume_std"]
    df_daily["volume_lower_band"] = df_daily["volume_mean"] - anomaly_threshold * df_daily["volume_std"]

    df_daily["energy_upper_band"] = df_daily["energy_mean"] + anomaly_threshold * df_daily["energy_std"]
    df_daily["energy_lower_band"] = df_daily["energy_mean"] - anomaly_threshold * df_daily["energy_std"]

    # We are dividing the page into 2 columns.
    q1, q2 = st.columns(2)
    q3, q4 = st.columns(2)

    # Q1 Time Series on Distress
    with q1:
        plot_anomaly_chart(
            data=df_daily,
            metric_col="distress_score",
            lower_band_col="distress_lower_band",
            upper_band_col="distress_upper_band",
            anomaly_col="distress_is_anomaly",
            z_score_col="distress_z",
            title="Distress Analysis",
            y_axis_title="Distress Score",
            line_color="blue"
        )

    with q2:
        plot_anomaly_chart(
            data=df_daily,
            metric_col="daily_volume",
            lower_band_col="volume_lower_band",
            upper_band_col="volume_upper_band",
            anomaly_col="volume_is_anomaly",
            z_score_col="volume_z",
            title="Volume Analysis",
            y_axis_title="Number of Posts",
            line_color="green"
        )

    with q3:
        plot_anomaly_chart(
            data=df_daily,
            metric_col="distress_energy",
            lower_band_col="energy_lower_band",
            upper_band_col="energy_upper_band",
            anomaly_col="energy_is_anomaly",
            z_score_col="energy_z",
            title="Total Distress Load",
            y_axis_title="Distress Load",
            line_color="orange"
        )

    # Q4 RCA
    with q4:
        display_rca_results(current_rca)

    #### SHapley Additive exPlanations (SHAP) ####
    
    st.divider()
    st.subheader("SHapley Additive exPlanations on example posts")

    # 1. Determine the default text (Highest distress score or empty)
    default_text = ""
    
    # If negative posts exist, find the one with minimum sentiment score (highest distress)
    if not range_df.empty:
        try:
            # Find the post with the least sentiment score
            top_idx = range_df["sentiment_score"].idxmin()
            top_row = range_df.loc[top_idx]
            default_text = top_row["clean_text"]
        except Exception:
            pass
    else:
        st.warning("No negative posts available in this date range to pre-fill.")

    # 2. Standard Text Area
    # Using 'value=default_text' pre-fills the box. 
    # The user can edit this, and 'text_to_analyze' will capture their changes.
    text_to_analyze = st.text_area(
        "", 
        value=default_text, 
        height=150
    )

    # 3. SHAP Button and Logic
    if st.button("Explain with SHAP", type="primary"):
        
        # Check if the text area is empty
        if not text_to_analyze or not text_to_analyze.strip():
            st.warning("The text box is empty. Please enter text to analyze.")
        else:
            try:
                with st.spinner("Creating SHAP explanation"):
                    # Run the SHAP function on whatever is currently in the text box
                    html_code = fn.run_shap_explanation(text_to_analyze, shap_tokenizer, shap_model, shap_class_names)
                    
                    # Render the SHAP plot
                    components.html(html_code, height=600, scrolling=True)
                    
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")


#############    PAGE-2 (LIVE REDDIT ANALYSIS)    #############


def page_2(embedder, llm_pipeline, shap_tokenizer, shap_model, shap_class_names):
    # Single sidebar logger for status messages
    sidebar_status = st.sidebar.empty()

    ## I am dividing the sidebar into 2 columns to have a text input and a button next to it
    col1, col2= st.sidebar.columns([4,1], vertical_alignment="bottom")
    with col1:
        subreddit_input = st.text_input(
            "Enter a subreddit",
            key="page2_subreddit_input_key",
            value=st.session_state.page2_current_input
        ).strip()

    ## This is the unicode for check mark
    with col2:
        fetch_button = st.button("\u2714", type="primary")
    # Text input to get the subreddit name
    

    # keep in session state
    st.session_state.page2_current_input = subreddit_input

    ## Sidebar controls
    # These will be shown after data is loaded.

    # We will load the controls no matter what. So, we need to use placeholder dataset before the data gets successfully loaded.
    # default placeholders in case df_page2 is None — we'll replace them when data exists
    df_for_controls = st.session_state.df_page2 if st.session_state.df_page2 is not None else pd.DataFrame({
        "created_utc": pd.to_datetime([pd.Timestamp.now()]),
        "prediction_label": ["Anxiety"],
        "sentiment_score": [0.0],
        "clean_text": [""]
    })

    # Ensure safe datetime conversion for the control defaults
    df_for_controls["created_utc"] = pd.to_datetime(df_for_controls["created_utc"], errors="coerce")
    # Dropping null values in created_utc
    df_for_controls = df_for_controls.dropna(subset=["created_utc"])

    # If the dataset is empty after cleaning, both sates will be shown to todays date
    if df_for_controls.empty:
        # fallback to today if nothing valid
        start_default = pd.Timestamp.now().date()
        end_default = pd.Timestamp.now().date()
    # else, min and max dates will be extracted from the data
    else:
        start_default = df_for_controls["created_utc"].min().date()
        end_default = df_for_controls["created_utc"].max().date()


    # If the data is empty, we set the available labels to anxiety only to avoid errors
    available_labels = sorted(df_for_controls["prediction_label"].unique().tolist())
    if not available_labels:
        available_labels = ["Anxiety"]
    selected_label = st.sidebar.selectbox(
        "Category",
        options=available_labels,
        index=available_labels.index("Anxiety") if "Anxiety" in available_labels else 0
    )

    # Anomaly slider
    anomaly_threshold = st.sidebar.slider(
        "Anomaly Sensitivity",
        1.0, 3.0, 1.5, 0.1
    )

    # Range date widget
    selected_dates = st.sidebar.date_input(
        "Select date range",
        value=(start_default, end_default)
    )

    # Creating RCA button in the sidebar
    _,col2,_ = st.sidebar.columns([1,6,1])
    with col2:
        run_rca_button = st.button("Find Root Cause", type="primary")

    ## Fetch & process data when button clicked
    if fetch_button:
        # If subreddit name is not entered, we will show an error
        if not subreddit_input:
            sidebar_status.error("Please enter a subreddit name.")
        else:
            # Else, we show info message that we are fetching posts
            sidebar_status.info(f"Fetching posts from r/{subreddit_input} ...")

            ## Get the env variables for Reddit API from .env file and pass it to the function
            df_raw, err = fn.fetch_reddit_posts_live(
                subreddit_input,
                os.getenv("CLIENT_ID"),
                os.getenv("CLIENT_SECRET"),
                os.getenv("USER_AGENT")
            )

            # The above function returns a dataframe and an error message (if any)
            # If there is an error message, we show that in the sidebar log and reset the session state varibles dataset and rca results to None
            if err:
                sidebar_status.error(err)
                st.session_state.df_page2 = None
                st.session_state.page2_rca = None
            else:
                # If there is no error, we will start loading the models and classifying posts
                sidebar_status.info("Cleaning / labeling / scoring posts...")

                # load page-2 models. This will return a classifier and tokenizer
                clf, _,= fn.load_page2_models()
                # classify posts. This will return a dataframe.
                df = fn.classify_posts_page2(df_raw, clf)

                # safe datetime
                df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

                # drop rows missing created_utc
                df = df.dropna(subset=["created_utc"])

                # If the dataframe is empty, we will show an error in the sidebar log and reset the session state varibles dataset and rca results to None
                if df.empty:
                    sidebar_status.error("No valid posts with timestamps were returned.")
                    st.session_state.df_page2 = None
                    st.session_state.page2_rca = None
                # else, we will save the dataframe to session state variable df_page2, set the previous subreddit to current subreddit input,
                # and reset the rca results to None (since new data is loaded)
                else:
                    st.session_state.df_page2 = df
                    st.session_state.page2_prev_subreddit = subreddit_input
                    st.session_state.page2_rca = None  # reset RCA cache for new data
                    sidebar_status.success("Preprocessed the Data")

  
    # If no data loaded yet, a info message will be shown in the center of the screen.
    if st.session_state.df_page2 is None:
        st.info("Enter a subreddit and click '\u2714' to begin analysis.")
        return

    df = st.session_state.df_page2.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
    df = df.dropna(subset=["created_utc"])
    if df.empty:
        st.error("Loaded data has no valid timestamps.")
        return

# The initial set of controls are when data was not loaded.
# We need to select the dates again based on the loaded data.
    # Recompute control defaults based on fetched df
    start_default = df["created_utc"].min().date()
    end_default = df["created_utc"].max().date()

    # If the user changed the category/date controls earlier, re-use those values where appropriate.

    # Checking if selected_dates is a tuple/list of length 2
    if not (isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2):
        selected_dates = (start_default, end_default)

    # Unpacking the selected dates if the user changed them
    start_date, end_date = selected_dates

    # Filter posts by date and label
    mask = (
        (df["created_utc"].dt.date >= start_date) &
        (df["created_utc"].dt.date <= end_date)
    )
    range_df = df[mask]
    range_df = range_df[range_df["prediction_label"] == selected_label]
    posts_to_analyze = range_df["clean_text"].tolist()

    #posts_to_analyze = range_df[range_df["sentiment_score"] < 0]["clean_text"].tolist()


    # If the rca_results variable (initially set to None when the website loads) or prev_subreddit not equal to selected_subreddit, we will run the function automatically
    selected_subreddit = st.session_state.page2_prev_subreddit
    if (st.session_state.page2_rca is None) or (st.session_state.page2_prev_subreddit != selected_subreddit):
        if posts_to_analyze:
            sidebar_status.info("Running Topic Modeling")
            # Storing results to a session variable so that even if we switch pages, the results persist.
            st.session_state.page2_rca = run_llm_rca(posts_to_analyze, [selected_label])
            sidebar_status.success("Done")
        else:
            st.session_state.page2_rca = []
        st.session_state.page2_prev_subreddit = selected_subreddit

    # We can also run RCA when the user clicks the button
    if run_rca_button:
        if posts_to_analyze:
            sidebar_status.info("Running Topic Modeling")
            st.session_state.page2_rca = run_llm_rca(posts_to_analyze, [selected_label])
            sidebar_status.success("Done")
        else:
            sidebar_status.warning("No posts found")

    # We are storing the current RCA results to a variable. This will be used when rendering the results on the screen
    current_rca = st.session_state.page2_rca

    ###### TIME SERIES ANALYSIS ######

    ## We are running a function to filter the data according to the classification selected, find distress score and other scores like rolling mean, rolling std, z-score etc.
    df_daily = fn.analyze_sentiment_over_time(
        df[df["prediction_label"] == selected_label],
        [selected_label]
    )

    ## Marking anomalies based on the selected threshold
    df_daily["distress_is_anomaly"] = (df_daily["distress_z"] > anomaly_threshold)
    df_daily["volume_is_anomaly"] = (df_daily["volume_z"] > anomaly_threshold)
    df_daily["energy_is_anomaly"] = (df_daily["energy_z"] > anomaly_threshold)

    ## Calculating upper and lower bands for visualization
    ## These bands are (anomaly_threshold * standard deviations) away from the rolling mean. Usually, the outliers fall outside these bands. This is just for the visualization purpose
    df_daily["distress_upper_band"] = df_daily["distress_mean"] + anomaly_threshold * df_daily["distress_std"]
    df_daily["distress_lower_band"] = df_daily["distress_mean"] - anomaly_threshold * df_daily["distress_std"]

    df_daily["volume_upper_band"] = df_daily["volume_mean"] + anomaly_threshold * df_daily["volume_std"]
    df_daily["volume_lower_band"] = df_daily["volume_mean"] - anomaly_threshold * df_daily["volume_std"]

    df_daily["energy_upper_band"] = df_daily["energy_mean"] + anomaly_threshold * df_daily["energy_std"]
    df_daily["energy_lower_band"] = df_daily["energy_mean"] - anomaly_threshold * df_daily["energy_std"]

    # We are dividing the page into 2 columns.
    q1, q2 = st.columns(2)
    q3, q4 = st.columns(2)

    # Q1 Time Series on Distress
    with q1:
        plot_anomaly_chart(
            data=df_daily,
            metric_col="distress_score",
            lower_band_col="distress_lower_band",
            upper_band_col="distress_upper_band",
            anomaly_col="distress_is_anomaly",
            z_score_col="distress_z",
            title="Distress Analysis",
            y_axis_title="Distress Score",
            line_color="blue"
        )

    with q2:
        plot_anomaly_chart(
            data=df_daily,
            metric_col="daily_volume",
            lower_band_col="volume_lower_band",
            upper_band_col="volume_upper_band",
            anomaly_col="volume_is_anomaly",
            z_score_col="volume_z",
            title="Volume Analysis",
            y_axis_title="Number of Posts",
            line_color="green"
        )

    with q3:
        plot_anomaly_chart(
            data=df_daily,
            metric_col="distress_energy",
            lower_band_col="energy_lower_band",
            upper_band_col="energy_upper_band",
            anomaly_col="energy_is_anomaly",
            z_score_col="energy_z",
            title="Total Distress Load",
            y_axis_title="Distress Load",
            line_color="orange"
        )

    # Q4 RCA
    with q4:
        display_rca_results(current_rca)

    #### SHapley Additive exPlanations (SHAP) ####
    
    st.divider()
    st.subheader("SHapley Additive exPlanations on example posts")

    # 1. Determine the default text (Highest distress score or empty)
    default_text = ""
    

    
    if not range_df.empty:
        try:
            # Find the post with the highest distress score
            top_idx = range_df["sentiment_score"].idxmax()
            top_row = range_df.loc[top_idx]
            default_text = top_row["clean_text"]
        except Exception:
            pass
    else:
        st.warning("No negative posts available in this date range to pre-fill.")

    # 2. Standard Text Area
    # Using 'value=default_text' pre-fills the box. 
    # The user can edit this, and 'text_to_analyze' will capture their changes.
    text_to_analyze = st.text_area(
        "", 
        value=default_text, 
        height=150
    )

    # 3. SHAP Button and Logic
    if st.button("Explain with SHAP"):
        
        # Check if the text area is empty
        if not text_to_analyze or not text_to_analyze.strip():
            st.warning("The text box is empty. Please enter text to analyze.")
        else:
            try:
                with st.spinner("Calculating SHAP values..."):
                    # Run the SHAP function on whatever is currently in the text box
                    html_code = fn.run_shap_explanation(text_to_analyze, shap_tokenizer, shap_model, shap_class_names)
                    
                    # Render the SHAP plot
                    components.html(html_code, height=600, scrolling=True)
                    
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

#### Our Router Logic to Switch between Pages ####


if st.session_state.page == "dashboard":

    # First, we load the data from google sheet for the selected subreddit. We are using the cached function defined earlier for performance
    df = load_data_cached(sheet_name, st.session_state.dashboard_subreddit)


    # Load the dashboard page function with data, subreddit and a sidebar logger placeholder. This sidebar logger is used in the entire code to show messages
    dashboard_page(
        df,
        st.session_state.dashboard_subreddit,
        sidebar_status=st.sidebar.empty()
    )
# If the state page is not "dashboard", we load the page-2 function. The models are passed as parameters to avoid re-loading them again
else:
    page_2(embedder, llm_pipeline, shap_tokenizer, shap_model, shap_class_names)