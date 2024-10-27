import streamlit as st
import pandas as pd
import instaloader
import spacy
import time
from collections import Counter
from transformers import pipeline
import torch
import os

# Load the spaCy model
@st.cache_resource
def load_spacy_model():
    nlp = spacy.load('en_core_web_sm')
    return nlp

# Predefined themes for classification
THEME_KEYWORDS = {
    'Fashion': ['fashion', 'style', 'outfit', 'designer', 'beauty', 'runway', 'trend', 'streetwear', 'accessories', 'pants', 'wardrobe'],
    'Tech': ['technology', 'tech', 'gadgets', 'innovation', 'software', 'AI', 'code', 'startup', 'electronics'],
    'Health': ['health', 'fitness', 'wellness', 'nutrition', 'workout', 'exercise', 'mentalhealth', 'yoga', 'healthyhabits'],
    'Food': ['food', 'cooking', 'recipes', 'foodie', 'nutrition', 'dining', 'chef', 'baking', 'foodphotography'],
    'Travel': ['travel', 'adventure', 'vacation', 'explore', 'wanderlust', 'destinations', 'travelgram', 'backpacking', 'tourism'],
    'Photography': ['photography', 'photo', 'camera', 'photographer', 'portrait', 'landscape', 'dslr', 'editing', 'instaphoto'],
    'Art': ['art', 'artist', 'painting', 'drawing', 'illustration', 'creative', 'sketch', 'gallery', 'artwork'],
    'Music': ['music', 'song', 'artist', 'concert', 'instrument', 'band', 'playlist', 'liveperformance', 'musician', 'hiphop', 'underground'],
    'Sports': ['sports', 'fitness', 'exercise', 'athlete', 'training', 'game', 'competition', 'team', 'workoutroutine'],
    'Education': ['education', 'learning', 'study', 'school', 'teacher', 'student', 'onlinelearning', 'tutorial', 'knowledge'],
    'Lifestyle': ['lifestyle', 'daily', 'life', 'happiness', 'motivation', 'inspiration', 'mindfulness', 'selfcare', 'routine'],
    'Business': ['business', 'entrepreneur', 'startup', 'marketing', 'finance', 'success', 'leadership', 'strategy', 'branding'],
    'Home': ['home', 'interior', 'decor', 'design', 'homedecor', 'DIY', 'organization', 'architecture', 'livingroom'],
    'Entertainment': ['entertainment', 'movies', 'tv', 'celebrity', 'series', 'show', 'hollywood', 'streaming', 'bingewatch'],
    'Gaming': ['gaming', 'games', 'gamer', 'videogames', 'streaming', 'esports', 'gamingcommunity', 'gameplay', 'console'],
    'Nature': ['nature', 'outdoors', 'wildlife', 'environment', 'sustainability', 'ecology', 'landscape', 'hiking', 'flora'],
    'Beauty': ['beauty', 'makeup', 'skincare', 'cosmetics', 'beautyblogger', 'hair', 'nails', 'glam', 'beautytips'],
    'Finance': ['finance', 'money', 'investing', 'stocks', 'wealth', 'budgeting', 'saving', 'financialplanning', 'cryptocurrency'],
    'Automotive': ['automotive', 'cars', 'motorcycle', 'vehicles', 'auto', 'driving', 'carsofinstagram', 'carphotography', 'motorcars'],
    'Parenting': ['parenting', 'family', 'children', 'mom', 'dad', 'parent', 'parenthood', 'kids', 'familytime'],
    'Pets': ['pets', 'animals', 'dog', 'cat', 'petsofinstagram', 'cuteanimals', 'petcare', 'petlovers', 'animalphotography'],
    'Fitness': ['fitness', 'gym', 'workout', 'training', 'fitlife', 'bodybuilding', 'cardio', 'strengthtraining', 'fitnessmotivation'],
    'Quotes': ['quotes', 'inspiration', 'motivation', 'quotesoftheday', 'quote', 'wordsofwisdom', 'lifequotes', 'motivationalquotes', 'quoteoftheday'],
    'DIY': ['DIY', 'crafts', 'handmade', 'diyprojects', 'homemade', 'crafting', 'diyideas', 'upcycling', 'creativeprojects'],
    'Books': ['books', 'reading', 'bookstagram', 'literature', 'booklover', 'bookworm', 'novel', 'bibliophile', 'bookreview'],
    'Mental Health': ['mentalhealth', 'selfcare', 'mindfulness', 'wellbeing', 'therapy', 'stressrelief', 'mentalwellness', 'selflove', 'mentalhealthawareness'],
    'Sustainability': ['sustainability', 'eco', 'green', 'environment', 'recycle', 'zerowaste', 'sustainableliving', 'ecofriendly', 'conservation'],
    'Fashion Accessories': ['accessories', 'jewelry', 'bags', 'shoes', 'watches', 'belts', 'scarves', 'sunglasses', 'handbags'],
    'Seasonal': ['summer', 'winter', 'spring', 'autumn', 'holiday', 'festive', 'seasonal', 'season', 'weather'],
    'Events': ['events', 'concert', 'festival', 'party', 'celebration', 'wedding', 'eventplanning', 'event', 'celebrate'],
    # Add more themes and keywords as needed
}

def classify_content_theme_combined(text, use_bert=False, model_name=None):
    nlp = load_spacy_model()
    doc = nlp(text)
    tokenized_text = [
        token.text for token in doc if not token.is_stop and token.is_alpha]

    theme_counter = Counter()
    for theme, keywords in THEME_KEYWORDS.items():
        for keyword in keywords:
            theme_counter[theme] += text.lower().count(keyword.lower())
    most_common_theme = theme_counter.most_common(1)

    if use_bert and model_name is not None:
        classifier = pipeline('zero-shot-classification', model=model_name)
        labels = list(THEME_KEYWORDS.keys())
        result = classifier(text, candidate_labels=labels, multi_label=False)
        # Get top two themes
        most_common_theme = list(zip(result['labels'], result['scores']))[:2]

    if most_common_theme:
        if not use_bert:
            return most_common_theme[0][0]
        else:
            return ', '.join([t[0] for t in most_common_theme])
    else:
        return "Unknown"

def get_influencer_content_theme(username, L, use_bert=False, model_name=None):
    # Load Instagram profile
    profile = instaloader.Profile.from_username(L.context, username)

    # Get bio (biography) from the profile
    bio_text = profile.biography

    # Get username and full name
    username_text = profile.username
    full_name_text = profile.full_name if profile.full_name else ""

    # Get external URL (if available)
    external_url = profile.external_url if profile.external_url else ""

    # Get recent 6 posts
    posts = profile.get_posts()
    post_captions = []
    for i, post in enumerate(posts):
        if i >= 6:  # Limit to 6 posts
            break
        caption = post.caption if post.caption else ""
        post_captions.append(caption)

    # Combine all available text sources
    combined_text = f"{bio_text} {username_text} {full_name_text} {external_url} " + \
        " ".join(post_captions)

    # Classify and return content theme using the combined data
    return classify_content_theme_combined(combined_text, use_bert, model_name)

def calculate_engagement_rate(profile, L):
    time.sleep(10)  # To avoid Instagram's rate limiting

    # Simulate calculating engagement rate (mockup)
    posts = profile.get_posts()
    engagement_sum = 0
    post_count = 0
    for post in posts:
        if post_count >= 5:
            break
        engagement_sum += post.likes
        post_count += 1

    # Engagement calculation
    if post_count > 0 and profile.followers > 0:
        engagement_rate = (engagement_sum / post_count) / \
            profile.followers * 100
    else:
        engagement_rate = 0

    return engagement_rate

def login_instagram(L):
    session_file = "instaloader_session"

    # Prompt for username
    username = st.text_input(
        "Instagram Username for Login", key='login_username')

    # Check if session file exists and load it
    if os.path.exists(session_file):
        try:
            L.load_session_from_file(username, session_file)
            st.success("Logged in using the saved session!")
            st.session_state['logged_in'] = True
        except Exception as e:
            st.error(
                f"Session file exists but could not be loaded. Error: {str(e)}")
            st.session_state['logged_in'] = False
    else:
        # Login to Instagram and save session
        password = st.text_input("Instagram Password", type="password")

        if st.button("Login"):
            try:
                L.login(username, password)
                L.save_session_to_file(session_file)  # Save session to file
                st.success("Successfully logged in and session saved!")
                st.session_state['logged_in'] = True
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
                st.session_state['logged_in'] = False

def main():
    st.title('Instagram Metrics and Content Theme Analyzer')

    L = instaloader.Instaloader()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Login or load session
    login_instagram(L)

    if st.session_state['logged_in']:
        st.header("Enter the Instagram Username to Analyze")
        target_username = st.text_input("Instagram Username to Analyze")

        if st.button("Analyze"):
            try:
                target_username = target_username.strip()
                profile = instaloader.Profile.from_username(
                    L.context, target_username)

                # Get metrics
                engagement_rate = calculate_engagement_rate(profile, L)
                follower_count = profile.followers
                following_count = profile.followees
                if following_count == 0:
                    following_count = 1  # Avoid division by zero
                if engagement_rate > 0.2 or follower_count / following_count >= 0.5:
                    real_or_fake = "Real"
                else:
                    real_or_fake = "Fake"

                # Get content themes
                content_theme_method1 = get_influencer_content_theme(
                    target_username, L, use_bert=False)
                content_theme_method2 = get_influencer_content_theme(
                    target_username, L, use_bert=True, model_name="joeddav/distilbert-base-uncased-mnli")  # Using lightweight model
                content_theme_method2 = get_influencer_content_theme(
                    target_username, L, use_bert=True, model_name="typeform/mobilebert-uncased-mnli")

                # Display the results
                st.write(f"**Username:** {target_username}")
                st.write(f"**Followers:** {follower_count}")
                st.write(f"**Following:** {following_count}")
                st.write(f"**Posts:** {profile.mediacount}")
                st.write(f"**Engagement Rate:** {engagement_rate:.2f}%")
                st.write(f"**Account Type:** {real_or_fake}")

                st.write("### Content Themes:")
                st.write(
                    f"**Method 1 (Keyword Matching):** {content_theme_method1}")
                st.write(
                    f"**Method 2 (Zero-shot with DistilBERT):** {content_theme_method2}")

            except Exception as e:
                st.error(f"Failed to process {target_username}: {str(e)}")

if __name__ == "__main__":
    main()
