import pickle
import string
from nltk.corpus import stopwords
import nltk
import streamlit as st

nltk.download('stopwords')

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [
        word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')
    ]
    return ' '.join(Test_punc_removed_join_clean)


try:
    with open('tweets_countvectorizer1.pkl', 'rb') as vectorizer_file:
        tweets_countvectorizer1 = pickle.load(vectorizer_file)
    with open('NB_classifier1.pkl', 'rb') as classifier_file:
        NB_classifier1 = pickle.load(classifier_file)
except FileNotFoundError:
    st.error("Required model files not found. Please ensure 'tweets_countvectorizer1.pkl' and 'NB_classifier1.pkl' are in the directory.")
    st.stop()

sentiment_mapping = {1: 'Negative', 0: 'Positive'}

def main():
    st.title("Twitter Sentiment Classifier")
    
    st.image("twitter.png", caption="Twitter Sentiment Classifier", width=250)
    
    st.write("Enter a tweet below:")
    
    input_text = st.text_area("Input Text:", "")
    
    if st.button("Predict"):
        if input_text.strip():
            try:
                cleaned_text = message_cleaning(input_text)
                vectorized_text = tweets_countvectorizer1.transform([cleaned_text])
           
                sentiment_prediction = NB_classifier1.predict(vectorized_text)[0]
                predicted_sentiment = sentiment_mapping.get(sentiment_prediction, 'Unknown Sentiment')
                

                st.write(f"Predicted Sentiment: **{predicted_sentiment}**")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
