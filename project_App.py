import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    sw = stopwords.words('english')
    lm = WordNetLemmatizer()
    processed_text = text.lower()
    processed_text = word_tokenize(processed_text)
    processed_text = [i for i in processed_text if i not in sw]
    processed_text = [lm.lemmatize(i) for i in processed_text]
    processed_text = " ".join(processed_text)
    return processed_text

# Load the dataset
df = pd.read_csv(r"dataset.csv")
df.dropna(axis=0, inplace=True)

# Preprocess the text data
df['processed_text'] = df['text'].apply(preprocess_text)

# Create the feature matrix
cv = CountVectorizer(max_features=2000)
x = cv.fit_transform(df['processed_text']).toarray()
y = df['is_rumor']

# Train the Naive Bayes classifier
m1 = MultinomialNB()
m1.fit(x, y)

# Create Streamlit app
st.title("Rumor Detection App :zipper_mouth_face:")
# Input text box for user input
user_input = st.text_input("Enter text to classify:",max_chars=500)
# Classification button
button=st.button("check")
if not user_input and button:
    st.write("### Please provide text!")
elif button:
    #Visualization of data
    words=user_input.split()
    word_freq=Counter(words)
    word_freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
    plt.figure(figsize=(10, 6))
    st.write('#### Top 10 Most Frequent Words in the given text')
    st.bar_chart(data=word_freq_df.head(10),x='Frequency', y='Word')
    # Preprocess the user input
    processed_input = preprocess_text(user_input)

    # Transform the input text into a feature vector
    input_vector = cv.transform([processed_input]).toarray()

    # Make prediction using the trained classifier
    prediction = m1.predict(input_vector)
    a=str(user_input)
    print(a)
    # Display prediction result
    if prediction[0] == 1:
        st.write("## The text is classified as a rumor,it's fake!!! :x:")
        st.write("Total characters given in the input:",len(a))
        print("rumor")
        st.success("Done Successfully")
    else:
        st.write("## The text is not classified as a rumor :heavy_check_mark: ")
        st.write("Total characters given in the input:",len(a))
        print("Not a rumor")
        st.success("Done Successfully")