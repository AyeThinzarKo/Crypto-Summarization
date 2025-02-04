import streamlit as st
# from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.models.bart import BartForConditionalGeneration, BartTokenizer

import torch

# Loading the trained model and tokenizer (Fine-Tuned Model)
# model = BartForConditionalGeneration.from_pretrained('AyeThinzarKo/CryptoSummarization')
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

#----------Home------------------


#st.title("BBC News Summarization")
st.markdown("<h1 style='text-align: center; font-size:50px;font-weight: bold;'>BBC News Summarization</h1>", unsafe_allow_html=True)
st.write(" ")
st.write(" ")
# st.markdown("***")
# st.markdown("<p style='font-size:18px;'>This app summarizes BBC news articles into concise summaries.</p>", unsafe_allow_html=True)
#st.write("**This app summarizes BBC news articles into concise summaries.**") 

# Ensure session state is initialized for the text area
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# Defining the form
with st.form("summarization_form"):
    user_input = st.text_area("Enter a news article", st.session_state["user_input"])
    
    col1, col2 = st.columns([1, 5])
    with col1:
        summarize_button = st.form_submit_button('Summarize')
    with col2:
        reset_button = st.form_submit_button('Reset')

def word_count(text):
    text=text.split()
    return len(text)

# Handling the form submission
if summarize_button:
    if user_input:
        # Store the input in session state
        st.session_state["user_input"] = user_input

        # Calculating the word count
        word_count_input = word_count(user_input)

        st.info("Input Text:")
        st.markdown(
            f"<p style='text-align: justify;'>{user_input}</p>", unsafe_allow_html=True)

        st.markdown(f"***Word count of input article: {word_count_input}***")

        if word_count_input < 31:
            st.error("The word counts for your input text must be greater than 30 words.")
        else:

            # Tokenizing the input text
            input_tokens = tokenizer.batch_encode_plus([user_input], return_tensors='pt', max_length=1024, truncation=True)['input_ids']

            # Generating the summary
            encoded_ids = model.generate(input_tokens,
                                        num_beams=4,
                                        length_penalty=2.0,
                                        max_length=180,
                                        min_length=120,
                                        early_stopping=True,
                                        no_repeat_ngram_size=3)


            # The generated summary IDs
            generated_summary_ids = encoded_ids

            # Decoding the generated summary
            summary_by_FineTuned_Model = tokenizer.decode(encoded_ids.squeeze(), skip_special_tokens=True)   

            # Calculating word count of the summary
            word_count_summary = word_count(summary_by_FineTuned_Model)         

            # Displaying the summary
            st.success("Summary:")
            # st.write(summary)
            st.markdown(
            f"<p style='text-align: justify;'>{summary_by_FineTuned_Model}</p>", unsafe_allow_html=True)

            # Displaying the word count of the summary     
            st.markdown(f"***Word count of input article: {word_count_summary}***")

    else:
        st.warning("Please enter a news article.")

if reset_button:
    st.session_state["user_input"] = "  "
  
