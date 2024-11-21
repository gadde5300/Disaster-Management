import os
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import google.generativeai as genai

# Set device for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load custom-trained BERT model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "saved_model_new"  # Replace with your saved model directory
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Load Gemini API
GOOGLE_API_KEY = 'Your API Here'  # Ensure that your API key is set in the environment variables
genai.configure(api_key=GOOGLE_API_KEY)

# Load the models
model, tokenizer = load_model_and_tokenizer()

# Streamlit UI
st.title("Multi-Label Classification with BERT")
st.write("This app uses a custom-trained multi-label BERT model to classify text, then generates an explanation using Gemini AI.")

# User input
input_text = st.text_area("Enter your text here", height=200)

# Define threshold for multi-label classification
threshold = 0.65

if st.button("Classify and Explain"):
    if input_text.strip():
        with st.spinner("Classifying the input text..."):
            # Tokenize the input text
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)
            
            # Get predictions from the BERT model
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Apply sigmoid to get probabilities and determine active labels
            probabilities = torch.sigmoid(logits)
            predicted_labels = (probabilities > threshold).int()

            # Get the label names from the model's configuration
            labels = model.config.id2label

            # Show the predicted labels
            predicted_label_names = [labels[i] for i, is_predicted in enumerate(predicted_labels[0]) if is_predicted.item() == 1]
            
            if predicted_label_names:
                st.write("### Predicted Labels:")
                st.write(", ".join(predicted_label_names))
            else:
                st.write("### No labels predicted")

            # Now generate an explanation using Gemini
            st.write("### Generating Explanation...")

            # Create a prompt for Gemini AI
            prompt = f"Based on the following predicted labels from a disaster response model, please identify and explain the immediate needs of the people in the affected area. This will help the response team prioritize their actions. Describe the specific requirements and how they relate to the given context: {', '.join(predicted_label_names)}. The input text is: {input_text}."


            try:
                # Start a chat with Gemini API
                model = genai.GenerativeModel('gemini-pro')
                chat = model.start_chat(history=[])
                
                # Send the prompt to the Gemini model and stream the response
                response = chat.send_message(prompt, stream=True)
                explanation = ""
                for chunk in response:
                    if chunk.text:
                        explanation += chunk.text

                # Show the explanation in Streamlit
                st.write("### Explanation from Gemini AI:")
                st.write(explanation)

            except Exception as e:
                st.error(f"Error generating explanation: {str(e)}")
    else:
        st.error("Please enter some text to classify.")
