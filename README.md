# **Disaster Management and NLP**

This repository showcases a robust workflow for multi-label text classification, combining effective preprocessing, balanced model training, and generative AI for label explanations. A user-friendly Streamlit app is included to demonstrate the model's capabilities.

---

## Workflow Overview

### **1. Preprocessing**
Preparing the text data for modeling with a series of cleaning and normalization steps:
- **HTML and Noise Removal:** 
  - Used regex patterns to remove HTML tags, mentions (`@username`), hashtags (`#hashtag`), URLs, numbers, and extra spaces.
- **Stop Word Elimination:** 
  - Leveraged the NLTK library to filter out common stop words.
- **Text Normalization:** 
  - Lowercased all text for uniformity.
- **Lemmatization:** 
  - Reduced words to their base form using lemmatization for better semantic understanding.

---

### **2. Addressing Label Skewness**
Ensuring a balanced dataset for improved model performance:
- **Reduction of Skewed Labels:** 
  - Limited the occurrence of over-represented labels to prevent overfitting.
- **Normalization of Label Distribution:** 
  - Adjusted label frequencies to enhance model generalization.

---

### **3. Model Training**
Building a state-of-the-art classification model:
- **Framework:** 
  - Utilized `BertForSequenceClassification` from Hugging Face for multi-label classification tasks.
- **Fine-Tuning:** 
  - Customized the pre-trained BERT model for task-specific optimization.

---

### **4. Generative AI Integration**
Enhancing interpretability with natural language explanations:
- **Gemini API:** 
  - Integrated open-source generative AI to produce explanations for the classified labels.
  - The API was prompted to generate meaningful insights alongside the classified labels.
- **Streamlit App:** 
  - Built an interactive Streamlit application to showcase the model's classification results and explanations.

---

## Repository Contents
- **Preprocessing Scripts:** Scripts for text cleaning and normalization.
- **Model Training Code:** Jupyter notebooks and Python scripts for training `BertForSequenceClassification`.
- **Streamlit App:** Source code for the interactive web application.
- **Sample Data:** Example datasets for replication of results.

---

## Getting Started

### **Prerequisites**
Ensure the following libraries and tools are installed:
- Python 3.8+
- NLTK
- Hugging Face Transformers
- Streamlit
- Gemini API key and SDK (if applicable)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-label-text-classification.git
2. ```bash
   streamlit run streamlit_app.py

## **Streamlit Outcome**

![image](https://github.com/user-attachments/assets/da024334-2c9b-41a9-b222-b749ea01fe79)
