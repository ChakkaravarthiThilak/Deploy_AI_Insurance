import streamlit as st
import pickle
import os
import numpy as np
import plotly.express as px
import torch
from sklearn.preprocessing import StandardScaler
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import docx
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Set Page Config
st.set_page_config(page_title="AI Insurance System", layout="wide")

# Function to Load Pickle Models Safely
def load_pickle_model(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            if not hasattr(model, "predict"):
                st.error(f"Loaded model from {file_path} does not support prediction.")
                return None
            return model
        except ModuleNotFoundError as e:
            st.error(f"Missing module: {e}. Install the required package and retry.")
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    return None

# Load all models using a dictionary and loop
model_paths = {
    "risk_classification_model": "C:/Users/Administrator/Risk_Prediction_model.pkl",
    "fraud_detection_model": "C:/Users/Administrator/Fraudulent_Detection_Model.pkl"
}

models = {}
for model_name, path in model_paths.items():
    models[model_name] = load_pickle_model(path)

# Unpack models from dictionary
risk_classification_model = models.get("risk_classification_model")
tfidf_vectorizer = models.get("tfidf_vectorizer")
scaler = models.get("scaler")
fraud_detection_model = models.get("fraud_detection_model")

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize translation models dictionary
translation_models = {
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Chinese": "Helsinki-NLP/opus-mt-en-zh"
}

# Summarization model
summarization_model = "facebook/bart-large-cnn"

# Load model and tokenizer once
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_user_inputs" not in st.session_state:
    st.session_state.past_user_inputs = []
if "past_bot_responses" not in st.session_state:
    st.session_state.past_bot_responses = []


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Risk Assessment", "Fraud Detection", 
                                "Sentiment Analysis", "Customer Segmentation", 
                                "Multilingual Insurance Policies", "Chatbot"])
# Home Page
if page == "Home":
    st.title("AI-Powered Intelligent Insurance System Dashboard")
    st.image("https://source.unsplash.com/1000x400/?insurance,ai")
    st.markdown("### Features:")
    st.markdown("- Risk Classification & Claim Prediction")
    st.markdown("- Fraud Detection")
    st.markdown("- Sentiment Analysis of Customer Feedback")
    st.markdown("- Customer Segmentation")
    st.markdown("- Multilingual Policy Translation & Summarization")
    st.markdown("- Chatbot")
    
# Risk Assessment Page
elif page == "Risk Assessment":
    st.title("Risk Assessment & Claim Prediction")

    age = st.number_input("Customer Age", min_value=18, max_value=100, step=1)
    income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
    claim_history = st.number_input("Claim History", min_value=0, step=1)
    fraudulent_claim = st.selectbox("Fraudulent Claim", ["Yes", "No"])
    premium_amount = st.number_input("Premium Amount", min_value=0.0, step=50.0)
    claim_amount = st.number_input("Claim Amount", min_value=0.0, step=100.0)
    risk_score = st.number_input("Risk Score", min_value=0.0, step=0.1)
    anomaly_score = st.number_input("Anomaly Score", min_value=0.0, step=0.1)

    policy_type = st.selectbox("Policy Type", ["Health", "Life", "Property"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    policy_type_mapping = {"Health": [1, 0, 0], "Life": [0, 1, 0], "Property": [0, 0, 1]}
    gender_mapping = {"Male": [1, 0], "Female": [0, 0], "Other": [0, 1]}

    policy_type_encoded = policy_type_mapping[policy_type]
    gender_encoded = gender_mapping[gender]
    fraudulent_claim = 1 if fraudulent_claim == "Yes" else 0

    input_features_dict = {
        "Customer_Age": age, "Annual_Income": income, "Claim_History": claim_history, 
        "Fraudulent_Claim": fraudulent_claim, "Premium_Amount": premium_amount, 
        "Claim_Amount": claim_amount, "Risk_Score": risk_score, "Anomaly_Score": anomaly_score, 
        "Policy_Type_Health": policy_type_encoded[0], "Policy_Type_Life": policy_type_encoded[1], 
        "Policy_Type_Property": policy_type_encoded[2], "Gender_Male": gender_encoded[0], 
        "Gender_Other": gender_encoded[1]
    }

    if risk_classification_model is not None:
        expected_features = getattr(risk_classification_model, "feature_names_in_", None)
        if expected_features is not None:
            input_features = np.array([[input_features_dict.get(feature, 0) for feature in expected_features]])
        else:
            input_features = np.array([[v for v in input_features_dict.values()]])
    else:
        input_features = np.array([[v for v in input_features_dict.values()]])

    if scaler is not None:
        input_features = scaler.transform(input_features)

    risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

    if st.button("Predict Risk Level"):
        if risk_classification_model is not None:
            try:
                risk_prediction = risk_classification_model.predict(input_features)
                risk_label = risk_labels.get(int(risk_prediction[0]), "Unknown Risk Level")
                st.success(f"Predicted Risk Level: {risk_label}")
            except Exception as e:
                st.error(f"Error in prediction: {e}")

# Fraud Detection Page
elif page == "Fraud Detection":
    st.title("Insurance Fraud Detection")
    st.markdown("Enter claim details below to check for potential fraud.")

    claim_id = st.text_input("Claim ID", value="1001")
    customer_id = st.text_input("Customer ID", value="CUST001")
    claim_amount = st.number_input("Claim Amount", min_value=0.0, step=100.0)
    claim_type = st.selectbox("Claim Type", ["Auto", "Home", "Health", "Life", "Travel"])
    suspicious_flags = st.selectbox("Suspicious Flags", ["Yes", "No"])
    annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
    fraud_label = st.selectbox("Known Fraud Label", ["Yes", "No"])

    claim_type_mapping = {"Auto": 0, "Home": 1, "Health": 2, "Life": 3, "Travel": 4}
    claim_type_encoded = claim_type_mapping[claim_type]
    suspicious_encoded = 1 if suspicious_flags == "Yes" else 0
    fraud_encoded = 1 if fraud_label == "Yes" else 0

    if st.button("Detect Fraud"):
        if fraud_detection_model is not None:
            try:
                claim_id_val = int(''.join(filter(str.isdigit, claim_id))) or 0
                customer_id_val = int(''.join(filter(str.isdigit, customer_id))) or 0

                features = np.array([[claim_id_val, customer_id_val, claim_amount, claim_type_encoded, suspicious_encoded, annual_income]])
                expected_features = getattr(fraud_detection_model, "feature_names_in_", None)
                if expected_features is not None:
                    st.write("Model expects features in this order:", expected_features)
                    if features.shape[1] != len(expected_features):
                        st.error(f"Expected {len(expected_features)} features, but got {features.shape[1]}.")
                    else:
                        proba = fraud_detection_model.predict_proba(features)[0][1]
                        custom_threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.5)
                        predicted_label = 1 if proba >= custom_threshold else 0
                        label = "Fraudulent" if predicted_label == 1 else "Legitimate"
                        st.info(f"Fraud Probability: {proba:.2f}")
                        st.success(f"Prediction: {label} Claim")
                else:
                    prediction = fraud_detection_model.predict(features)
                    label = "Fraudulent" if prediction[0] == 1 else "Legitimate"
                    proba = fraud_detection_model.predict_proba(features)[0][1]
                    st.info(f"Fraud Probability: {proba:.2f}")
                    st.success(f"Prediction: {label} Claim")
            except Exception as e:
                st.error(f"Error in fraud prediction: {e}")
        else:
            st.warning("Fraud detection model not loaded.")

# Sentiment Analysis Page
elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis of Customer Feedback")
    feedback_text = st.text_area("Enter Customer Feedback:")
    if st.button("Analyze Sentiment"):
        if feedback_text:
            sentiment_score = sentiment_analyzer.polarity_scores(feedback_text)
            compound_score = sentiment_score['compound']
            sentiment_label = "Positive" if compound_score >= 0.05 else "Negative" if compound_score <= -0.05 else "Neutral"
            st.write(f"Sentiment Score: {compound_score}")
            st.success(f"Predicted Sentiment: {sentiment_label}")
        else:
            st.error("Please enter feedback text to analyze.")
            
#Customer Segmentation page
elif page == "Customer Segmentation":
    st.title("Customer Segmentation Analysis")
    st.markdown("Analyze customer segments based on demographic and behavioral features.")

    # Load segmentation models with error handling
    segmentation_models = {
        "kmeans": "kmeans.pkl",
        "pca": "pca.pkl",
        "scaler": "customer_scaler.pkl",
        "labels": "cluster_labels.pkl"
    }

    # Model loading function
    @st.cache_resource
    def load_segmentation_model(name):
        try:
            model_path = os.path.join("./models/", segmentation_models[name])
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading {name} model: {str(e)}")
            return None

    # Load models
    kmeans = load_segmentation_model("kmeans")
    pca = load_segmentation_model("pca")
    customer_scaler = load_segmentation_model("scaler")
    cluster_labels = load_segmentation_model("labels")

    # Input Fields
    st.subheader("Customer Attributes")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 100, 30)
    with col2:
        income = st.number_input("Annual Income ($)", 
                               min_value=0.0, 
                               max_value=500000.0,
                               value=50000.0,
                               step=1000.0)
    with col3:
        spending_score = st.slider("Spending Score", 1, 100, 50)

    if st.button("Analyze Customer Segment"):
        if None in (kmeans, pca, customer_scaler):
            st.error("System initialization failed. Contact support.")
        else:
            try:
                # Process input
                input_data = np.array([[age, income, spending_score]])
                scaled_data = customer_scaler.transform(input_data)
                reduced_data = pca.transform(scaled_data)
                cluster = kmeans.predict(reduced_data)[0]
                label = cluster_labels.get(cluster, f"Segment {cluster+1}")
                
                # Display results
                st.success(f"**Customer Segment:** {label}")
                
                # Segment description
                st.subheader("Segment Characteristics")
                segment_info = {
                    0: "**Value Seekers**: Moderate spending, price-sensitive",
                    1: "**Premium Clients**: High income, luxury purchases",
                    2: "**Young Adventurers**: High engagement, trend-focused",
                    3: "**Conservative Buyers**: Stable spending patterns"
                }
                st.markdown(segment_info.get(cluster, "**General Customer**: Mixed characteristics"))
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                
# Multilingual Insurance Policies Page
elif page == "Multilingual Insurance Policies":
    st.title("Multilingual Translation & Summarization")
    st.markdown("Translate and summarize insurance policy documents using state-of-the-art NLP models.")
    
    # Define translation models with proper language codes
    translation_models = {
        "French": ("es", "fr"),  # Spanish to French
        "Spanish": ("es", "es"),  # Identity for Spanish
        "German": ("es", "de"),   # Spanish to German
        "Chinese": ("es", "zh")   # Spanish to Chinese via English
    }
    
    @st.cache_resource
    def load_translation_model(source_lang, target_lang):
        try:
            if (source_lang, target_lang) == ("es", "zh"):
                # Two-step translation through English
                model_es_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
                tokenizer_es_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
                model_en_zh = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
                tokenizer_en_zh = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
                return [tokenizer_es_en, tokenizer_en_zh], [model_es_en, model_en_zh]
            else:
                model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                return [tokenizer], [model]
        except Exception as e:
            st.error(f"Error loading translation model: {e}")
            return None, None

    @st.cache_resource
    def load_summarization_components():
        try:
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
            return tokenizer, model
        except Exception as e:
            st.error(f"Error loading summarization model: {e}")
            return None, None

    def extract_text_from_file(uploaded_file):
        """Extract text from uploaded file"""
        try:
            if uploaded_file is None:
                return ""
                
            if uploaded_file.type == "text/plain":
                return uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                return "\n".join([page.extract_text() for page in pdf_reader.pages])
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(uploaded_file)
                return "\n".join([para.text for para in doc.paragraphs])
            return ""
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return ""

    def translate_text(text, target_language):
        """Improved translation handling for complex language pairs"""
        try:
            if not text.strip():
                return ""
            
            lang_codes = translation_models.get(target_language)
            if not lang_codes:
                st.error("Language not supported")
                return text
                
            source_lang, target_lang = lang_codes
            tokenizers, models = load_translation_model(source_lang, target_lang)
            
            if not tokenizers or not models:
                return text
                
            # Split text into manageable chunks
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            
            translated_chunks = []
            with st.spinner(f"Translating to {target_language}..."):
                for chunk in chunks:
                    if target_lang == "zh":
                        # Two-step translation: Spanish -> English -> Chinese
                        inputs_es_en = tokenizers[0](chunk, return_tensors="pt", padding=True)
                        outputs_es_en = models[0].generate(**inputs_es_en)
                        english_text = tokenizers[0].decode(outputs_es_en[0], skip_special_tokens=True)
                        
                        inputs_en_zh = tokenizers[1](english_text, return_tensors="pt", padding=True)
                        outputs_en_zh = models[1].generate(**inputs_en_zh)
                        translated = tokenizers[1].decode(outputs_en_zh[0], skip_special_tokens=True)
                    else:
                        # Direct translation
                        inputs = tokenizers[0](chunk, return_tensors="pt", padding=True)
                        outputs = models[0].generate(**inputs)
                        translated = tokenizers[0].decode(outputs[0], skip_special_tokens=True)
                    
                    translated_chunks.append(translated)
            
            return " ".join(translated_chunks)
        except Exception as e:
            st.error(f"Translation error: {e}")
            return text

    def summarize_text(text):
        """Summarize text using BART model"""
        try:
            if not text.strip():
                return ""
            
            tokenizer, model = load_summarization_components()
            if tokenizer is None or model is None:
                return text
                
            # Split text into chunks of 1024 tokens
            inputs = tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
                padding=True
            )
            
            with st.spinner("Summarizing..."):
                summary_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=150,
                    early_stopping=True
                )
                return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
        except Exception as e:
            st.error(f"Summarization error: {e}")
            return text

    # File upload and processing
    uploaded_file = st.file_uploader(
        "Upload Insurance Document", 
        type=["pdf", "txt", "docx"],
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    document_text = ""
    if uploaded_file is not None:
        document_text = extract_text_from_file(uploaded_file)
        if document_text:
            st.subheader("Document Content")
            st.text_area("Original Text", document_text, height=300, key="original_text")

    if document_text:
        target_language = st.selectbox(
            "Select Target Language",
            list(translation_models.keys()),
            index=0,
            help="Select the language to translate to"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Translate Document"):
                translated_text = translate_text(document_text, target_language)
                st.subheader(f"Translated Text ({target_language})")
                st.text_area("Translation Result", translated_text, height=300, key="translated_text")
        
        with col2:
            if st.button("Summarize Document"):
                summarized_text = summarize_text(document_text)
                st.subheader("Document Summary")
                st.text_area("Summary Result", summarized_text, height=300, key="summary_text")
        
        if st.button("Translate Summary"):
            summarized_text = summarize_text(document_text)
            if summarized_text:
                translated_summary = translate_text(summarized_text, target_language)
                st.subheader(f"Translated Summary ({target_language})")
                st.text_area("Translated Summary", translated_summary, height=300, key="translated_summary")

#Chatbot Page
elif page == "Chatbot":
    st.title("🤖 AI Insurance Chatbot")
    st.markdown("Ask me anything about your insurance policy, claims, or coverage.")

    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None
    if "past_user_inputs" not in st.session_state:
        st.session_state.past_user_inputs = []
    if "past_bot_responses" not in st.session_state:
        st.session_state.past_bot_responses = []

    user_input = st.chat_input("Ask something...")

    if user_input:
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = (
            torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
            if st.session_state.chat_history_ids is not None else new_input_ids
        )

        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        st.session_state.chat_history_ids = chat_history_ids
        st.session_state.past_user_inputs.append(user_input)
        st.session_state.past_bot_responses.append(bot_output)

    for user, bot in zip(st.session_state.past_user_inputs, st.session_state.past_bot_responses):
        with st.chat_message("user"):
            st.markdown(user)
        with st.chat_message("assistant"):
            st.markdown(bot)
