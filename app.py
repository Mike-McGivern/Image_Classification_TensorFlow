import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import re
from typing import List, Tuple, Optional
#import google.generativeai as genai

# Fill this with your Gemini / Generative API key (you can paste your key here).
# WARNING: do NOT commit your real API key to source control. This placeholder lets
# you drop the key into the app before running locally.
GEMINI_API_KEY: str = "AIzaSyBpr10elpzjTeCC4JxJROTTQUD-HAZGhC0"  # <-- paste your GEMINI API key here when ready

# Load the saved model
@st.cache_resource
def load_model():
    model_path = "butterfly_model.keras"
    return tf.keras.models.load_model(model_path)

# Initialize label encoder
@st.cache_resource
def get_label_encoder():
    species = [
        'Danaus_plexippus', 'Heliconius_charitonius', 'Heliconius_erato',
        'Junonia_coenia', 'Lycaena_phlaeas', 'Nymphalis_antiopa',
        'Papilio_cresphontes', 'Pieris_rapae', 'Vanessa_atalanta',
        'Vanessa_cardui'
    ]
    le = LabelEncoder()
    le.fit(species)
    return le

def main():
    st.title("Butterfly Species Classifier")
    st.write("Upload an image of a butterfly to classify its species")
    
    # Load model and label encoder
    try:
        model = load_model()
        label_encoder = get_label_encoder()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    


    # Gemini Generative API controls (now the only comparison)
    st.sidebar.header("Gemini Generative AI")
    use_generative = st.sidebar.checkbox("Compare with Gemini Generative AI", value=True)
    if use_generative:
        st.sidebar.write("Gemini will produce a guessed label, confidence, and a short explanation using your API key.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Convert uploaded file to image
            image_bytes = uploaded_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            image = np.array(pil_image)
            
            # Display uploaded image
            st.image(image, caption='Uploaded Image')
            
            if st.button('Classify'):
                # Preprocess image
                img = cv2.resize(image, (224, 224))
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Make prediction
                with st.spinner('Classifying...'):
                    prediction = model.predict(img)
                    predicted_class = np.argmax(prediction)
                    predicted_species = label_encoder.inverse_transform([predicted_class])[0]
                    confidence = prediction[0][predicted_class]

                    # Show results in two columns: Neural Network and Gemini Generative
                    if use_generative:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Neural Network")
                            st.success("Classification complete!")
                            st.write(f"**Predicted Species:** {predicted_species}")
                            st.write(f"**Confidence:** {confidence:.2%}")
                        with col2:
                            st.subheader("Gemini Generative AI")
                            with st.spinner("Contacting Gemini..."):
                                try:
                                    gen_label, gen_conf, gen_expl = classify_with_generative(image_bytes, predicted_species, label_encoder)
                                    st.write(f"**Gemini guess:** {gen_label} — {gen_conf:.2%}")
                                    st.write("**Explanation:**")
                                    st.write(gen_expl)
                                except Exception as e:
                                    st.error(f"Gemini generative error: {e}")
                    else:
                        st.success("Classification complete!")
                        st.write(f"**Predicted Species:** {predicted_species}")
                        st.write(f"**Confidence:** {confidence:.2%}")
    
    # Note: Only Gemini Generative AI integration is active below.
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")





def _normalize_species_name(name: str) -> str:
    """Small helper to normalize species-like strings for comparison/matching."""
    name = name.replace('_', ' ')
    name = re.sub(r'[^a-zA-Z0-9 ]+', '', name)
    return name.strip().lower()


def classify_with_generative(image_bytes: bytes, nn_species: str, label_encoder: LabelEncoder) -> Tuple[str, float, str]:
    """Call Gemini Pro Vision using the REST API and return (label, confidence, explanation), with robust error handling."""
    import requests
    import base64
    
    label = "Unknown"
    confidence = 0.0
    explanation = "Gemini generative AI could not be reached."

    if not GEMINI_API_KEY:
        explanation = "GEMINI_API_KEY is empty. Please set your key in app.py."
        return label, confidence, explanation

    # Convert image to base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = (
        "You are an expert lepidopterist. Given a photograph of a butterfly, provide a single-line species guess, a confidence score between 0 and 1, "
        "and a short (1-2 sentence) explanation for the guess. Format the response as:\nlabel: <species name>\nconfidence: <0-1>\nexplanation: <text>\n"
    )
    prompt += f"\nContext: a local neural-network predicted: {nn_species}. Provide a corrected or confirmed guess and explanation."

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64
                    }
                }
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 256,
            "temperature": 0.2
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        # Extract the text from the response
        text = ""
        if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
            c0 = j["candidates"][0]
            parts = c0.get("content", {}).get("parts", [])
            if parts and isinstance(parts, list):
                text = parts[0].get("text", "")
            else:
                text = c0.get("content", "")
        else:
            text = str(j)

        explanation = text or ""
        m_label = re.search(r'label:\s*(.+)', explanation, re.IGNORECASE)
        m_conf = re.search(r'confidence:\s*([0-9]*\.?[0-9]+)', explanation, re.IGNORECASE)
        m_expl = re.search(r'explanation:\s*(.+)', explanation, re.IGNORECASE | re.DOTALL)

        if m_label:
            label = m_label.group(1).strip()
        if m_conf:
            try:
                confidence = float(m_conf.group(1))
                if confidence > 1:
                    confidence = min(confidence / 100.0, 1.0)
            except Exception:
                confidence = 0.0
        if m_expl:
            explanation = m_expl.group(1).strip()
    except Exception as e:
        explanation = f"Gemini REST API request failed: {e}"

    return label, confidence, explanation

if __name__ == '__main__':
    main()