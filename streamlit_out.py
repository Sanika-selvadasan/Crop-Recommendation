import streamlit as st
import numpy as np
from ml_models import model_accuracies, rfc
from crop_recommendation import ms
import groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Streamlit UI
def main():
    st.title("Crop Recommendation System")
    st.write("This tool recommends the best crop to cultivate based on soil and environmental factors.")

    # Input fields
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, value=50.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, value=50.0)
    k = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, value=50.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    
    if st.button("Recommend Crop"):
        features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
        transformed_features = ms.transform(features)
        
        # # Evaluate all models
        # model_accuracies = {}
        # for name, model in models.items():
        #     accuracy = model.score(ms.transform(features), rfc.predict(ms.transform(features)))
        #     model_accuracies[name] = accuracy
        

        best_prediction = rfc.predict(transformed_features)[0]
        
        # Crop dictionary
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
        recommended_crop = crop_dict.get(best_prediction)

        st.subheader(f"Recommended Crop: {recommended_crop}")
        
        # Show model performances
        st.subheader("Model Accuracies")
        st.write("Below are the accuracies of different models used:")
        for model, accuracy in model_accuracies.items():
            st.write(f"{model}: {accuracy:.4f}")
        
        # Generate LLM explanation
        if GROQ_API_KEY:
            prompt = f"""
            Explain why {recommended_crop} is recommended based on these conditions:
            Nitrogen: {N}, Phosphorus: {P}, Potassium: {k}, Temperature: {temperature}, 
            Humidity: {humidity}, pH: {ph}, Rainfall: {rainfall}.
            Provide detailed information about the suitability of these conditions for growing {recommended_crop} And include optimal Farming Practices and Irrigation Needs for {recommended_crop} under these two topics "optimal Farming Practices" and "Irrigation needs".
            """

            client = groq.Groq(api_key=GROQ_API_KEY)
            chat_completion = client.chat.completions.create(
                model="llama3-70b-8192", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            explanation = chat_completion.choices[0].message.content
            
            st.subheader("LLM Explanation")
            st.write(explanation)
        else:
            st.warning("Groq API Key is not set. Please configure it to generate LLM explanations.")

if __name__ == "__main__":
    main()
