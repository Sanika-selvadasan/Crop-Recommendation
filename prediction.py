import numpy as np
from ml_models import rfc
from crop_recommendation import ms
import groq
import os
from dotenv import load_dotenv

print("prediction is running")

load_dotenv()


GROQ_API_KEY = os.environ.get("GROQ_API_KEY") # set your API key in environment variables

def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)
    prediction = rfc.predict(transformed_features)
    return prediction[0]

# User input (same as before)
N = float(input("Enter Nitrogen (N) value: "))
P = float(input("Enter Phosphorus (P) value: "))
k = float(input("Enter Potassium (K) value: "))
temperature = float(input("Enter Temperature value: "))
humidity = float(input("Enter Humidity value: "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall value: "))

predict = recommendation(N, P, k, temperature, humidity, ph, rainfall)

crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

recommended_crop = crop_dict.get(predict)

if recommended_crop:
    print(f"{recommended_crop} is a best crop to be cultivated.")

    # Generate LLM prompt
    prompt = f"""
    Explain why {recommended_crop} is recommended based on these conditions:
    Nitrogen: {N}, Phosphorus: {P}, Potassium: {k}, Temperature: {temperature}, Humidity: {humidity}, pH: {ph}, Rainfall: {rainfall}.
    Provide detailed information about the suitability of these conditions for growing {recommended_crop}.
    """

    # Use Groq API
    client = groq.Groq(api_key=GROQ_API_KEY)
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",  # Or another Llama 3 model
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
    )

    print("\nDetailed Explanation:")
    print(chat_completion.choices[0].message.content)

else:
    print("Sorry! Not able to recommend a proper crop for this environment.")