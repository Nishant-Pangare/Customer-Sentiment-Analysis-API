import pandas as pd
from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('Customer_Sentiment_Model.pkl')

# Load training data
data = pd.read_csv("customer_sentiment_data.csv")

# Clean and preprocess text (reviewText column)
data["Cleaned Review"] = data["reviewText"].str.lower().replace("[^a-zA-Z0-9 ]", "", regex=True)

# Convert "Product Rating" to numeric type
data["Product Rating"] = pd.to_numeric(data["Product Rating"], errors="coerce")

# Drop rows with NaN values in "Product Rating"
data = data.dropna(subset=["Product Rating"])

# Create sentiment labels based on Product Rating
def derive_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

data["Sentiment"] = data["Product Rating"].apply(derive_sentiment)

# Fit the vectorizer with training data
vectorizer = TfidfVectorizer(max_features=1000)
vectorizer.fit(data["Cleaned Review"])

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json  # Assuming JSON payload
    reviews = data['reviews']
    product_ratings = data['product_ratings']
    descriptions = data['descriptions']
    product_ids = data['product_ids']  # Include Product Ids
    
    # Preprocess reviews
    preprocessed_reviews = [review.lower().replace("[^a-zA-Z0-9 ]", "", True) for review in reviews]

    # Feature extraction using the loaded vectorizer
    new_features = vectorizer.transform(preprocessed_reviews)

    # Predict sentiment
    predicted_sentiments = model.predict(new_features)

    # Format output
    output = []
    for review, sentiment, product_id in zip(reviews, predicted_sentiments, product_ids):
        output.append({'product_id': product_id, 'review': review, 'predicted_sentiment': sentiment})

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
