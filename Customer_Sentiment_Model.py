import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Function to derive sentiment labels
def derive_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

# Load data
data = pd.read_csv("customer_sentiment_data.csv")

# Count non-numeric rows in "Product Rating"
non_numeric_count = len(data) - len(data[pd.to_numeric(data["Product Rating"], errors="coerce").notnull()])
print(f"Number of rows removed with non-numeric Product Rating values: {non_numeric_count}")

# Filter out rows with non-numeric values
data = data[pd.to_numeric(data["Product Rating"], errors="coerce").notnull()]

# Convert "Product Rating" to integer
data["Product Rating"] = data["Product Rating"].astype(int)

# Clean and preprocess text (reviewText column)
data["Cleaned Review"] = data["reviewText"].str.lower().replace("[^a-zA-Z0-9 ]", "", regex=True)


# Create sentiment labels based on Product Rating
data["Sentiment"] = data["Product Rating"].apply(derive_sentiment)

# Calculate sentiment distribution
sentiment_counts = data["Sentiment"].value_counts()
total_reviews = len(data)
sentiment_percentages = (sentiment_counts / total_reviews) * 100

# Print sentiment distribution
print("\n**Sentiment Distribution in Training Data:**")
for sentiment, percentage in sentiment_percentages.items():
    print(f"{sentiment}: {percentage:.2f}%")

# Feature engineering
vectorizer = TfidfVectorizer(max_features=1000)
features = vectorizer.fit_transform(data["Cleaned Review"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, data["Sentiment"], test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(solver="lbfgs")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Overall Accuracy:", int(accuracy * 100), "%")
print(classification_report(y_test, y_pred))

# Predict sentiment for sample reviews
positive_review = "This product is amazing! I love it."
neutral_review = "This product is okay. It does the job."
negative_review = "This product is terrible. I would not recommend it."

# Preprocess and predict sentiment
reviews = [positive_review, neutral_review, negative_review]
preprocessed_reviews = [review.lower().replace("[^a-zA-Z0-9 ]", "", True) for review in reviews]
new_features = vectorizer.transform(preprocessed_reviews)
predicted_sentiments = model.predict(new_features)

# Format output for retailer
print("\n**Sample Review Sentiment Predictions:**")
for review, sentiment in zip(reviews, predicted_sentiments):
    print(f"\nReview: {review}\nPredicted Sentiment: {sentiment}")

# Calculate metrics for each sentiment
sentiment_metrics = classification_report(y_test, y_pred, output_dict=True)
print("\n**Detailed Sentiment Performance:**")
print(pd.DataFrame(sentiment_metrics).transpose().round(3))

import joblib

# Export the model
joblib.dump(model, 'Customer_Sentiment_Model.pkl')

