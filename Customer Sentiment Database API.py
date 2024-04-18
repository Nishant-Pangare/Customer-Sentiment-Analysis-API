import pandas as pd
from flask import Flask, jsonify, request
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Database Configuration for MySQL
mysql_config = {
    'dialect': 'mysql',
    'host': 'uae-staging.c5ekgugckxnm.ap-south-1.rds.amazonaws.com',
    'port': '3306',
    'user': 'admin',
    'password': 'LlTQ7RnClHM15xcji0q6',
    'database': 'iceipts_apiserver'
}

app = Flask(__name__)

# Connect to the MySQL database
def connect_to_database(config):
    if config['dialect'] == 'mysql':
        return mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )
    else:
        raise ValueError("Unsupported database dialect")

# Choose the appropriate database configuration
db_config = mysql_config  # Change to mysql_config for MySQL

# Connect to the database
conn = connect_to_database(db_config)

# Load the model and vectorizer
model = joblib.load('/app/Customer_Sentiment_Model.pkl')
vectorizer = joblib.load('/app/TFIDF_Vectorizer.pkl')

@app.route('/predict_sales/', methods=['GET'])
def predict_sales():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"message": "User ID not provided."}), 400
    
    # Fetch data from the "customers" table
    cursor = conn.cursor()
    query = f"SELECT p.id, p.itemName, r.* FROM iceipts_inventory.products p INNER JOIN iceipts_inventory.ratings r ON r.productId = p.id WHERE p.userId = '{user_id}'"
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    
    # Check if data is available
    if not data:
        return jsonify({"message": "Data not available for prediction."}), 404
    
    reviews_dict = {}
    sentiment_distribution = {}

    for row in data:
        product_id = str(row[0], 'utf-8')  # Convert byte array to string
        review = str(row[5]).lower() if row[5] else ""  

        # Transform review using the loaded vectorizer
        new_features = vectorizer.transform([review])
        
        # Predict sentiment
        predicted_sentiment = model.predict(new_features)[0]
        
        # Update sentiment counts
        if product_id not in reviews_dict:
            reviews_dict[product_id] = {
                'reviews': [],
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
        
        reviews_dict[product_id]['reviews'].append({
            'review': review,
            'predicted_sentiment': predicted_sentiment
        })

        reviews_dict[product_id]['sentiment_distribution'][predicted_sentiment.lower()] += 1

    # Aggregate sentiment counts for each product
    for product_id, product_info in reviews_dict.items():
        sentiment_counts = product_info['sentiment_distribution']
        total_reviews = sum(sentiment_counts.values())
        for sentiment in sentiment_counts:
            sentiment_counts[sentiment] = str(int((sentiment_counts[sentiment] / total_reviews) * 100)) + "%"

    # Construct response
    response = {
        'products': []
    }
    for product_id, product_info in reviews_dict.items():
        product = {
            'product_id': product_id,
            'reviews': product_info['reviews'],
            'sentiment_distribution': product_info['sentiment_distribution']
        }
        response['products'].append(product)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=4000)

