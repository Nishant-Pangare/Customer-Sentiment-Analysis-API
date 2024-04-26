import pandas as pd
from flask import Flask, jsonify, request, make_response
import mysql.connector
import joblib

# UAE Staging Database Configs
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

# Load the model and vectorizer
model = joblib.load('Customer Sentiment Analysis/Customer Sentiment/Customer_Sentiment_Model.pkl')
vectorizer = joblib.load('Customer Sentiment Analysis/Customer Sentiment/TFIDF_Vectorizer.pkl')

@app.route('/predict/', methods=['GET'])
def predict_sales():
    user_id = request.args.get('userId')
    product_id = request.args.get('productId')

    print("User ID:", user_id)
    print("Product ID:", product_id)
    
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host=mysql_config['host'],
        port=mysql_config['port'],
        user=mysql_config['user'],
        password=mysql_config['password'],
        database=mysql_config['database']
    )
    
    # Both userId and productId are provided
    if user_id and product_id:
        print("Fetching data for both user ID and product ID...")
        query = f"SELECT p.id, p.itemName, r.* FROM iceipts_inventory.products p INNER JOIN iceipts_inventory.ratings r ON r.productId = p.id WHERE p.userId = '{user_id}' AND p.id = '{product_id}'"
    # Only userId is provided
    elif user_id:
        print("Fetching data for user ID only...")
        query = f"SELECT p.id, p.itemName, r.* FROM iceipts_inventory.products p INNER JOIN iceipts_inventory.ratings r ON r.productId = p.id WHERE p.userId = '{user_id}'"
    else:
        return jsonify({"message": "Check at least User ID is provided."}), 400
    
    # Fetch data from the database based on the constructed query
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()

    print(data)
    
    # Check if data is available
    if not data:
        return jsonify({"message": "Data not available for prediction."}), 404
    
    if product_id:  # Only one product requested
        reviews_list = []
        sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}

        for row in data:
            review = str(row[5]).lower() if row[5] else ""  

            # Transform review using the loaded vectorizer
            new_features = vectorizer.transform([review])
            
            # Predict sentiment
            predicted_sentiment = model.predict(new_features)[0]
            
            reviews_list.append({
                'review': review,
                'predicted_sentiment': predicted_sentiment
            })

            sentiment_distribution[predicted_sentiment.lower()] += 1

        total_reviews = sum(sentiment_distribution.values())
        for sentiment in sentiment_distribution:
            sentiment_distribution[sentiment] = str(int((sentiment_distribution[sentiment] / total_reviews) * 100)) + "%"

        # Construct response
        response = {
            'product_id': product_id,
            'reviews': reviews_list,
            'sentiment_distribution': sentiment_distribution
        }
    else:  # Multiple products requested
        products = {}
        
        for row in data:
            product_id = str(row[0], 'utf-8')
            product_name = str(row[1])
            review = str(row[5]).lower() if row[5] else ""
              

            if product_id not in products:
                products[product_id] = {
                    'product_name': product_name,
                    'reviews': [],
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
                }

            # Transform review using the loaded vectorizer
            new_features = vectorizer.transform([review])
            
            # Predict sentiment
            predicted_sentiment = model.predict(new_features)[0]
            
            products[product_id]['reviews'].append({
                'review': review,
                'predicted_sentiment': predicted_sentiment
            })

            products[product_id]['sentiment_distribution'][predicted_sentiment.lower()] += 1

        # Aggregate sentiment counts for each product
        for product_id, product_info in products.items():
            sentiment_counts = product_info['sentiment_distribution']
            total_reviews = sum(sentiment_counts.values())
            for sentiment in sentiment_counts:
                sentiment_counts[sentiment] = str(int((sentiment_counts[sentiment] / total_reviews) * 100)) + "%"

        # Construct response
        response = {
            'products': []
        }
        for product_id, product_info in products.items():
            product = {
                'product_id': product_id,
                'product_name': product_info['product_name'],
                'reviews': product_info['reviews'],
                'sentiment_distribution': product_info['sentiment_distribution']
            }
            response['products'].append(product)

    # Close the database connection
    conn.close()

    # Prevent caching of responses
    response = make_response(jsonify(response))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response

if __name__ == '__main__':
    app.run(debug=True, port=4000)
