import json
import csv

# Replace 'your_file.json' with the actual path to your JSON file
json_file = 'Customer Sentiment Analysis\Clothing_Shoes_and_Jewelry_5.json'

# Define the expected keys (modify as needed)
expected_keys = ['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin',
                  'style', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime']

# Open the file in read mode (`'r'`)
with open(json_file, 'r') as f:
    # Initialize the CSV writer
    csv_writer = csv.writer(open('customer_sentiment_data.csv', 'w', newline=''))

    # Write the column names as the first row
    csv_writer.writerow(expected_keys)

    # Counter for keeping track of processed rows
    row_count = 0

    # Read the file line by line
    for line in f:
        # Load the JSON data from the line
        data = json.loads(line)

        # Check if all expected keys are present
        if all(key in data for key in expected_keys):
            # Write the data to the CSV file (transpose keys and values)
            csv_writer.writerow(data.values())

            # Increment the counter
            row_count += 1

        # Stop after processing 5000 rows
        if row_count >= 5000:
            break

print(f"First 5000 rows with complete data successfully converted to CSV and stored in 'customer_sentiment_data.csv'.")
