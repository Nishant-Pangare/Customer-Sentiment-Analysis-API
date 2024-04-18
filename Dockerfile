# Use the official Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port number on which the Flask app will run
EXPOSE 4000

# Define environment variable
ENV FLASK_APP=Customer Sentiment Database API.py

# Run the Flask API using the python command
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=4000"]
