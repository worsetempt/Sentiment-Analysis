# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY docker_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r docker_requirements.txt

# Copy all necessary files to the container
COPY deploy.py .
COPY tfidf_vectorizer.pkl .
COPY model.pkl .
COPY xgb_model.pkl .

# Expose port 5000 for Flask
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "deploy.py"]
