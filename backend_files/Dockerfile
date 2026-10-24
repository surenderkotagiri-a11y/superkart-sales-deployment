# Use an official lightweight Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /workspace

# Copy dependency list
COPY requirements.txt .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the serialized unified machine learning model pipeline
COPY app.py .
COPY superkart_model.joblib .

# Expose port 7860 for incoming HTTP API requests
EXPOSE 7860

# Define the command to run the application using Python
CMD ["python", "app.py"]
