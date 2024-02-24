FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .
RUN apt-get update && apt-get install -y curl 

# Install the Python dependencies
RUN pip install -r requirements.txt


# Install the Google Cloud SDK 
# RUN curl https://sdk.cloud.google.com | bash > /dev/null && \
#     /usr/local/google-cloud-sdk/bin/gcloud components update --quiet

# Copy the rest of the application code into the container
COPY . .

# Run the database script

# Expose the port on which the application will run
# EXPOSE 8501

# Run the FastAPI application using uvicorn server
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["python","temp.py"]