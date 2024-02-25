FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .


# Install the Python dependencies
RUN pip install -r requirements.txt


# Copy the rest of the application code into the container
COPY . .

# Expose the port on which the application will run
EXPOSE 5000

CMD ["python","app.py"]