FROM python:3.8-slim

WORKDIR /app

# Copy your files into the container
COPY ./app ./app
COPY ./model ./model
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Use a different port
EXPOSE 8502

# Run Streamlit app on new port
CMD ["streamlit", "run", "app/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
