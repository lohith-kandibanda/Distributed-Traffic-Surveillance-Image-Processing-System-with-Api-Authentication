FROM traffic-base:latest

WORKDIR /app

COPY . /app

# Optional: this is only needed if your requirements.txt has extra frontend-specific packages not in base
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
