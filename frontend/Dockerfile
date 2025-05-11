FROM traffic-base:latest

WORKDIR /app

# Copy all frontend files (app.py, requirements.txt, etc.)
COPY . /app

# Ensure Streamlit is installed (if not in base image)
RUN pip install --no-cache-dir -r requirements.txt

# Optional: avoid Streamlit from opening in browser inside container
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS false

# Ensure this static mount works (if user downloads media etc.)
RUN mkdir -p /app/static

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
