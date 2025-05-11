FROM traffic-base:latest

WORKDIR /app

# Copy all contents including static/ folder
COPY . /app

# Optional: make sure static/annotated exists so mount doesn’t fail
RUN mkdir -p /app/static/annotated

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
