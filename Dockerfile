FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV HF_DATASETS_CACHE=/app/.cache

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
