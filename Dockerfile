# 1. Use an official, lightweight Python base image
FROM python:3.12-slim

# 2. Prevent Python from writing .pyc files and buffer outputs for real-time logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system build dependencies required for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code, trained models, and dataset into the container
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# 7. Expose port 8000 for the FastAPI web server
EXPOSE 8000

# 8. Command to launch the FastAPI app using Uvicorn when the container starts
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]