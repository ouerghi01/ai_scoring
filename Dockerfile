# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.main:app

# Set working directory
WORKDIR /app

# Install system dependencies for DVC and Python packages
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Initialize and run DVC pipeline (no cache/stage commit here)
RUN dvc init --no-scm -f
RUN dvc repro

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--with-threads"]


