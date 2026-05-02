FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create dirs
RUN mkdir -p instance models mlflow_runs

# Train model at build time
RUN cd models && python train_model.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/api/stats || exit 1

# Run
CMD ["gunicorn", "app:app", "--workers=2", "--timeout=120", "--bind=0.0.0.0:5000"]
