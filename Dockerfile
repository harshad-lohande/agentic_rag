# Stage 1: Use Poetry to generate a requirements.txt file for production
FROM python:3.12-slim AS builder
WORKDIR /app

# Use a specific, known-good version of Poetry
RUN pip install "poetry==2.1.4"
COPY pyproject.toml poetry.lock* ./

# Install only production dependencies into Poetry's virtual environment
RUN poetry install --no-root --without dev

# Generate requirements.txt from the installed packages
RUN poetry run pip freeze > requirements.txt

# Stage 2: Create the final production image using pip
FROM python:3.12-slim
WORKDIR /app

# Copy the requirements.txt file from the builder stage
COPY --from=builder /app/requirements.txt .
# Install the dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app ./app
COPY ./config.py ./
COPY ./logging.ini ./

# Command to run the application
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]