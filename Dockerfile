# Stage 1: Use Poetry to export a requirements.txt file
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install "poetry==1.8.2"
COPY pyproject.toml poetry.lock* ./

# In case poetry.lock is missing, generate it
RUN poetry lock --no-cache

# Export production dependencies to a requirements.txt file
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev

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

# Command to run the application as a module
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]