# Stage 1: Use a specific poetry version for reproducible builds
FROM python:3.12-slim AS builder

WORKDIR /app

# Set Poetry environment variables to keep things tidy
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true

# Install poetry
RUN pip install --no-cache-dir "poetry==2.1.4"

# Copy only dependency files first to leverage Docker's layer caching
COPY pyproject.toml poetry.lock ./

# Install dependencies without installing the project itself
# --without dev ensures only production dependencies are included
RUN poetry install --no-root --without dev

# ---

# Stage 2: Create the final, lean production image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Create a non-root user and its home directory for security
RUN groupadd --system --gid 1001 appgroup && \
    useradd --system --uid 1001 --gid appgroup --create-home appuser

# Set environment variables for caching to the user's home directory
ENV HOME=/home/appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

# Copy the virtual environment with dependencies from the builder stage
COPY --from=builder /app/.venv ./.venv

# Activate the virtual environment for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code from the local 'src' directory
COPY ./src/agentic_rag ./agentic_rag
COPY ./logging.ini ./

# Change the ownership of the files to the non-root user
# Also, ensure the home directory and cache paths are owned by the user
RUN chown -R appuser:appgroup /app /home/appuser

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# We reference the api module inside our package now
CMD ["python", "-m", "uvicorn", "agentic_rag.app.api:app", "--host", "0.0.0.0", "--port", "8000"]