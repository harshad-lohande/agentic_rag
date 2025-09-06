.PHONY: up restart build recreate up-build up-gpu down logs ps pull-models test-ollama check-gpu logs-backend

# Start (CPU)
up:
	docker compose up -d

restart:
	docker compose down && docker compose up -d

# Recreate containers (useful if Dockerfile changes)
recreate:
	docker compose up -d --force-recreate

# Build images (without starting containers)
build:
	docker compose build

# Build and start containers
up-build:
	docker compose up -d --build

# Start with GPU (uses override file)
up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

logs-backend:
	docker compose logs -f backend

ps:
	docker compose ps

# Pull one or more models into the Ollama cache volume
pull-models:
	# Change/add models as needed
	docker compose exec ollama ollama pull llama3.1:8b

# Quick sanity checks against Ollama from the backend
test-ollama:
	docker compose exec backend sh -lc 'curl -s http://ollama:11434/api/version && echo'
	docker compose exec backend sh -lc 'curl -s http://ollama:11434/api/generate -H "Content-Type: application/json" -d '\''{"model":"llama3.1:8b","prompt":"Say hello in one sentence."}'\''' | sed 's/\\n/\\n/g'

# Check that GPUs are visible in the Ollama container
check-gpu:
	# If present, you should see /dev/nvidia* devices listed
	docker compose exec ollama sh -lc 'ls -l /dev/nvidia* || true'