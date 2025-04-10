# Makefile for Vector Store Creation
.PHONY: store clean all update-lock install-deps check-deps

# Configuration
REDIS_CONTAINER_NAME = redis-stack
REDIS_HOST_PORT = 6380
REDIS_CONTAINER_PORT = 6379
REDIS_UI_HOST_PORT = 8002
REDIS_UI_CONTAINER_PORT = 8001
DATA_VOLUME_PATH = /home/jamessmithies/Dropbox/Technical/projects/aiinfra/aiinfra-create-store
CONTAINER_DATA_PATH = /data
INDEX_NAME = blert_2000

# Main target - builds the vector store and generates statistics
store: setup_redis check-deps
	@echo "Creating vector store and generating statistics..."
	python3 blert_HNSW.py



# Clean up
clean:
	@echo "Cleaning up..."
	rm -f checkpoint_*.json
	docker stop $(REDIS_CONTAINER_NAME) || true
	docker rm $(REDIS_CONTAINER_NAME) || true

# Setup Redis container
setup_redis:
	@echo "Setting up Redis container..."
	@if [ "$$(docker ps -q -f name=$(REDIS_CONTAINER_NAME))" ]; then \
		echo "Redis container is already running"; \
	elif [ "$$(docker ps -aq -f status=exited -f name=$(REDIS_CONTAINER_NAME))" ]; then \
		echo "Starting existing Redis container..."; \
		docker start $(REDIS_CONTAINER_NAME); \
	else \
		echo "Creating new Redis container..."; \
		docker run -d \
			--name $(REDIS_CONTAINER_NAME) \
			-p $(REDIS_HOST_PORT):$(REDIS_CONTAINER_PORT) \
			-p $(REDIS_UI_HOST_PORT):$(REDIS_UI_CONTAINER_PORT) \
			-v $(DATA_VOLUME_PATH):$(CONTAINER_DATA_PATH) \
			redis/redis-stack; \
	fi
	@echo "Waiting for Redis to initialize..."
	@sleep 5

# Run everything
all: clean check-deps setup_redis store

# Install dependencies from lock file (preferred method for consistent environment)
install-deps:
	@echo "Installing dependencies from lock file..."
	pip3 install -r requirements.lock

# Check if all dependencies are installed correctly
check-deps:
	@echo "Checking if essential dependencies are installed..."
	@if [ ! -f requirements.lock ]; then \
		echo "Warning: requirements.lock file missing. Using installed packages."; \
	fi
	@# Check only critical dependencies instead of all
	@python3 -c "import langchain, redis, transformers, nltk, torch" || \
		(echo "Critical dependencies missing. Run 'make install-deps'" && exit 1)
	@echo "Essential dependencies verified."

# Generate lock file - use only when intentionally updating dependencies
update-lock:
	@echo "WARNING: This will update the lock file. Only use when intentionally updating dependencies."
	@echo "Generating requirements.lock file..."
	pip-compile --resolver=backtracking requirements.txt --output-file requirements.lock

# Show help
help:
	@echo "Vector Store Creation Makefile"
	@echo "------------------------------"
	@echo "Available targets:"
	@echo "  store         - Set up Redis, create the vector store, and generate statistics"
	@echo "  clean         - Stop and remove the Redis container, remove checkpoint files"
	@echo "  all           - Run clean, setup_redis, and store in sequence"
	@echo "  update-lock   - Generate or update requirements.lock file (use sparingly)"
	@echo "  install-deps  - Install dependencies from the lock file"
	@echo "  help          - Show this help message"