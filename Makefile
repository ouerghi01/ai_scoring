# === Install dependencies ===
install:
	pip install -r requirements.txt

# === Linting ===
lint:
	black .
	flake8 .
	mypy .

# === Run tests ===
test:
	pytest

# === Run DVC pipeline ===
dvc-pipeline:
	dvc repro

# === Build Docker image ===
docker-build:
	docker build -t mlops-starter .

# === Run Docker container ===
docker-run:
	docker run -p 8000:8000 mlops-starter

# === One command for full CI ===
ci: lint test dvc-pipeline docker-build
