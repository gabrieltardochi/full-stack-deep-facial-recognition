VENV = .venv
PIP = $(VENV)/bin/pip3
PYTHON = python3.10

venv: requirements.txt
	rm -rf $(VENV)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
dev-venv: requirements.txt requirements-dev.txt
	rm -rf $(VENV)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache *.egg-info .coverage** .ruff_cache build dist