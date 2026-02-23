.PHONY: setup clean create-directories install-system-deps create-venv install-packages

# =============================================================================
# Variables Definition
# =============================================================================
PYTHON := python3
VENV := .venv
V_PIP := $(VENV)/bin/pip
V_PYTHON := $(VENV)/bin/python

# Default target when simply running `make`
all: setup

# 1. Clean up existing virtual environment
clean:
	@echo "Cleaning up existing virtual environment..."
	@rm -rf $(VENV)

# 2. Create output directory
create-directories:
	@echo "Creating output directory..."
	@mkdir -p output

# 3. Install system-level dependencies for OpenCV
install-system-deps:
	@echo "Installing system-level dependencies..."
	@sudo apt-get update && sudo apt-get install -y libjpeg-dev zlib1g-dev libpng-dev libgl1

# 4. Create virtual environment
create-venv:
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@$(V_PIP) install --upgrade pip

# 5. Generate strict requirements.txt and install packages
install-packages: create-venv
	@echo "Generating strict requirements.txt for SIFT standalone..."
	@echo "opencv-python" > requirements.txt
	@echo "numpy" >> requirements.txt
	@echo "tqdm" >> requirements.txt
	@echo "matplotlib" >> requirements.txt
	@echo "scikit-image" >> requirements.txt
	
	@echo "Installing libraries from requirements.txt..."
	@$(V_PIP) install -r requirements.txt

# =============================================================================
# Master Setup Target
# =============================================================================
setup: clean create-directories install-system-deps install-packages
	@echo "=================================================="
	@echo "Minimal Setup completed successfully!"
	@echo "Virtual environment is ready for SIFT Matching."
	@echo "To activate, run: source $(VENV)/bin/activate"
	@echo "=================================================="