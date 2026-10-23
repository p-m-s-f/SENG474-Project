# SENG474-Project

This repo is home to our SENG 474 term project. Its authors are Parris Mook-Sang-Forbes, Max Patchell, and Varun Manon.

Our goal is to classify aerial wildfire images into three operational risk tiers: low, medium, and high. False positive detections are minimized by training a model to differentiate genuine fire development from confounding environmental variables (e.g., fog, weather, terrain, foliage). This ensures critical firefighting resources are deployed based on actual threats rather than algorithmic errors.


## Setup

The initial setup was tested with Python 3.14.7.

### 1. Clone the repository

```bash
git clone https://github.com/p-m-s-f/SENG474-Project.git
cd SENG474-Project
```

Skip if repository is already cloned.

### 2. Choose a Python environment

Use your preferred Python environment (venv, Conda, etc.).

If you don’t have a preference, create and activate a virtual environment:

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

From the project root, with a python environment active:

```bash
python -m pip install -r requirements.txt
```

### 4. Run the project

```bash
python src/main.py
```

**Current status:** Script prints the project name and installed PyTorch/Torchvision versions.
