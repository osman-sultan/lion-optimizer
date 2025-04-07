# Lion Optimizer

## Overview

This repository contains pipelines for Natural Language Processing (NLP) and Computer Vision (CV) tasks. This guide will help you set up and run these pipelines.

## Setting Up the Environment

### 1. Create and Activate a Python Virtual Environment

First, you need to create and activate a Python virtual environment. This helps to manage dependencies and avoid conflicts.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (Linux/Mac)
source venv/bin/activate

# Activate the virtual environment (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies

Install the required Python libraries specified in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch

You will need to install PyTorch for the pipelines to work. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions to install the appropriate version for your system.

For example, to install PyTorch with CUDA 11.1, use the following command:

```bash
pip install torch torchvision torchaudio
```

## Running the NLP Pipeline

### 1. Configure the Model

You can change the transformer model by modifying the model string. (Replace `model_name` from your selected model from Hugging Face)

```python
model_name = "your_model_name_here"
```

### 2. Run the NLP Pipeline

To run the NLP pipeline, use the following command:

```bash
python nlp.py
```

## Running the CV Pipeline

### 1. Run the CV Pipeline

To run the CV pipeline, use the following command:

```bash
python ImageClassification.py  
```

