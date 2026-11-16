# COMP1818 - Sentiment Analysis

## Project Overview

[Write a short description of the project in your own words.]

## Dataset

Cornell Movie Review Polarity Dataset v2.0.

- 1,000 positive reviews
- 1,000 negative reviews
- 80% training data
- 20% testing data

Dataset source:
https://www.cs.cornell.edu/people/pabo/movie-review-data/

Extract the dataset into `data/txt_sentoken/`.

## Installation

Python 3.10 is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Project

```bash
python main.py
```

## Models

- Dummy Classifier
- Multinomial Naive Bayes
- Linear SVM
- TF-IDF SVM with GridSearchCV

Trained models are saved in the `models/` directory.

## Results

Experimental results are stored in `results/results.txt`.

## Project Structure

- `main.py` - Python implementation
- `data/` - Movie review dataset
- `models/` - Trained models
- `results/` - Experimental results
- `requirements.txt` - Dependencies