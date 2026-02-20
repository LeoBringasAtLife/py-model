# Digit Classifier Project

This project generates synthetic digit images, trains a neural-network classifier, and displays generated images in a web gallery.

## What it does

- Generates grayscale digit images (`0` to `9`) in `data/train` and `data/test`
- Trains an `MLPClassifier` from scikit-learn
- Prints loss and accuracy after training
- Lets you view generated images in `model.html`

## Project files

- `model.py`: dataset generation + training + evaluation
- `model.html`: browser gallery for generated images
- `model.js`: gallery rendering logic
- `requirements.txt`: Python dependencies

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the model

Run:

```bash
python model.py
```

Expected output includes:

- image generation status
- train/test image counts
- training status
- final loss and accuracy
- saved model path (`digit_model.joblib`)

## View images in the browser

1. Start a local static server:

```bash
python -m http.server 8000
```

2. Open:

- `http://localhost:8000/model.html`

## Notes

- The dataset is synthetic and regenerated on each training run.
- Accuracy can vary between runs because generated images include random noise/transformations.
