# Real Estate Price Prediction - README

## Project Overview
This project predicts real estate prices using machine learning. It includes scripts for training a model and making predictions.

## File Structure
- `training.py`: Script to train the model.
- `main.py`: Script to run predictions.
- `data_loader.py`: Loads and processes data.
- `properties.csv`: Main dataset for training and prediction.
- `real_estate_model.cbm`: Saved model file after training.
- `descriptions/`: Contains property description text files.
- `images/`: Contains property images (if used).

## Training the Model
To train the model, run:

```
python training.py
```

This will use `properties.csv` and files in `descriptions/` to train and save the model as `real_estate_model.cbm`.

## Running Predictions
Before running predictions, **set your Gemini API key** as an environment variable:

```
set GEMINI_API_KEY=your_key_here
```

Then launch:

```
python main.py
```

This will use the trained model and data files to make predictions.

## Data Files
- `properties.csv`: Tabular data for properties.
- `descriptions/`: Text descriptions for each property (named by property ID).
- `images/`: Images for each property (optional).

---
**Note:** Ensure all required data files are present and the Gemini API key is set before running `main.py`.
