# Amazon ML Challenge - Price Prediction Model

This repository contains the code and resources for the Amazon ML Challenge price prediction model using BGE (BERT) embeddings and XGBoost.

⚠️ **Note**: Large files (embeddings, models, and datasets) are not included in this repository due to size limitations. Please contact the repository owner for access to these files:
- Model files (fine_tuned_bge_log_price_model_3_epochs/)
- Embedding files (*.npy)
- Dataset files (*.csv)
- output file(*.csv)
- The large file also contain a .ipynb file which is same as "Last working.ipynb"
- 
##  Drive Link of compressed large files(.zip) : https://drive.google.com/file/d/1K06SCNZBXRH5acvJcVkWwSsCG1ke7RCf/view?usp=sharing
##  You need access permission. Request access. You may also mail on: "ut7320@gmail.com" to ask access permission.

## Recommended: Setup, activate and use a Virtual Enviroment for Dependency Isolation.  It prevent any version compatibility error.

## Project Structure
```
BGE_XGB_bestPricePredictor/
├── data/
│   ├── train_csv.csv
│   └── test_csv.csv
├── models/
│   └── fine_tuned_bge_log_price_model_3_epochs/
├── embeddings/
│   ├── embeddings_finetuned_train_3_epochs.npy
│   └── embeddings_finetuned_test_3_epochs.npy
├── notebooks/
│   └── Last working.ipynb
├── output/
│   └── submission.csv
└── requirements.txt
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/AMAZON-ML-CHALLENGE.git
cd AMAZON-ML-CHALLENGE/BGE_XGB_bestPricePredictor
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- sentence_transformers
- scikit-learn
- lightgbm
- torch

## Usage
1. Place the training and test data in the `data` folder
2. Run the Jupyter notebook `notebooks/Last working.ipynb`
3. The predictions will be saved in `output/submission.csv`

## Model Description
The model uses a combination of:
- Text embeddings from fine-tuned BGE model
- Structured features from product metadata
- LightGBM regressor for final price prediction

The model is trained on log-transformed prices and achieves competitive performance on the test set.
