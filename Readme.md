# Project Title 
Comparative Study of Traditional ML model with BERT model in Sentiment Analysis
## Description
Logisitc Regression (LR), Multi-Layer Perceptron (MLP) and BERT model are tested and compared under different settings of data (sizes, distribution, bias etc...)
## Prerequisites
See the start of src.py file to understand the packages needed for this project.
## Usage
For the fundemantal repo to see the comparison of each model on entire dataset, just run main.py. If you wish to see model's on biased data, uncomment line 2 and line 20-29 in main.py file, bias data is pre-set, if you wish to create customized biased dataset, changes needs to be made in function bias_data and bias_datasplit in src.py file.
## Note
If you're going to produce the repo in google colab, add %pip install transformers at the begining and changes the csv file path for dataset function from src.py file. Please bear in mind, a complete repo takes 15 mins for LR model, 2.6 Hours for MLP and 4 hours per epoch for BERT based on purely cpu environment.