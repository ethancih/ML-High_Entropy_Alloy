# ML-High_Entropy_Alloy
**What it is:**  
- An uni final project - training and predicting high entropy alloys with good hardness and low cost.
  
**What it uses:**  
1. Mast-ML 3.1.7
2. The research paper: https://doi.org/10.1016/j.actamat.2019.03.010

## Model Records:
I recorded everything in this Google Sheet:  
https://docs.google.com/spreadsheets/d/12B4Szen0w6W-l8HUELGQ7lpmx_uVusodcEEF_xwrpoc/edit?usp=sharing  

### This file includes:
1. All the hyperparameters I have tested.
2. The results predicting and testing a separate set of data we found elsewhere on the Internet.
3. The prices of all 6 metals.
4. The final prediction results - PredictedResults.csv - in the 4th worksheet.
5. The folder I have in the end.

## Difficulties Encountered:
1. The training data the paper provides does not include every parameter that influences hardness, e.g. molecular composition, method of manufacture.
2. It is hard to find testing data on the Internet.

## Model Performance:
1. Training accuracy: 93.3559%
2. Testing accuracy: 96.6041%
3. Testing with 13 alloys from other papers: 81.5161%

## Results:
1. Generated 620,247 alloys with predicted hardness.
2. According to materials costs found online (units: USD/KG).
3. Found the best cost-efficient alloy is Al37Cr34Ni16Fe8Cu5 (wt%) with a predicted hardness of 777.2160 HV and a unit price of 1152.9838 USD/KG.
