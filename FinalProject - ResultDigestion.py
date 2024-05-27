#   LAST UPDATED: 2024 May 26th
#   This program is to augment the TopCandidates.csv result from "FinalProject - Prediction.py"
#   to help us decide which alloy compositions are the best.

from mastml.mastml import Mastml
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel
from mastml.preprocessing import SklearnPreprocessor,NoPreprocessor
from mastml.data_splitters import SklearnDataSplitter, NoSplit,JustEachGroup
from mastml.mastml_predictor import make_prediction
import pandas as pd
import os

#######################################################################
####### Dataset pre-generated with FinalProject - Prediction.py #######
#######################################################################
df = pd.read_csv('./PredictedResults/TopCandidates.csv', index_col=None)  # read file into dataframe
df.drop(['TYPE', 'Hardness, HV'],axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True
# df.drop(['modulus_mismatch', '6th_square_of_work_function', 'cohesive_energy', 'config_entropy',
#           'vec', 'itinerant_electrons'],axis=1,inplace=True)
df.rename(columns={'y_pred': 'Predicted Hardness'}, inplace=True)


### Prices of each metal (unit: USD per KG)
p_Al = 2.61658; p_Co = 29.70177827; p_Cr = 7.75; p_Cu = 10.20459; p_Fe = 0.12055; p_Ni = 20.01846

df['Price'] = (df['Al'].astype('float')*p_Al + df['Co'].astype('float')*p_Co +
               df['Cr'].astype('float')*p_Cr + df['Cu'].astype('float')*p_Cu +
               df['Fe'].astype('float')*p_Fe + df['Al'].astype('float')*p_Ni)
df['CP Value'] = ((df['Predicted Hardness']*200) / df['Price'])

df.sort_values(by=['CP Value'], ascending=False) # This sorting does not work, so will need to sort manually in Excel.
print(df)

df.to_csv('./PredictedResults/DigestedResult.csv', index=False)