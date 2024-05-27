#   LAST UPDATED: 2024 May 26th
#   this is training

from mastml.mastml import Mastml
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel
from mastml.preprocessing import SklearnPreprocessor, NoPreprocessor
from mastml.data_splitters import SklearnDataSplitter, NoSplit
from mastml.datasets import LocalDatasets
from sklearn.model_selection import GridSearchCV,LeaveOneGroupOut
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

############################################################################
####### Dataset pre-prepared by FinalProject - DatasetPreparation.py #######
############################################################################

DATASET = 4
#   1 for 'Project_1_dataset-DonePrepared_RAW.csv'
#   2 for 'Project_1_dataset-DonePrepared_EXTRADATA_LOG.csv'
#   3 for 'Project_1_dataset-DonePrepared_EXTRADATA_SIX_LOG.csv'
#   4 for 'Project_1_dataset-DonePrepared_EXTRADATA_SIX_LOG-ExcludingSomeData.csv'
ALPHA = 0.00545559478116852
GAMMA = 0.8858667904100820

#################### \/ BASICS \/ ####################
if DATASET == 1:
    FILEPATH = './Project_1_dataset-DonePrepared_RAW.csv'
    FEATURES = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni']
    EXTRA_COLUMNS = 'ARBITRARY GROUP'
elif DATASET == 2:
    FILEPATH = './Project_1_dataset-DonePrepared_EXTRADATA_LOG.csv'
    FEATURES = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni',
                'modulus_mismatch', '6th_square_of_work_function', 'cohesive_energy']
    EXTRA_COLUMNS = 'shear_modulus'
elif DATASET == 3:
    FILEPATH = './Project_1_dataset-DonePrepared_EXTRADATA_SIX_LOG.csv'
    FEATURES = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni',
                'modulus_mismatch', '6th_square_of_work_function', 'cohesive_energy',
                'config_entropy', 'vec', 'itinerant_electrons']
    EXTRA_COLUMNS = 'shear_modulus'
elif DATASET == 4:
    FILEPATH = './Project_1_dataset-DonePrepared_EXTRADATA_SIX_LOG-ExcludingSomeData.csv'
    FEATURES = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni',
                'modulus_mismatch', '6th_square_of_work_function', 'cohesive_energy',
                'config_entropy', 'vec', 'itinerant_electrons']
    EXTRA_COLUMNS = 'shear_modulus'
else:
    print("YOU HAVE ASSIGNED AN INVALID DATASET NUMBER!")
SAVEPATH = './20240526/Training'
mastml_instance = Mastml(savepath=SAVEPATH)
savepath = mastml_instance.get_savepath
target = 'Hardness, HV'
extra_columns = ['Alloy No.']
d = LocalDatasets(file_path=FILEPATH,
                  target=target,
                  feature_names=FEATURES,
                  extra_columns=EXTRA_COLUMNS,
                  group_column='Alloy No.',
                  testdata_columns=None,
                  as_frame=True)
data_dict = d.load_data()
X = data_dict['X']
y = data_dict['y']
groups = data_dict['groups']
####################BASICS

model_gkrr = SklearnModel(model='KernelRidge', kernel='rbf', alpha=ALPHA, gamma=GAMMA)
# model_gkrr = SklearnModel(model='LinearRegression') # Sklearn = Scikit-learn; built-in Scikit-learn model in MAST-ML
# model_lasso = SklearnModel(model='Lasso')
metrics = ['r2_score', 'root_mean_squared_error']


# preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)
preprocessor = SklearnPreprocessor(preprocessor='MinMaxScaler', as_frame=True)
# preprocessor = NoPreprocessor()   # No Normalize


# # #  This represents a full fit to all of the data.
splitter = NoSplit()    # 全擬合(full-fit), we fit every data in the data set, no splitting the data into training and
                        # validation
splitter.evaluate(X=X,
                  y=y,
                  models=[model_gkrr], #models
                  mastml=mastml_instance,
                  preprocessor=preprocessor, #pre-processing; built-in by Scikit-learn and MAST-ML
                  metrics=metrics,
                  plots=['Scatter', 'Histogram'],
                  savepath=savepath,
                  verbosity=2)


#### The following is copied from Demo_following_linear_model.py
#### further validating

# # 5-fold CV method
# # Note that each splitter.evaluate() method has a verbosity tag- this tag controls the extent of analysis output plotting that is performed.
# # A value of 3 is the highest, meaning that the most output is produced
splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=20, n_splits=5) # 5 => 5-fold; 20 => 5-fold 20 times
splitter.evaluate(X=X,
                  y=y,
                  models=[model_gkrr], #models
                  mastml=mastml_instance,
                  metrics=metrics,
                  preprocessor=preprocessor,
                  plots=['Scatter', 'Histogram'],
                  savepath=savepath,
                  verbosity=2)

# # # Leave-out group CV method
# # # We demonstrate group = time
# splitter = SklearnDataSplitter(splitter='LeaveOneGroupOut')
# splitter.evaluate(X=X,
#                   y=y,
#                   groups=groups,
#                   models=[model_gkrr],
#                   metrics=metrics,
#                   preprocessor=preprocessor,
#                   plots=['Scatter', 'Histogram'],
#                   savepath=savepath,
#                   verbosity=2)
################################################LEAVE_OUT GROUP currently doesn't work because my group column are ones