#   LAST UPDATED: 2024 May 26th

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
ALPHA_LOGSPACE = np.logspace(-3, -1, 20)
GAMMA_LOGSPACE = np.logspace(-3, 1, 20)

#################### \/ HYPERPARAMETERS \/ ####################
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
SAVEPATH = './20240526/Hyperparameter'
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


##Choose either way to create the matrix of alpha and gamma.
# kr_param_grid = {"alpha": [100, 50, 25, 10, 5, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
#                  "gamma": [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
kr_param_grid = {"alpha": ALPHA_LOGSPACE, "gamma": GAMMA_LOGSPACE}

##Model assign
#Using 5-fold CV as the scoring metrix
# model_krr = KernelRidge(kernel='rbf')
# clf = GridSearchCV(estimator=model_krr, param_grid=kr_param_grid,
#                        cv=5,    # 5-fold CV
#                        n_jobs=1,   # hardware acceleration (activates all cores of CPU to do this job)
#                        scoring='neg_root_mean_squared_error', refit=True)

#Using Leave-out-group CV as the scoring metrix
model_krr = KernelRidge(kernel='rbf')
clf = GridSearchCV(estimator=model_krr, param_grid=kr_param_grid,
                       cv=LeaveOneGroupOut().get_n_splits(groups=groups), # only difference from 5-fold
                       n_jobs=1,
                       scoring = 'neg_root_mean_squared_error', refit=True)

min_max_scaler = preprocessing.MinMaxScaler()     # normalize
# min_max_scaler = preprocessing.StandardScaler()   # standardize
X_train_minmax = min_max_scaler.fit_transform(X)

##Model fit to the optimized hyperparameter and return the results
clf.fit(X_train_minmax, y)
data=clf.cv_results_
for key in data:
    c = dict((k, data[k]) for k in ('mean_test_score', 'std_test_score'))
    out = pd.DataFrame(c, data['params'])
    try:
        best = pd.DataFrame(clf.best_params_, index=['Best Parameters'])
    except:
        best = pd.DataFrame(clf.best_params_)
    out.to_excel(os.path.join(savepath, 'output.xlsx'))
    best.to_excel(os.path.join(savepath, 'bestparams.xlsx'))