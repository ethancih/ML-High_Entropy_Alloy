#   LAST UPDATED: 2024 May 24th

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

DATASET = 3
#   1 for 'Project_1_dataset-DonePrepared_RAW.csv'
#   2 for 'Project_1_dataset-DonePrepared_EXTRADATA_LOG.csv'
#   3 for 'Project_1_dataset-DonePrepared_EXTRADATA_SIX_LOG.csv'
LOG_BASE = 1.05 #   log base for a couple of terms

#################### \/ PREPARING THE DATASET FOR USE \/ ####################
if DATASET == 1:
    df = pd.read_csv('./Project_1_dataset.csv', index_col=None)  # read file into dataframe
    # df.drop("Elemental content, at%", axis=1, inplace=True)  # drop factor from axis 1 and make changes permanent by inplace=True
    # df.drop("Alloy No.",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True
    df.rename(columns={'Unnamed: 1': 'Alloy No.', 'Elemental content, at%': 'Al', 'Unnamed: 2': 'Co',
                       'Unnamed: 3': 'Cr', 'Unnamed: 4': 'Cu', 'Unnamed: 5': 'Fe', 'Unnamed: 6': 'Ni'}, inplace=True)
    df.drop([0], axis=0, inplace=True)
    df.insert(0, 'ARBITRARY GROUP', '1')
    print(df)
    df.to_csv('./Project_1_dataset-DonePrepared_RAW.csv', index=False)
elif DATASET == 2 or DATASET ==3:
    df = pd.read_csv('./Project_1_dataset.csv', index_col=None)  # read file into dataframe
    # df.drop("Elemental content, at%", axis=1, inplace=True)
    df.rename(columns={'Unnamed: 1': 'Alloy No.', 'Elemental content, at%': 'Al', 'Unnamed: 2': 'Co',
                       'Unnamed: 3': 'Cr', 'Unnamed: 4': 'Cu', 'Unnamed: 5': 'Fe', 'Unnamed: 6': 'Ni'}, inplace=True)
    df.drop([0], axis=0, inplace=True)

    alloy_list = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni']
    # Shear Modulus (Pa)
    g_Al = 2.4e10; g_Co = 8.26e10; g_Cr = 1.15e11; g_Cu = 4.6e10; g_Fe = 7.75e10; g_Ni = 7.6e10
    # Cohesive Energy (kJ/mol)
    ec_al = 3.27e5; ec_co = 4.24e5; ec_cr = 3.95e5; ec_cu = 3.36e5; ec_fe = 4.13e5; ec_ni = 4.28e5
    # Work Function
    w_al = 4.08; w_co = 5; w_cr = 5; w_cu = 4.7; w_fe = 4.5; w_ni = 5.01
    # Valence Electron
    vec_al = 3; vec_co = 9; vec_cr = 6; vec_cu = 11; vec_fe = 8; vec_ni = 10
    # No of electrons
    ea_al = 13; ea_co = 27; ea_cr = 24; ea_cu = 29; ea_fe = 26; ea_ni = 28

    df['modulus_mismatch'] = 0
    df['shear_modulus'] = (df['Al'].astype('float') / 100 * g_Al + df['Co'].astype('float') / 100 * g_Co +
                           df['Cr'].astype('float') / 100 * g_Cr + df['Cu'].astype('float') / 100 * g_Cu +
                           df['Fe'].astype('float') / 100 * g_Fe + df['Ni'].astype('float') / 100 * g_Ni)
    for j in alloy_list:
        df['modulus_mismatch'] += (
                (df[f'{j}'].astype('float') / 100 * ((2 * (globals()[f'g_{j}'] -
                df['shear_modulus'].astype('float'))) / (globals()[f'g_{j}']) +
                df['shear_modulus'].astype('float'))) / (1 + (0.5 * abs(df[f'{j}'].astype('float')) * ((2 * (globals()[f'g_{j}'] -
                df['shear_modulus'].astype('float'))) / (globals()[f'g_{j}']) + df[
                'shear_modulus']))))
    df['cohesive_energy'] = (df['Al'].astype('float') / 100 * ec_al + df['Co'].astype('float') / 100 * ec_co +
                             df['Cr'].astype('float') / 100 * ec_cr + df['Cu'].astype('float') / 100 * ec_cu +
                             df['Fe'].astype('float') / 100 * ec_fe + df['Ni'].astype('float') / 100 * ec_ni)
    df['6th_square_of_work_function'] = ((df['Al'].astype('float') / 100 * w_al) ** 6 +
                                         (df['Co'].astype('float') / 100 * w_co) ** 6 +
                                         (df['Cr'].astype('float') / 100 * w_cr) ** 6 +
                                         (df['Cu'].astype('float') / 100 * w_cu) ** 6 +
                                         (df['Fe'].astype('float') / 100 * w_fe) ** 6 +
                                         (df['Ni'].astype('float') / 100 * w_ni) ** 6)
    if DATASET == 3:
        df['config_entropy'] = -1.5 * (
                (df['Al'].astype('float') / 100 * np.where(df['Al'].astype('float') != 0, np.log(df['Al'].astype('float') / 100), 0)) +
                (df['Co'].astype('float') / 100 * np.where(df['Co'].astype('float') != 0, np.log(df['Co'].astype('float') / 100), 0)) +
                (df['Cr'].astype('float') / 100 * np.where(df['Cr'].astype('float') != 0, np.log(df['Cr'].astype('float') / 100), 0)) +
                (df['Cu'].astype('float') / 100 * np.where(df['Cu'].astype('float') != 0, np.log(df['Cu'].astype('float') / 100), 0)) +
                (df['Fe'].astype('float') / 100 * np.where(df['Fe'].astype('float') != 0, np.log(df['Fe'].astype('float') / 100), 0)) +
                (df['Ni'].astype('float') / 100 * np.where(df['Ni'].astype('float') != 0, np.log(df['Ni'].astype('float') / 100), 0)))
        df['vec'] = (df['Al'].astype('float') / 100 * vec_al + df['Co'].astype('float') / 100 * vec_co +
                     df['Cr'].astype('float') / 100 * vec_cr + df['Cu'].astype('float') / 100 * vec_cu +
                     df['Fe'].astype('float') / 100 * vec_fe + df['Ni'].astype('float') / 100 * vec_ni)
        df['itinerant_electrons'] = (df['Al'].astype('float') / 100 * ea_al + df['Co'].astype('float') / 100 * ea_co +
                                     df['Cr'].astype('float') / 100 * ea_cr + df['Cu'].astype('float') / 100 * ea_cu +
                                     df['Fe'].astype('float') / 100 * ea_fe + df['Ni'].astype('float') / 100 * ea_ni)

    # LOGATHRMIC-FY LARGE TERM
    # df['shear_modulus'] = np.log10(df['shear_modulus']) # actually not used for training, but I'm doing it anyway
    df['shear_modulus'] = np.emath.logn(LOG_BASE, (df['shear_modulus'])) # actually not used for training, but I'm doing it anyway
    df['cohesive_energy'] = np.emath.logn(LOG_BASE, (df['cohesive_energy']))

    print(df)
    if DATASET == 2:
        df.to_csv('./Project_1_dataset-DonePrepared_EXTRADATA_LOG.csv', index=False)
    elif DATASET == 3:
        df.to_csv('./Project_1_dataset-DonePrepared_EXTRADATA_SIX_LOG.csv', index=False)

else:
    print("YOU HAVE ASSIGNED AN INVALID DATASET NUMBER!")
#################### ^ PREPARING THE DATASET FOR USE ^ ####################