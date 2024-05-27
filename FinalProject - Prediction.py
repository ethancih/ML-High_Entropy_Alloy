#   LAST UPDATED: 2024 May 26th

from mastml.mastml import Mastml
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel
from mastml.preprocessing import SklearnPreprocessor,NoPreprocessor
from mastml.data_splitters import SklearnDataSplitter, NoSplit,JustEachGroup
from mastml.mastml_predictor import make_prediction
import pandas as pd
import os

############################################################################
####### Dataset pre-prepared by FinalProject - DatasetPreparation.py #######
############################################################################

DATASET = 3
#   1 for 'Project_1_dataset-DonePrepared_RAW.csv'
#   2 for 'Project_1_dataset-DonePrepared_EXTRADATA_LOG.csv'
#   3 for 'Project_1_dataset-DonePrepared_EXTRADATA_SIX_LOG.csv'
#   4 for 'Project_1_dataset-DonePrepared_EXTRADATA_SIX_LOG-ExcludingSomeData.csv'
#   This set of hyperparameter is of set 17
ALPHA = 0.01128837891684690
GAMMA = 0.8858667904100820

#################### \/ Designate training dataset \/ ####################
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

Testing_Sets = 10
TopCandidates = pd.DataFrame()


for i in range(Testing_Sets):
    # Basic Setup
    SAVEPATH = './20240527/Prediction-' + str(i+1)
    target = 'Hardness, HV'
    # extra_columns = ['Alloy No.']
    preprocessor = SklearnPreprocessor(preprocessor='MinMaxScaler', as_frame=True)
    model_alpha = ALPHA  # For GKRR
    model_gamma = GAMMA  # For GKRR
    Training_set = FILEPATH

    #*------Main_code
    mastml_instance = Mastml(savepath=SAVEPATH)
    savepath = mastml_instance.get_savepath
    #Load the Training set
    d = LocalDatasets(file_path=Training_set,
                      target=target,
                      feature_names=FEATURES,
                      extra_columns=EXTRA_COLUMNS,
                      group_column='Alloy No.',
                      testdata_columns=None,
                      as_frame=True)
    data_dict = d.load_data()
    X = data_dict['X']
    INDEXX=[]
    for col in X.columns:
        INDEXX.append(col)
    y = data_dict['y']
    X['TYPE']='TRAIN'

    #Load the Testing set
    d2 = LocalDatasets(file_path='gen_test_extra_data_' + str(i+1) + '.csv',
                      target=target,
                      feature_names=FEATURES,
                      # extra_columns=EXTRA_COLUMNS, # NEED TO COMMENT IF USING _RAW.csv
                      as_frame=True)
    data_dict1 = d2.load_data()
    X1 = data_dict1['X']
    y1 = data_dict1['y']
    X1['TYPE']='TEST'


    # Data Concat
    XX=pd.concat([X,X1],axis=0)
    XX_n=XX['TYPE']
    XX_t=XX.loc[:, INDEXX]

    # Data Normalize
    preprocessor.fit(XX_t)
    XX_tt=preprocessor.transform(XX_t)
    XX_total=pd.concat([XX_tt,XX_n],axis=1)

    # Data Split
    X_test=XX_total.loc[(XX_total['TYPE']=='TEST')]
    X_test=X_test.loc[:,INDEXX]
    X_train=XX_total.loc[(XX_total['TYPE']=='TRAIN')]
    X_train=X_train.loc[:,INDEXX]


    #Performing Prediction
    #Model_selection:GKRR
    model_gkrr = SklearnModel(model='KernelRidge', kernel='rbf', alpha=model_alpha, gamma=model_gamma)
    preprocessor = NoPreprocessor()
    metrics = ['r2_score', 'mean_absolute_error', 'root_mean_squared_error', 'rmse_over_stdev']
    splitter = NoSplit()
    splitter.evaluate(X=X_train,
                      y=y,
                      models=[model_gkrr],
                      mastml=mastml_instance,
                      preprocessor=preprocessor,
                      metrics=metrics,
                      plots=['Scatter', 'Histogram'],
                      savepath=savepath,
                      verbosity=2)
    path_fullfit = splitter.splitdirs[0]
    model_path = os.path.join(path_fullfit, 'KernelRidge.pkl')
    preprocessor_path = os.path.join(path_fullfit, 'NoPreprocessor.pkl')

    #Make Prediction
    pred_df = make_prediction(X_test=X_test,
                              model=model_path,
                              preprocessor=preprocessor_path)
    FinalTable=pd.concat([X1,pred_df],axis=1)
    FinalTable=pd.concat([FinalTable,y1],axis=1)
    print('----Prediction Results----')
    print(FinalTable)
    print('----End Results----')
    FinalTable.to_csv(savepath+'/PREDICTION_RESULT_'+ str(i+1) +'.csv')
    # TopValues = FinalTable.loc[(FinalTable['y_pred'] >= FinalTable['y_pred'].max() - 1)]
    TopValues = FinalTable.loc[(FinalTable['y_pred'] >= 775)]
    TopValues.sort_values(by=['y_pred'])
    TopCandidates = pd.concat([TopCandidates, TopValues])
    print(TopValues)
TopCandidates.sort_values(by=['y_pred'])
print(TopCandidates)
TopCandidates.to_csv(savepath+'/TopCandidates.csv')