
#Standard Library modules
import math
import json
import warnings

#Third Party modules
from pathlib import Path
import pandas as pd 
import joblib
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split

#local modules
import utility as utils 
warnings.filterwarnings("ignore")

model_name = 'LSTM'

IOFiles = {
    "inputData": "featureEngineeredData.dat",
    "metaData": "modelMetaData.json",
    "monitor": "monitoring.json",
    "log": "LSTM_aion.log",
    "model": "LSTM_model.pkl",
    "metrics": "metrics.json",
    "metaDataOutput": "LSTM_modelMetaData.json",
    "production": "production.json"
}
        
def validateConfig():        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = utils.read_json(config_file)        
    return config

def getdlparams(config):
    return config['activation'], config['optimizer'], config['loss'], int(config['first_layer']), int(config['lag_order']), int(config['hidden_layers']), float(config['dropout']), int(config['batch_size']), int(config['epochs'])

def numpydf(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        subset = dataset[i:(i + look_back), 0]
        dataX.append(subset)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def startTraining(dataset,test_size,mlpConfig,filename_scaler,target_feature,scoreParam,log):
    log.info('Training started')
    activation_fn, optimizer, loss_fn, first_layer,  look_back,hidden_layers, dropout, batch_size, epochs= getdlparams(mlpConfig)
    n_features = len(target_feature)
    n_input = look_back

    dataset_np = dataset.values
    train, test = train_test_split(dataset_np, test_size=test_size, shuffle=False)
    generatorTrain = TimeseriesGenerator(train, train, length=n_input, batch_size=8)
    generatorTest = TimeseriesGenerator(test, test, length=n_input, batch_size=8)
    batch_0 = generatorTrain[0]
    x, y = batch_0
    epochs = int(epochs)
    ##Multivariate LSTM model
    model = Sequential()
    model.add(LSTM(first_layer, activation=activation_fn, input_shape=(n_input, n_features)))
    model.add(Dropout(dropout))
    model.add(Dense(n_features))
    model.compile(optimizer=optimizer, loss=loss_fn)
    # model.fit(generatorTrain,epochs=epochs,batch_size=self.batch_size,shuffle=False)
    model.fit_generator(generatorTrain, steps_per_epoch=1, epochs=epochs, shuffle=False, verbose=0)
    # lstm_mv_testScore_mse = model.evaluate(x, y, verbose=0)


    predictions = []
    future_pred_len = n_input
    # To get values for prediction,taking look_back steps of rows
    first_batch = train[-future_pred_len:]
    c_batch = first_batch.reshape((1, future_pred_len, n_features))
    current_pred = None
    for i in range(len(test)):
        # get pred for firstbatch
        current_pred = model.predict(c_batch)[0]
        predictions.append(current_pred)
        # remove first val
        c_batch_rmv_first = c_batch[:, 1:, :]
        # update
        c_batch = np.append(c_batch_rmv_first, [[current_pred]], axis=1)
    ## Prediction, inverse the minmax transform
    scaler = joblib.load(filename_scaler)
    prediction_actual = scaler.inverse_transform(predictions)
    test_data_actual = scaler.inverse_transform(test)
    mse = None
    rmse = None
    ## Creating dataframe for actual,predictions
    pred_cols = list()
    for i in range(len(target_feature)):
        pred_cols.append(target_feature[i] + '_pred')

    predictions = pd.DataFrame(prediction_actual, columns=pred_cols)
    actual = pd.DataFrame(test_data_actual, columns=target_feature)
    actual.columns = [str(col) + '_actual' for col in dataset.columns]
    df_predicted = pd.concat([actual, predictions], axis=1)
    print("LSTM Multivariate prediction dataframe: \n" + str(df_predicted))
    # df_predicted.to_csv('mlp_prediction.csv')
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    target = target_feature
    mse_dict = {}
    rmse_dict = {}
    mae_dict = {}
    r2_dict = {}
    lstm_var = 0
    for name in target:
        index =  dataset.columns.get_loc(name)
        mse = mean_squared_error(test_data_actual[:, index], prediction_actual[:, index])
        mse_dict[name] = mse
        rmse = sqrt(mse)
        rmse_dict[name] = rmse
        lstm_var = lstm_var + rmse
        print("Name of the target feature: " + str(name))
        print("RMSE of the target feature: " + str(rmse))
        r2 = r2_score(test_data_actual[:, index], prediction_actual[:, index])
        r2_dict[name] = r2
        mae = mean_absolute_error(test_data_actual[:, index], prediction_actual[:, index])
        mae_dict[name] = mae
    ## For VAR comparison, send last target mse and rmse from above dict
    lstm_var = lstm_var / len(target)
    select_msekey = list(mse_dict.keys())[-1]
    l_mse = list(mse_dict.values())[-1]
    select_rmsekey = list(rmse_dict.keys())[-1]
    l_rmse = list(rmse_dict.values())[-1]
    select_r2key = list(r2_dict.keys())[-1]
    l_r2 = list(r2_dict.values())[-1]
    select_maekey = list(mae_dict.keys())[-1]
    l_mae = list(mae_dict.values())[-1]
    log.info('Selected target feature of LSTM for best model selection: ' + str(select_rmsekey))
    scores = {}
    scores['R2'] = l_r2
    scores['MAE'] = l_mae
    scores['MSE'] = l_mse
    scores['RMSE'] = l_rmse
    scores[scoreParam] = scores.get(scoreParam.upper(), scores['MSE'])
    log.info("lstm rmse: "+str(l_rmse))
    log.info("lstm mse: "+str(l_mse))
    log.info("lstm r2: "+str(l_r2))
    log.info("lstm mae: "+str(l_mae))

    return model,look_back,scaler, scores

def train(config, targetPath, log):
    dataLoc = targetPath / IOFiles['inputData']
    if not dataLoc.exists():
        return {'Status': 'Failure', 'Message': 'Data location does not exists.'}
    status = dict()
    usecase = config['targetPath']
    df = utils.read_data(dataLoc)
    target_feature = config['target_feature']
    dateTimeFeature= config['dateTimeFeature']
    scoreParam = config['scoring_criteria']
    testSize = (1 - config['train_ratio'])
    lstmConfig = config['algorithms']['LSTM']
    filename = meta_data['transformation']['Status']['Normalization_file']
    if (type(target_feature) is list):
        pass
    else:
        target_feature = list(target_feature.split(","))
    df.set_index(dateTimeFeature, inplace=True)
    log.info('Training LSTM for TimeSeries')
    mlp_model, look_back, scaler, error_matrix = startTraining(df,testSize,lstmConfig,filename,target_feature,scoreParam,log)
    score = error_matrix[scoreParam]
    log.info("LSTM Multivariant all scoring param results: "+str(error_matrix))
    # Training model
    


    model_path = targetPath/'runs'/str(meta_data['monitoring']['runId'])/model_name
    model_file_name = str(model_path/'model')
    mlp_model.save(model_file_name)
    meta_data['training'] = {}
    meta_data['training']['model_filename'] = model_file_name
    meta_data['training']['dateTimeFeature'] = dateTimeFeature
    meta_data['training']['target_feature'] = target_feature
    utils.write_json(meta_data, targetPath / IOFiles['metaData'])
    utils.write_json({'scoring_criteria': scoreParam, 'metrices': error_matrix,'score':error_matrix[scoreParam]}, model_path / IOFiles['metrics'])
    # return status
    status = {'Status': 'Success', 'errorMatrix': error_matrix,'score':error_matrix[scoreParam]}
    log.info(f'score: {error_matrix[scoreParam]}')
    log.info(f'output: {status}')
    return json.dumps(status)
    
if __name__ == '__main__':
    config = validateConfig()
    targetPath = Path('aion') / config['targetPath']
    if not targetPath.exists():
        raise ValueError(f'targetPath does not exist')
    meta_data_file = targetPath / IOFiles['metaData']
    if meta_data_file.exists():
        meta_data = utils.read_json(meta_data_file)
    else:
        raise ValueError(f'Configuration file not found: {meta_data_file}')
    log_file = targetPath / IOFiles['log']
    log = utils.logger(log_file, mode='a', logger_name=Path(__file__).parent.stem)
    try:
        print(train(config, targetPath, log))
    except Exception as e:
        
        status = {'Status': 'Failure', 'Message': str(e)}
        print(json.dumps(status))
    