#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This file is automatically generated by AION for AI0017_1 usecase.
File generation time: 2023-02-21 10:13:56
'''
#Standard Library modules
import json
import logging

#Third Party modules
from pathlib import Path
import pandas as pd 

#local modules
from utility import *

IOFiles = {
    "inputData": "transformedData.dat",
    "metaData": "modelMetaData.json",
    "log": "aion.log",
    "outputData": "featureEngineeredData.dat"
}
        
def validateConfig():        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = read_json(config_file)        
    return config

def featureSelector(config, targetPath, log):
    dataLoc = targetPath / IOFiles['inputData']
    if not dataLoc.exists():
        return {'Status': 'Failure', 'Message': 'Data location does not exists.'}

    status = dict()
    df = pd.read_csv(dataLoc)
    log.log_dataframe(df)

    csv_path = str(targetPath / IOFiles['outputData'])
    write_data(df, csv_path, index=False)
    status = {'Status': 'Success', 'dataFilePath': IOFiles['outputData']}
    log.info(f'Selected data saved at {csv_path}')
    meta_data['featureengineering'] = {}
    meta_data['featureengineering']['Status'] = status
    write_json(meta_data, str(targetPath / IOFiles['metaData']))
    log.info(f'output: {status}')
    return json.dumps(status)
    
if __name__ == '__main__':
    config = validateConfig()
    targetPath = Path('aion') / config['targetPath']
    if not targetPath.exists():
        raise ValueError(f'targetPath does not exist')
    meta_data_file = targetPath / IOFiles['metaData']
    if meta_data_file.exists():
        meta_data = read_json(meta_data_file)
    else:
        raise ValueError(f'Configuration file not found: {meta_data_file}')
    log_file = targetPath / IOFiles['log']
    log = logger(log_file, mode='a', logger_name=Path(__file__).parent.stem)
    try:
        print(featureSelector(config,targetPath, log))
    except Exception as e:
        
        status = {'Status': 'Failure', 'Message': str(e)}
        print(json.dumps(status))
    