#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This file is automatically generated by AION for AI0017_1 usecase.
File generation time: 2023-02-21 10:13:56
'''
#Standard Library modules
import json
import warnings

#Third Party modules
from pathlib import Path

#local modules
import utility as utils 

warnings.filterwarnings("ignore")

IOFiles = {
    "log": "aion.log",
    "metaData": "modelMetaData.json",
    "metrics": "metrics.json",
    "production": "production.json"
}
        
def validateConfig():        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = utils.read_json(config_file)        
    return config        

def get_best_model(run_path):
    models_path = [d for d in run_path.iterdir() if d.is_dir]
    scores = {}
    for model in models_path:
        metrics = utils.read_json(model/IOFiles['metrics'])
        if metrics.get('score', None):
            scores[model.stem] = metrics['score']
    best_model = min(scores, key=scores.get)    
    return best_model

def __merge_logs(log_file_sequence,path, files):        
    if log_file_sequence['first'] in files:        
        with open(path/log_file_sequence['first'], 'r') as f:        
            main_log = f.read()        
        files.remove(log_file_sequence['first'])        
        for file in files:        
            with open(path/file, 'r') as f:        
                main_log = main_log + f.read()        
            (path/file).unlink()        
        with open(path/log_file_sequence['merged'], 'w') as f:        
            f.write(main_log)        
        
def merge_log_files(folder, models):        
    log_file_sequence = {        
        'first': 'aion.log',        
        'merged': 'aion.log'        
    }       
    log_file_suffix = '_aion.log'        
    log_files = [x+log_file_suffix for x in models if (folder/(x+log_file_suffix)).exists()]        
    log_files.append(log_file_sequence['first'])        
    __merge_logs(log_file_sequence, folder, log_files)  

def register(config, targetPath, log):        
    meta_data_file = targetPath / IOFiles['metaData']
    if meta_data_file.exists():
        meta_data = utils.read_json(meta_data_file)
    else:
        raise ValueError(f'Configuration file not found: {meta_data_file}')
    run_id = meta_data['monitoring']['runId']
    usecase = config['targetPath']
    current_run_path = targetPath/'runs'/str(run_id)
    register_model_name = get_best_model(current_run_path)
    models = config['models']        
    merge_log_files(targetPath, models) 
    meta_data['register'] = {'runId':run_id, 'model': register_model_name}
    utils.write_json(meta_data, targetPath/IOFiles['metaData'])
    utils.write_json({'Model':register_model_name,'runNo':str(run_id)}, targetPath/IOFiles['production'])
    status = {'Status':'Success','Message':f'Model Registered: {register_model_name}'}        
    log.info(f'output: {status}')        
    return json.dumps(status)
        
if __name__ == '__main__':
    config = validateConfig()
    targetPath = Path('aion') / config['targetPath']
    if not targetPath.exists():
        raise ValueError(f'targetPath does not exist')
    log_file = targetPath / IOFiles['log']
    log = utils.logger(log_file, mode='a', logger_name=Path(__file__).parent.stem)
    try:
        print(register(config, targetPath, log))
    except Exception as e:
        
        status = {'Status': 'Failure', 'Message': str(e)}
        print(json.dumps(status))
        