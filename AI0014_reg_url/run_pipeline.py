
from pathlib import Path
import subprocess
import sys
import json
import argparse

def run_pipeline(data_path):
	print('Data Location:', data_path)
	cwd = Path(__file__).parent

	monitor_file = str(cwd/'ModelMonitoring'/'code.py')

	load_file = str(cwd/'DataIngestion'/'code.py')
	transformer_file = str(cwd/'DataTransformation'/'code.py')
	selector_file = str(cwd/'FeatureEngineering'/'code.py')
	train_folder = cwd
	register_file = str(cwd/'ModelRegistry'/'code.py')
	deploy_file = str(cwd/'ModelServing'/'code.py')

	print('Running modelMonitoring')
	cmd = ['sys.executable', monitor_file, '-i', data_path]
	result = subprocess.check_output(cmd)
	result = result.decode('utf-8')
	print(result)    
	result = json.loads(result[result.find('{"Status":'):])
	if result['Status'] == 'Failure':
		exit()
	
	print('Running dataIngestion')
	cmd = ['sys.executable', load_file]
	result = subprocess.check_output(cmd)
	result = result.decode('utf-8')
	print(result)    
	result = json.loads(result[result.find('{"Status":'):])
	if result['Status'] == 'Failure':
		exit()

	print('Running DataTransformation')
	cmd = ['sys.executable', transformer_file]
	result = subprocess.check_output(cmd)
	result = result.decode('utf-8')
	print(result)
	result = json.loads(result[result.find('{"Status":'):])
	if result['Status'] == 'Failure':
		exit()

	print('Running FeatureEngineering')
	cmd = ['sys.executable', selector_file]
	result = subprocess.check_output(cmd)
	result = result.decode('utf-8')
	print(result)
	result = json.loads(result[result.find('{"Status":'):])
	if result['Status'] == 'Failure':
		exit()

	train_models = [f for f in train_folder.iterdir() if 'ModelTraining' in f.name]
	for model in train_models:
		print('Running',model.name)
		cmd = ['sys.executable', str(model/'code.py')]
		train_result = subprocess.check_output(cmd)
		train_result = train_result.decode('utf-8')
		print(train_result)    

	print('Running ModelRegistry')
	cmd = ['sys.executable', register_file]
	result = subprocess.check_output(cmd)
	result = result.decode('utf-8')
	print(result)
	result = json.loads(result[result.find('{"Status":'):])
	if result['Status'] == 'Failure':
		exit()

	print('Running ModelServing')
	cmd = ['sys.executable', deploy_file]
	result = subprocess.check_output(cmd)
	result = result.decode('utf-8')
	print(result)

if __name__ == '__main__':        
	parser = argparse.ArgumentParser()        
	parser.add_argument('-i', '--inputPath', help='path of the input data')        
	args = parser.parse_args()        
	if args.inputPath:        
		filename =  args.inputPath
	else:
		filename = r"https://aion-data.s3.ap-south-1.amazonaws.com/IPL2013.csv"
	try:        
		print(run_pipeline(filename))        
	except Exception as e:        
		print(e)
