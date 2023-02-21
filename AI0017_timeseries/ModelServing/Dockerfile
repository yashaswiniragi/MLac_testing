FROM python:3.8-slim-buster
LABEL "usecase"="AI0017"
LABEL "usecase_test"="AI0017_test"

COPY  utility.py utility.py
COPY  __init__.py __init__.py
COPY  predict.py predict.py
COPY  aionCode.py aionCode.py
COPY  requirements.txt requirements.txt
COPY  config.json config.json
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "aionCode.py","-ip","0.0.0.0","-pn","8094"]
