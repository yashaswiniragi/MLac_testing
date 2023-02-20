FROM python:3.8-slim-buster
LABEL "usecase"="AI0014"
LABEL "usecase_test"="AI0014_test"
COPY  requirements.txt requirements.txt
COPY  config.json config.json
COPY  code.py code.py
COPY  utility.py utility.py
RUN pip install --no-cache-dir -r requirements.txt