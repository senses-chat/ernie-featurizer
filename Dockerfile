FROM python:3.9.9

WORKDIR /opt/server

RUN pip install pipenv

COPY Pipfile Pipfile.lock Makefile /opt/server/

RUN pipenv install --system --deploy

COPY . /opt/server/

RUN python download_models.py

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
