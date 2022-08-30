-include .env.local
export

download-models:
	pipenv run python download_models.py

executor:
	pipenv run jina executor --uses config.yml --port $$EXECUTOR_PORT

flow:
	pipenv run jina flow --uses featurizer.yml

publish:
	pipenv run jina hub push --public .
