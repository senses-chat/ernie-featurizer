import os
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers import AutoModel

model_name = os.environ.get('MODEL_NAME', 'ernie-3.0-base-zh')

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"model {model_name} downloaded")
