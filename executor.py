import os
import paddle
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers import AutoModel
from jina import Executor, requests
from docarray import DocumentArray, Document

# use env vars to load model name
model_name = os.environ.get('MODEL_NAME', 'ernie-3.0-base-zh')

class PaddleNLPFeaturizer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_faster=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @requests
    def featurize(self, docs: DocumentArray, **kwargs) -> DocumentArray:
        print(len(docs))

        encoded_inputs = self.tokenizer(
            text=docs.texts,
            max_seq_len=512,
            pad_to_max_seq_len=True,
        )

        input_ids = encoded_inputs['input_ids']
        token_type_ids = encoded_inputs['token_type_ids']

        sequence_output, pooled_output = self.model(
            input_ids=paddle.to_tensor(input_ids),
            token_type_ids=paddle.to_tensor(token_type_ids),
        )

        for i, doc in enumerate(docs):
            tokens = Document(modality='tokens', embedding=input_ids[i])
            sequence_features = Document(modality='sequence_features', embedding=sequence_output[i].numpy())
            sentence_feature = Document(modality='sentence_feature', embedding=pooled_output[i].numpy())
            doc.chunks.append(tokens)
            doc.chunks.append(sequence_features)
            doc.chunks.append(sentence_feature)

        return docs
