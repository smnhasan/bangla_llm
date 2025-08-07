from .llm import ModelConfig, LlamaModel
from .nlu import convert


class BanglaLLM:
    def __init__(self):
        self.config = ModelConfig()
        self.llm = LlamaModel(self.config)


    def invoke(self, text):
        converted = convert(text, target='en')
        res = self.llm.generate(converted[0])
        response = convert(res, target='bn')
        return response[0]
