from .llm import ModelConfig, LlamaModel
from .nlu import convert


class BanglaLLM:
    def __init__(self):
        self.config = ModelConfig()
        self.llm = LlamaModel(self.config)


    def invoke(self, text):
        converted = convert(text, target='en')
        print(f'Converted: {converted}')
        res = self.llm.generate(converted[0])
        print(f'Generated: {res}')
        response = convert(res, target='bn')
        print(f'Converted: {response}')
        return response[0]
