


class BaseModel:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        self.tokenizer = None
        self.temperature = 0

    #def load_model(self):
        #raise NotImplementedError("This method should be overridden in child classes.")

    def predict(self, query, source):
        raise NotImplementedError("This method should be overridden in child classes.")