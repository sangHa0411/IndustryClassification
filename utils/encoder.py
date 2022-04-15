
class Encoder :
    def __init__(self, tokenizer, max_input_length, mode_test=False) :
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.mode = 'test' if mode_test else 'train'
    
    def __call__(self, examples):
        inputs = examples['inputs']
        model_inputs = self.tokenizer(inputs, 
            max_length=self.max_input_length, 
            return_token_type_ids=False,
            truncation=True
        )
        if self.mode == 'train':
            if len(examples) > 2 :
                model_inputs['large_labels'] = examples['large_labels']
                model_inputs['medium_labels'] = examples['medium_labels']
            model_inputs['labels'] = examples['labels']
        return model_inputs