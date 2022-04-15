
class Preprocessor :
    def __init__(self, tokenizer, label_dict=None, mode_test=False) :
        self.tokenizer = tokenizer        
        self.label_dict = label_dict if not mode_test else None
        self.mode = 'test' if mode_test else 'train'

    def __call__(self, dataset) :
        inputs = []
        labels = []
        large_labels = []
        medium_labels = []

        sep_token = ' ' + self.tokenizer.sep_token + ' '
        size = len(dataset['AI_id'])
        for i in range(size) :
            obj = '' if dataset['text_obj'][i] == None else dataset['text_obj'][i]
            mthd = '' if dataset['text_mthd'][i] == None else dataset['text_mthd'][i]
            deal = '' if dataset['text_deal'][i] == None else dataset['text_deal'][i]
            if self.mode == 'train':
                label = str(dataset['digit_3'][i])
                
                if isinstance(self.label_dict, list) :       
                    large_label = dataset['digit_1'][i]
                    medium_label = str(dataset['digit_2'][i])
                    large_labels.append(self.label_dict[0][large_label])
                    medium_labels.append(self.label_dict[1][medium_label])
                    labels.append(self.label_dict[2][label])
                else :
                    labels.append(self.label_dict[str(label)])
            input_sen = obj + sep_token + mthd + sep_token + deal
            inputs.append(input_sen)

        dataset['inputs'] = inputs
        if self.mode == 'train':
            if isinstance(self.label_dict, list) : 
                dataset['large_labels'] = large_labels
                dataset['medium_labels'] = medium_labels
            dataset['labels'] = labels
        return dataset