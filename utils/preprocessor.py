from typing import Optional

class Preprocessor :
    def __init__(self, tokenizer, label_dict:Optional[dict] = None, train_flag:bool = False) :
        self.tokenizer = tokenizer        
        self.label_dict = label_dict
        self.train_flag = train_flag

    def __call__(self, dataset) :
        inputs = []
        labels = []
        sep_token = self.tokenizer.sep_token

        for i in range(len(dataset['AI_id'])) :
            obj = '' if dataset['text_obj'][i] == 'nan' else dataset['text_obj'][i]
            mthd = '' if dataset['text_mthd'][i] == 'nan' else dataset['text_mthd'][i]
            deal = '' if dataset['text_deal'][i] == 'nan' else dataset['text_deal'][i]
            
            if self.train_flag == True :
                label_str = str(dataset['digit_3'][i])
                label_index = self.label_dict[label_str]
                labels.append(label_index)

            input_sen = obj + sep_token + mthd + sep_token + deal
            inputs.append(input_sen)

        dataset['inputs'] = inputs
        if self.train_flag == True :
            dataset['labels'] = labels
        return dataset