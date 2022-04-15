import os
import torch
import pickle
import random
import importlib
import collections
import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import Dataset

from utils.encoder import Encoder
from utils.preprocessor import Preprocessor

from arguments import ModelArguments, DataTrainingArguments, MyTrainingArguments, InferenceArguments
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    Trainer
)

with open('./data/map/large_to_medium.pickle', 'rb') as f:
    LARGE_TO_MEDIUM = pickle.load(f)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, InferenceArguments)
    )
    model_args, data_args, training_args, inference_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)
    
    with open('data/labels/small_num_to_label.pickle', 'rb') as f:
        num_to_label = pickle.load(f)
    
    # loading dataset
    if data_args.use_spaced:
        test_df = pd.read_csv('./data/spaced_test.csv', index_col=False)
    else :
        test_df = pd.read_csv('./data/test.csv', index_col=False)
    
    dset = Dataset.from_pandas(test_df)
    print(dset)

    # preprocessing dataset
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    preprocessor = Preprocessor(tokenizer, mode_test=True)
    dset = dset.map(preprocessor, 
        batched=True, 
        num_proc=4,
        remove_columns=dset.column_names
    )
    print(dset)

    # encoding dataset
    encoder = Encoder(tokenizer, data_args.max_length, mode_test=True)
    test_dataset = dset.map(encoder, batched=True, remove_columns=dset.column_names)
    print(test_dataset)
    
    # data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
    
    if inference_args.k_fold == False :
        print('Inference test data')

        # model class
        model_lib = importlib.import_module('model')
        if training_args.model_type == 'average' :
            model_class = getattr(model_lib, 'RobertaWeighAverage')
        elif training_args.model_type == 'lstm' :
            model_class = getattr(model_lib, 'RobertaLSTM')
        elif training_args.model_type == 'cnn' :
            model_class = getattr(model_lib, 'RobertaCNN')
        elif training_args.model_type == 'rbert' :
            model_class = getattr(model_lib, 'RobertaRBERT')
        elif training_args.model_type == 'arcface' :
            model_class = getattr(model_lib, 'RobertaArcface')
        else :
            model_class = AutoModelForSequenceClassification

        # model element
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = model_class.from_pretrained(model_args.model_name_or_path, config=config)

        trainer = Trainer(                       # the instantiated 🤗 Transformers model to be trained
            model=model,                         # trained model
            args=training_args,                  # training arguments, defined above
            data_collator=data_collator,         # collator
            tokenizer=tokenizer,
        )
            
        # predicting model
        outputs = trainer.predict(test_dataset)

        # save predicted file
        submission = pd.read_csv('./data/test.csv', index_col=False)
        submission.digit_3 = outputs[0].argmax(axis=1)
        submission.digit_3 = submission.digit_3.map(num_to_label).astype(str)
        submission.digit_2 = submission.digit_3.map(lambda x : x[:-1])
        submission.digit_1 = submission.digit_2.map(mapping_function)
        
        submission.to_csv(os.path.join('./output', data_args.output_name), index=False)
    else :
        K_FOLD = inference_args.k
        print('Inference test data (K_fold)')

        predictions = []
        ids = []
        for i in tqdm(range(K_FOLD)) :
            dir_name = f'fold{i}'
            model_name_or_path = os.path.join(model_args.model_name_or_path, dir_name)
        
            # model class
            model_lib = importlib.import_module('model')
            if training_args.model_type == 'average' :
                model_class = getattr(model_lib, 'RobertaWeighAverage')
            elif training_args.model_type == 'lstm' :
                model_class = getattr(model_lib, 'RobertaLSTM')
            elif training_args.model_type == 'cnn' :
                model_class = getattr(model_lib, 'RobertaCNN')
            elif training_args.model_type == 'rbert' :
                model_class = getattr(model_lib, 'RobertaRBERT')
            else :
                model_class = AutoModelForSequenceClassification

            config = AutoConfig.from_pretrained(model_name_or_path)
            model = model_class.from_pretrained(model_name_or_path, config=config)

            trainer = Trainer(                       # the instantiated 🤗 Transformers model to be trained
                model=model,                         # trained model
                args=training_args,                  # training arguments, defined above
                data_collator=data_collator,         # collator
                tokenizer=tokenizer,
            )
            
            # predicting model
            outputs = trainer.predict(test_dataset)
            predictions.append(outputs[0])
            ids.append(outputs[0].argmax(axis=1))

        # soft voting
        print('Soft Voting')
        soft_prediction = np.sum(predictions, axis=0)
        soft_submission = pd.read_csv('./data/test.csv', index_col=False)
        soft_submission.digit_3 = soft_prediction.argmax(axis=1)
        soft_submission.digit_3 = soft_submission.digit_3.map(num_to_label).astype(str)
        soft_submission.digit_2 = soft_submission.digit_3.map(lambda x : x[:-1])
        soft_submission.digit_1 = soft_submission.digit_2.map(mapping_function)
        
        soft_submission.to_csv(os.path.join('./output', data_args.output_name, 'softvoting.csv'), index=False)

        # hard voting
        print('Hard Voting')
        voted_labels = []
        counter = collections.Counter()

        hard_submission = pd.read_csv('./data/test.csv', index_col=False)
        for i in tqdm(range(len(hard_submission))) :
            labels = [id_list[i] for id_list in ids]
            counter.update(labels)
            counter_dict = dict(counter)

            items = sorted(counter_dict.items(), key=lambda x : x[1], reverse=True)
            voted_labels.append(items[0][0])
            counter.clear()

        hard_submission.digit_3 = voted_labels
        hard_submission.digit_3 = hard_submission.digit_3.map(num_to_label).astype(str)
        hard_submission.digit_2 = hard_submission.digit_3.map(lambda x : x[:-1])
        hard_submission.digit_1 = hard_submission.digit_2.map(mapping_function)
        
        hard_submission.to_csv(os.path.join('./output', data_args.output_name, 'hardvoting.csv'), index=False)

def mapping_function(example):
    for k, v in LARGE_TO_MEDIUM.items():
        if example in v:
            return k
    
if __name__ == "__main__" :
    main()