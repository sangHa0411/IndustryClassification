import os
import wandb
import torch
import random
import pickle
import importlib
import pandas as pd
import numpy as np
import transformers

from dotenv import load_dotenv
from datasets import load_metric, Dataset
from sklearn.model_selection import StratifiedKFold
from utils.encoder import Encoder
from utils.preprocessor import Preprocessor

from arguments import (ModelArguments, 
    DataTrainingArguments, 
    MyTrainingArguments, 
    LoggingArguments
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    EarlyStoppingCallback, 
    Trainer,
)

from utils.scheduler import get_noam

ACC_METRIC = load_metric('accuracy')
F1_METRIC = load_metric('f1')
PATH = './input'

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

def compute_metrics(EvalPrediction):
    preds, labels = EvalPrediction
    preds = np.argmax(preds, axis=1)
    
    f1 = F1_METRIC.compute(predictions = preds, references = labels, average="macro")
    acc = ACC_METRIC.compute(predictions = preds, references = labels)
    acc.update(f1)
    return acc

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    dataset = pd.read_csv('./data/spaced_train.csv', index_col=False)
    dset = Dataset.from_pandas(dataset)
    print(dset)
    with open('data/labels/small_label_to_num.pickle', 'rb') as f:
        label_dict = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    preprocessor = Preprocessor(tokenizer, label_dict)
    dset = dset.map(preprocessor, 
        batched=True, 
        num_proc=4,
        remove_columns=dset.column_names
    )
    print(dset)

    encoder = Encoder(tokenizer, data_args.max_length)
    dset = dset.map(encoder, batched=True, num_proc=4, remove_columns=dset.column_names)
    print(dset)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = 225
    
    MODEL_NAME = model_args.model_name_or_path
    if training_args.model_type == 'base' :
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    else :
        model_type_str = 'model'
        model_lib = importlib.import_module(model_type_str)

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
            raise NotImplementedError

        model = model_class.from_pretrained(MODEL_NAME, config=config)

    if training_args.do_eval:
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        
        for i, (train_idx, valid_idx) in enumerate(skf.split(dset, dset['labels'])):
            if i >= 1 : break
            print(f"######### Fold : {i+1} !!! ######### ")
            train_dataset = dset.select(train_idx.tolist())
            valid_dataset = dset.select(valid_idx.tolist())
    else :
        train_dataset = dset
        valid_dataset = None
            
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=data_args.max_length)
    
    # wandb
    load_dotenv(dotenv_path=logging_args.dotenv_path)
    WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
    wandb.login(key=WANDB_AUTH_KEY)

    wandb.init(
        entity="metamong",
        project=logging_args.project_name,
        group='Single-label',
        name=training_args.run_name
    )
    wandb.config.update(training_args)

    epoch_steps = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    training_steps = epoch_steps * training_args.num_train_epochs
    warmup_steps = int(training_steps * training_args.warmup_ratio)
    optimizer = transformers.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        
    if training_args.use_noam :
        scheduler = get_noam(optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            d_model=config.hidden_size,
        )
    else :
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
    
    if training_args.use_rdrop :
        trainer_lib = importlib.import_module('trainer')
        trainer_class = getattr(trainer_lib, 'RdropTrainer')
    else :
        trainer_class = Trainer

    trainer = trainer_class(                    # the instantiated 🤗 Transformers model to be trained
        model=model,                            # model
        args=training_args,                     # training arguments, defined above
        train_dataset=train_dataset,            # training dataset
        eval_dataset=valid_dataset,             # evaluation dataset
        data_collator=data_collator,            # collator
        tokenizer=tokenizer,                    # tokenizer
        compute_metrics=compute_metrics,        # define metrics function
        optimizers=(optimizer, scheduler)       # optimizer, scheuler
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)] if training_args.do_eval else None
    )

    if training_args.do_train:
        train_result = trainer.train()
        print("######### Train result: ######### ", train_result)
        trainer.args.output_dir = data_args.save_path
        
        trainer.save_model()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    wandb.finish()
  
if __name__ == '__main__':
    main()