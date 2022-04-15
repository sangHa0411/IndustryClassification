from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class ModelArguments : 
    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    
@dataclass
class DataTrainingArguments:
    data_path: str = field(
        default="./input",
        metadata={
            "help": "Data path"
        },
    )
    save_path: str = field(
        default="./checkpoints/roberta-large",
        metadata={
            "help": "Path to save checkpoint from fine tune model"
        },
    )
    max_length: int = field(
        default=128,
        metadata={
            "help": "Max length of input sequence"
        },
    )
    output_name: str = field(
        default="Base.csv",
        metadata={
            "help": "Path to save output.csv from fine tune model"
        },
    )
    use_spaced : bool = field(
        default=False,
        metadata={
            "help" : "using spaced file"
        }
    )
    
@dataclass
class MyTrainingArguments(TrainingArguments):
    report_to: Optional[str] = field(
        default='wandb',
    )
    use_lstm: bool = field(
        default=False,
        metadata={
            "help" : "using lstm model"
        }
    )
    use_noam: bool = field(
        default=False,
        metadata={
            "help" : "using noam scheduler"
        }
    )
    use_cosine: bool = field(
        default=False,
        metadata={
            "help" : "using cosine scheduler"
        }
    )
    use_rdrop: bool = field(
        default=False,
        metadata={
            "help" : "rdop trinaer"
        }
    )
    model_type: str = field(
        default='base',
        metadata={
            'help' : 'model type'
        }
    )

@dataclass
class LoggingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dotenv_path: Optional[str] = field(
        default='./wandb.env',
        metadata={"help":'input your dotenv path'},
    )
    project_name: Optional[str] = field(
        default="Industry Classification",
        metadata={"help": "project name"},
    )

@dataclass
class InferenceArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    k_fold: bool = field(
        default=False,
        metadata={"help":'k fold inference'},
    )
    k : Optional[int] = field(
        default=5,
        metadata={"help" : "The number of folds"}
    )