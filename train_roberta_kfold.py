import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig, Trainer, TrainingArguments, EarlyStoppingCallback
#from transformers import AdamW , get_linear_schedule_with_warmup
from load_data import *
import random
import wandb
import argparse


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  return {
        'accuracy': acc,
  }

def train(args):  
  seed_everything(args.random_seed)
  wandb.login()
  
  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # setting model hyperparameter
  bert_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 42

  # load data
  dataset = load_data("/opt/ml/input/data/train/train_new.tsv")
  label  = dataset['label'].values

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   

  # StratifiedKFold setting
  cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=args.random_seed)
  
  for idx , (train_idx , val_idx) in enumerate(cv.split(dataset, label)):
    train_dataset = dataset.iloc[train_idx]
    val_dataset = dataset.iloc[val_idx]

    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_val = tokenized_dataset(val_dataset, tokenizer)

    train_y = label[train_idx]
    val_y = label[val_idx]
 
    RE_train_dataset = RE_Dataset(tokenized_train, train_y)
    RE_eval_dataset = RE_Dataset(tokenized_val, val_y)

    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config) 
    model.to(device)

    output_dir = './results_kfold' + str(idx)
    training_args = TrainingArguments(
      output_dir= output_dir,          # output directory
      group_by_length = True, # Whether or not to group together samples of roughly the same length in the training dataset. Only useful if applying dynamic padding.
      save_total_limit=1,              # number of total save model. 
      num_train_epochs=10,              # total number of training epochs
      learning_rate=1e-5,               # learning_rate
      save_steps=100,
      per_device_train_batch_size=32,  # batch size per device during training
      per_device_eval_batch_size=32,   # batch size for evaluation
      warmup_steps=300,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=100,              # log saving step.
      eval_steps=100,
      evaluation_strategy='steps', # `no`, `steps`, `epoch`
      #fp16 = True,                #  Whether to use 16-bit (mixed) precision training instead of 32-bit training.
      dataloader_num_workers = 4,
      load_best_model_at_end = True,
      metric_for_best_model = 'accuracy',
      run_name = 'kfold',
      label_smoothing_factor = 0.5,
      seed = args.random_seed
      )

    early_stopping = EarlyStoppingCallback(early_stopping_patience = 5, early_stopping_threshold = 0.001)

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=RE_train_dataset,         # training dataset
      eval_dataset=RE_eval_dataset,             # evaluation dataset
      compute_metrics=compute_metrics,         # define metrics function
      #optimizers=[optimizer, scheduler]
      #callbacks=[early_stopping] 
    )
    # train model
    trainer.train()

    # to be free from OOM CURSE, every fold needs to be ends with this.
    model.cpu()
    del model
    torch.cuda.empty_cache()

  wandb.finish()
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed' , type = int , default = 0 , help = 'random seed (default = 2021)')
    parser.add_argument('--name' , type = str , default = 'roberta' , help = 'running name')

    args = parser.parse_args()
    print(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train(args)