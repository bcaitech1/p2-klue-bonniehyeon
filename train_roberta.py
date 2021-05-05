import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig, Trainer, TrainingArguments, EarlyStoppingCallback
#from transformers import AdamW , get_linear_schedule_with_warmup
from load_data import *
import random
import wandb
import argparse
from dataclasses import dataclass


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
  preds = pred.predictions.argmax(-1))
  acc = accuracy_score(labels, preds)
  return {
        'accuracy': acc,
  }

# for dynamic padding
# https://gist.github.com/pommedeterresautee/1a334b665710bec9bb65965f662c94c8
def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]

@dataclass
class SmartCollator:
    pad_token_id: int
    
    def __call__(self, batch):
      batch_inputs = list()
      batch_attention_masks = list()
      batch_token_type_ids = list()
      labels = list()
      max_size = max([len(ex['input_ids']) for ex in batch])
      for item in batch:
          batch_inputs += [pad_seq(item['input_ids'], max_size, self.pad_token_id)]
          batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]
          #batch_token_type_ids += [pad_seq(item['token_type_ids'], max_size, 0)]
          labels.append(item['labels'])

      return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
              "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
              #"token_type_ids" : torch.tensor(batch_token_type_ids, dtype=torch.long),
              "labels": torch.tensor(labels, dtype=torch.long)
              }

def train(args):
  seed_everything(args.random_seed)
  wandb.login()
  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("/opt/ml/input/data/train/train_new.tsv")
  train_label = train_dataset['label'].values[:8000]
  eval_label = train_dataset['label'].values[8000:]
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset[:8000], tokenizer)
  tokenized_eval = tokenized_dataset(train_dataset[8000:], tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_eval_dataset = RE_Dataset(tokenized_eval, eval_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  bert_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 42
  
  model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=bert_config) 
  model.to(device)
  
  # initializing parameter of classification layer -for tanh activation function(default)
  #torch.nn.init.xavier_uniform_(model.classifier.dense.weight)
  #torch.nn.init.xavier_uniform_(model.classifier.out_proj.weight)

  # initializing parameter of classification layer -for relu activation function
  #torch.nn.init.kaiming_normal_(model.classifier.dense.weight)
  #torch.nn.init.kaiming_normal_(model.classifier.out_proj.weight)
  
  # change dropout probability of classification layer
  #model.classifier.dropout.p = 0.3
  
  # set learning rate 
  #embedding_para =[{'params': v, 'lr':1E-5} for k, v in model.roberta.embeddings.named_parameters()] 
  #classifier_para=[{'params': v, 'lr':1E-5}  for k, v in model.classifier.named_parameters()]
  
  # set encoder layer's learning rate (depending on layer number)
  #encoder_layer_para=[{'params': v, 'lr':1E-5*0.95**(23-int(k.split('.')[0]))} for k, v in model.roberta.encoder.layer.named_parameters()]
  #all_para = embedding_para + encoder_layer_para + classifier_para
  
  # when you set learning rate respectively, need to set optimizer and scheduler here(not TrainingArguments())
  # and need to add option at Trainer()
  #optimizer = AdamW(all_para,weight_decay= 0.01, eps =1e-8)
  #scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=300, num_training_steps=25000)
  
  # freezing layer
  #for param in model.roberta.embeddings.parameters():
  #    param.requires_grad = False
  #for param in model.roberta.encoder.parameters():
  #    param.requires_grad = False


  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    #group_by_length = True, # Whether or not to group together samples of roughly the same length in the training dataset. Only useful if applying dynamic padding.
    save_total_limit=5,              # number of total save model.
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
    evaluation_strategy='steps',  # `no`, `steps`, `epoch`
    #fp16 = True,                 #  Whether to use 16-bit (mixed) precision training instead of 32-bit training.
    dataloader_num_workers = 4,
    load_best_model_at_end = True,
    metric_for_best_model = 'accuracy',
    run_name = args.name,
    label_smoothing_factor = 0.5,
    seed = args.random_seed,
    )

  early_stopping = EarlyStoppingCallback(early_stopping_patience = 5, early_stopping_threshold = 0.001)

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_eval_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,         # define metrics function
    #callbacks=[early_stopping],
    #optimizers=[optimizer, scheduler]    
    #data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id)   # dynamic padding     
  )

  # train model
  trainer.train()
  wandb.finish() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed' , type = int , default = 2021 , help = 'random seed (default = 2021)')
    parser.add_argument('--name' , type = str , default = 'roberta' , help = 'running name')
    
    args = parser.parse_args()
    print(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train(args)