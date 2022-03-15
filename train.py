import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from datasets import Dataset
from utils import make_dataset
import random
import os
import numpy as np
import torch
import wandb

wandb.login()

train_path = "input/klue-nli-v1.1_train.json"

f = open(train_path, encoding="UTF-8")
train_data = json.loads(f.read())

train_data, valid_data = train_data[:20000], train_data[20000:]

print('train data samples: ', len(train_data))
print('valid data samples: ', len(valid_data))
# print(len(test_data))


tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

label2id = {'entailment':0, 'neutral':1, 'contradiction':2}

train_dataset = make_dataset(train_data, do_train=True)
valid_dataset = make_dataset(valid_data, do_train=True)

def tokenized_dataset(example):
    encoding = tokenizer(
        example['premise'],
        example['hypothesis'],
        # return_tensors= "pt", # pytorch type      # map할때는 이 옵션을 쓰지 말것 !! dict로 리턴해야하니까 
        padding= True, # 문장의 길이가 짧다면 padding
        truncation= True, # 문장 자르기
        max_length= 256, # 토큰 최대 길이...
        return_token_type_ids= True # 두 문장을 한문장으로 인식하게 하기 위해 token_type_ids=False 시킴
    )    
    
    encoding['labels'] = example['label']
    
    return encoding
    
train_tokenized_data = train_dataset.map(tokenized_dataset, batched=True, batch_size=1000)
valid_tokenized_data = valid_dataset.map(tokenized_dataset, batched=True, batch_size=1000)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    config = AutoConfig.from_pretrained('klue/bert-base')
    config.num_labels = 3 # ['entailment', 'neutral', 'contradiction']

    model = AutoModelForSequenceClassification.from_pretrained('klue/bert-base', config=config)

    training_args = TrainingArguments(
        output_dir = './output',
        evaluation_strategy = 'epoch',
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        gradient_accumulation_steps = 1,
        learning_rate = 1e-5,
        weight_decay = 0.01,
        max_grad_norm = 10,
        num_train_epochs = 6,
        warmup_ratio = 0.1,
        logging_strategy = 'steps',
        logging_steps = 50,
        save_strategy = 'epoch',
        save_total_limit = 1,
        seed = 42,
        dataloader_num_workers = 2,
        load_best_model_at_end = True,
        metric_for_best_model = 'accuracy',
        report_to='wandb'
    )

    def compute_metrics(out):    
        pred, labels = out
        pred = np.argmax(pred, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average='micro')
        precision = precision_score(y_true=labels, y_pred=pred, average='micro')
        f1 = f1_score(y_true=labels, y_pred=pred, average='micro')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_tokenized_data,
        eval_dataset=valid_tokenized_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
   
    run = wandb.init(project='KLUE_NLI', entity='jspirit01', name='klue/bert_v2')
        
    trainer.train()
    trainer.save_model('./best_model/test')
    run.finish()


if __name__ == "__main__":
    seed_everything(42)
    train()