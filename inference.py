import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from datasets import Dataset
import random
import os
import numpy as np
import torch
from utils import make_dataset

valid_path = "input/klue-nli-v1.1_dev.json"

f = open(valid_path, encoding="UTF-8")
test_data = json.loads(f.read())


tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

id2label = ['entailment', 'neutral', 'contradiction']
label2id = {'entailment':0, 'neutral':1, 'contradiction':2}

test_dataset = make_dataset(test_data, do_train=False)

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
        
        return encoding
    
test_tokenized_data = test_dataset.map(tokenized_dataset, batched=True, batch_size=1000)

training_args = TrainingArguments(per_device_eval_batch_size=512, output_dir = './pred_output')

model_path = './best_model/test'
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=None,
    eval_dataset=test_tokenized_data,
    tokenizer=tokenizer,
)

predictions, _, _ = trainer.predict(test_dataset = test_tokenized_data)
print("shape of prediciton", predictions.shape)

pred_ids = predictions.argmax(axis=1)
pred_labels = [id2label[id] for id in pred_ids]

f = open('./pred_output/output.csv', 'w')
for i in range(len(test_dataset)):
    line = str(i) + '/'+ test_dataset['premise'][i] + '/' + test_dataset['hypothesis'][i] + '/' + pred_labels[i] + '\n'
    f.write(line)
f.close()
