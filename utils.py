import pandas as pd
from datasets import Dataset

def make_dataset(dataset, do_train=False):
    label2id = {'entailment':0, 'neutral':1, 'contradiction':2}
    
    premise_list = [data['premise'] for data in dataset]
    hypothesis_list = [data['hypothesis'] for data in dataset]
    if do_train:
        labels = [label2id[data['label']] for data in dataset]

    if do_train:
        temp = {'premise': premise_list,
                'hypothesis': hypothesis_list,
                'label': labels}
    else:
        temp = {'premise': premise_list,
                'hypothesis': hypothesis_list}
        
    df = pd.DataFrame(temp)
    dataset = Dataset.from_pandas(df)
    return dataset