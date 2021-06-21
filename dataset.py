import torch
from torch.utils.data import Dataset


class PuncDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_length=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        return self.dataset[index]

    
    def collate_fn(self, samples):
        tokens = self.tokenizer([sample['tokens'] for sample in samples], max_length=self.max_length, padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)
        puncs = self._pad_labels([sample['punc'] for sample in samples], max_length=self.max_length)
        punc_types = self._pad_labels([sample['punc_type'] for sample in samples], max_length=self.max_length)
        return {**tokens, 'puncs': puncs, 'punc_types': punc_types}

    
    def _pad_labels(self, labels, padding=True, truncation=True, max_length=256):
        ''' labels: 2d list of labels(int), the second dimension may have different lengths'''
        if padding == True or padding == 'longest':
            length = max(len(row) for row in labels) + 2
        else:
            length = max_length
        if truncation:
            length = min(length, max_length)
        ret = torch.zeros((len(labels), length), dtype=torch.long)
        for i, row in enumerate(labels):
            ret[i, 1: 1 + len(row)] = torch.tensor(row[:length - 2])
        return ret


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
    raw_dataset = load_dataset('json', data_files={'train': 'train.json',}, field='data')
    punc_dataset = PuncDataset(raw_dataset['train'], tokenizer)
    print(punc_dataset.collate_fn([punc_dataset[0], punc_dataset[1]]))
    

    
