from itertools import cycle, islice
import torch
from torch.utils.data import IterableDataset


class MyIterableDataset(IterableDataset):

    def __init__(self,
                 tokenizer,
                 max_len,
                 file_path,
                 ):
        super(MyIterableDataset).__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len

    def parse_file(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                enc = self.tokenizer(
                    line.strip(),
                    max_length=self.max_len,
                    truncation=True,
                    return_token_type_ids= False,
                    return_tensors='pt'
                )

                yield {
                    'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                }

    def get_stream(self, file_path):
        return cycle(self.parse_file(file_path))

    def __len__(self):
        count = 0
        with open(self.file_path, 'r') as f:
            for line in f:
                count += 1

        return count

    def __iter__(self):
        return self.get_stream(self.file_path)

    def __getitem__(self, index):
        instance = self.get_stream(self.file_path)

        enc = self.tokenizer(
            instance,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids= False,
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }

