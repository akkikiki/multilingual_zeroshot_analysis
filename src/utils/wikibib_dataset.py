import torch, jsonlines, os, sys, pickle, random
from tqdm import tqdm

class WikiBibDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer,
                 source_directory,
                 target_directory,
                 train_sampler,
                 max_len,
                 source_lang=None,
                 target_lang=None,
                 num_langs=0,
                 file_path=None,
                 langs_to_use=None,
                 id2lang=None,
                 remove_underscore=False,
                 seed=42):
        self.seed = seed
        self.remove_underscore = remove_underscore
        self.langs_to_use = langs_to_use
        self.id2lang = id2lang

        self.num_langs = num_langs

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.source_directory = source_directory
        self.target_directory = target_directory

        self.source_lang = source_lang
        self.target_lang = target_lang

        self.train_sampler = train_sampler

        if file_path:
            self.source_examples = self.read_file(file_path)
        else:
            self.source_examples = self.read_files(source_directory, source=True)
        self.target_examples = self.read_files(target_directory)

        print('Dataset created, with settings: source: {}\n target: {} \n max len: {} \n source_lang: {}\n target_lang: {}'.format(self.source_directory, self.target_directory, self.max_len, self.source_lang, self.target_lang))

        self.finalized_examples = None

        if self.train_sampler == 'baseline':
            self.finalized_examples = self.create_baseline_dataset()
            print('Baseline sampling: created {} examples ready for training.'.format(len(self.finalized_examples)))

        elif self.train_sampler == 'upsample':
            self.finalized_examples = self.create_upsampled_dataset()
            print('Upsampled target dataset.')

        print('First example: {}'.format(self.finalized_examples[0]))

    def __getitem__(self, idx):
        instance = self.finalized_examples[idx]

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

    def __len__(self):
        return len(self.finalized_examples)

    def read_files(self, directory, source=False):

        if directory is None:
            return []

        inps = []
        files = [directory + x for x in os.listdir(directory)]
        files.sort()

        with tqdm(total=len(files), file=sys.stdout, ascii=True) as pbar:
            for i, file in enumerate(files):
                if i >= self.num_langs:
                    print(f"Skipping lang {i} due to num langs set to {self.num_langs}")
                    continue
                if self.id2lang and (not self.id2lang[i] in self.langs_to_use):
                    print(f"Skipping lang {i} ({self.id2lang[i]}) due to not in {self.langs_to_use}")
                    continue
 
                with open(file, 'r') as f:
                    for line in f:
                        if self.remove_underscore:
                            temp = ""
                            for token in self.tokenizer.tokenize(line.strip()):
                                if token != "▁":
                                    temp += token.replace("▁", " ")
                            inps.append(temp)
                        else:
                            inps.append(line.strip())
                print('Dataset: Finished loading file: {}'.format(file))
                pbar.update(1)

        if source:
            print('Read in {} source examples across {} files'.format(len(inps), len(files)))
        else:
            print('Read in {} target examples across {} files'.format(len(inps), len(files)))

        return inps

    def create_baseline_dataset(self):
        x = self.source_examples + self.target_examples
        random.seed(self.seed)
        random.shuffle(x)
        return x

    def read_file(self, file_path, source=False):

        inps = []
        with open(file_path, 'r') as f:
            for line in f:
                if self.remove_underscore:
                    temp = ""
                    for token in self.tokenizer.tokenize(line.strip()):
                        if token != "▁":
                            temp += token.replace("▁", " ")
                    inps.append(temp)

                else:
                    inps.append(line.strip())
 
        print('Dataset: Finished loading file: {}'.format(file_path))
        print('Read in {} examples across'.format(len(inps)))

        return inps

    def create_upsampled_dataset(self):

        num_source_examples = len(self.source_examples)

        random.seed(42)
        target_samples = random.choices(self.target_examples, k = num_source_examples)

        print('Starting with {} source examples and {} target examples'.format(num_source_examples, len(self.target_examples)))
        print('Sampled {} examples from target'.format(len(target_samples)))

        x = self.source_examples + target_samples
        random.shuffle(x)

        print('Created {} examples for training'.format(len(x)))

        return x
