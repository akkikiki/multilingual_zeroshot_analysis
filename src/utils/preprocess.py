import argparse, pickle, os, sys
from tqdm import tqdm
from transformers import XLMRobertaTokenizer


class WikiPreprocessor:
    def __init__(self,
                source_directory,
                output_directory,
                tokenizer_path,
                max_seq_len,
                min_load_len,
                train_sampler,
                rank,
                target_directory = None,
                is_chinese = False):

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path, max_len=max_seq_len)

        self.target_directory = target_directory
        self.source_directory = source_directory
        self.output_directory = output_directory

        self.max_seq_len = max_seq_len
        self.min_load_len = min_load_len
        self.is_chinese = is_chinese

        self.train_sampler = train_sampler

        self.rank = rank
        #self.fast_tokenizer = 
        self.fast_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    def load_from_text(self):

        self._read_directory(self.source_directory, source=True)

        if self.target_directory is not None:
            self._read_directory(self.target_directory)

    def _read_directory(self, directory, source=False):

        files = [directory + x for x in os.listdir(directory)]
        print(files[:5])
        num_examples = 0
        with tqdm(total=len(files), file=sys.stdout) as pbar:
            for i, file in enumerate(files):
                temp = []
                with open(file, 'r') as f:
                    for line in f.readlines():
                        if self.is_chinese and len(line.strip()) > self.min_load_len:
                            temp.append((line.strip(), len(self.fast_tokenizer.encode(line.strip()))))
                            num_examples += 1
                        elif len(line.strip().split(' ')) > self.min_load_len:
                            #temp.append((line.strip(), len(self.fast_tokenizer.encode(line.strip()).ids)))
                            temp.append((line.strip(), len(self.fast_tokenizer.encode(line.strip()))))
                            num_examples += 1
                print('Dataset: Finished loading file: {}'.format(file))

                if source:
                    print(self.output_directory)
                    self._pack_examples(temp, self.output_directory+'source_packed' + str(self.rank) + '.txt')
                else:
                    self._pack_examples(temp, self.output_directory+'target_packed' + str(self.rank) + '.txt')

                pbar.update(1)

        if source:
            print('Read in {} source examples across {} files'.format(num_examples, len(files)))
        else:
            print('Read in {} target examples across {} files'.format(num_examples, len(files)))

    def _pack_examples(self, batch, outfile_name):

        running_length = 2
        temp_input_sents = ''

        print('Packing examples...')

        packed_counter = 0

        with tqdm(total=len(batch), file=sys.stdout) as pbar:
            with open(outfile_name, 'a') as outfile:
                for i in range(0,len(batch)):
                    toks = batch[i][0]
                    example_len = batch[i][1]

                    if example_len + running_length >= self.max_seq_len:
                        outfile.write((temp_input_sents) + '\n')
                        packed_counter += 1
                        running_length = 2 + example_len
                        temp_input_sents = toks
                        continue
                    if example_len + running_length < self.max_seq_len:
                        temp_input_sents += (' ' + toks)
                        running_length += example_len


                pbar.update(1)

                if running_length > 2 and running_length < self.max_seq_len - 1:
                    outfile.write(temp_input_sents + '\n')

        print('Packed {} examples into {} examples'.format(len(batch), packed_counter))





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--source_directory', type=str)
    parser.add_argument('--target_directory', type=str)
    parser.add_argument('--output_directory', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--min_load_len', type=int, default=10)
    parser.add_argument('--chinese', action='store_true', default=False)
    parser.add_argument('--rank', type=str, default='0')

    args = parser.parse_args()

    preproc = WikiPreprocessor(
        args.source_directory,
        args.output_directory,
        args.tokenizer_path,
        args.max_seq_len,
        args.min_load_len,
        args.rank,
        args.target_directory,
        is_chinese=args.chinese,
    )

    preproc.load_from_text()

