import argparse
import os
import random
import numpy as np
import sys
from tqdm import tqdm


def read_files(directory):
    if directory is None:
        return []

    inps = []
    files = [directory + x for x in os.listdir(directory)]
    files.sort()

    line_num = 0
    with tqdm(total=len(files), file=sys.stdout, ascii=True) as pbar:
        for i, file in enumerate(files):
            with open(file, 'r') as f:
                for line in f:
                    #inps.append(line.strip())
                    line_num += 1
            print('Dataset: Finished loading file: {}'.format(file))
            pbar.update(1)

    print('Read in {} examples across {} files'.format(len(inps), len(files)))

    return line_num


def write_files(directory, line_nums, output_dir, lang_num):
    if directory is None:
        return []

    inps = []
    files = [directory + x for x in os.listdir(directory)]
    files.sort()
    line_num = 0
    with tqdm(total=len(files), file=sys.stdout, ascii=True) as pbar:
        for i, file in enumerate(files):
            print(output_dir)
            print(os.path.basename(file))
            f_out = open(output_dir + os.path.basename(file) + f"_{lang_num}.txt", "w")

            with open(file, 'r') as f:
                for line in f:
                    if line_num in line_nums:
                        f_out.write(line)
                    #inps.append(line.strip())
                    line_num += 1
            f_out.close()
            print('Dataset: Finished loading file: {}'.format(file))
            pbar.update(1)

    print('Finish writing done')

    return inps


def main(args):
    print("downsampling to the minimum corpus size...")
    random.seed(42)
    np.random.seed(42)
    examples = []
    for dir in args.dirs:
        examples.append(read_files(dir))

    if args.num_examples == -1:
        num_examples = min(examples) * args.scale

    else:
        num_examples = args.num_examples
    print(f"(scaled) num_examples={num_examples}")

    for i, (lang_examples, dir) in enumerate(zip(examples, args.dirs)):
        if lang_examples < num_examples:
            lang_samples = np.random.choice(lang_examples, size=num_examples, replace=True)
        else:
            lang_samples = np.random.choice(lang_examples, size=num_examples, replace=False)
        lang_samples_dev = set(np.random.choice(lang_samples, size=1000, replace=False))
        lang_samples_train = set(lang_samples).difference(lang_samples_dev)

        write_files(dir, lang_samples_dev, args.dev_output_dir, i + args.lang_num_start_idx)
        write_files(dir, lang_samples_train, args.output_dir, i + args.lang_num_start_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--dirs', type=str, nargs="+")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--lang_num_start_idx', type=int, default=0)
    parser.add_argument('--num_examples', type=int, default=-1)
    parser.add_argument('--dev_output_dir', type=str, help='1000 sequences held-out for validation')
    args = parser.parse_args()

    main(args)
