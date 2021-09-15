from conllu import parse_incr
import os, argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str)
    parser.add_argument("--output_directory", type=str)

    args = parser.parse_args()

    files = os.listdir(args.input_directory)
    total_lines_read = 0

    for file in files:
        if not '-wikipedia' in file: continue
        with open(args.output_directory + file[:-6] + 'txt', 'w') as out_file:
            print(file)
            file_iter = 0
            if '-wikipedia' in file:
                with open(args.input_directory + file, 'r') as f:
                    for toklist in parse_incr(f):
                        out_file.write(toklist.metadata['text'] + ' ')
                        out_file.write('\n')
                        file_iter += 1

                        if file_iter % 200000 == 0:
                            print('Finished line {} of file {}'.format(file_iter, file))
                print('Finished file: {}'.format(file))
                total_lines_read += file_iter

    print('Total lines read: {}'.format(total_lines_read))

