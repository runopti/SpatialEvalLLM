import math
import json
import os
import argparse
from utils import read_jsonl 

def main(args):
    samples = read_jsonl(args.pred_file)
    filename = args.pred_file.replace(".jsonl", f"_nsampled-{args.nsampled}_nstep-{args.nstep}.jsonl")
    dic_list = []
    for sample in samples:
        num_steps = sample["question"].count("where you find")
        if num_steps == args.nstep:
            dic_list.append(sample)
            if len(dic_list) == args.nsampled:
                with open(filename, "w") as outfile:
                    for entry in dic_list:
                        json.dump(entry, outfile)
                        outfile.write('\n')
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--pred-file', type=str, metavar='N',
                        default='', help='prediction file')
    parser.add_argument('--nstep', type=int, required=True)
    parser.add_argument('--nsampled', type=int, required=True)
    args = parser.parse_args()
    main(args)
