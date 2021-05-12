import argparse
"""
python fusion.py
"""
parser = argparse.ArgumentParser(description='Process fusion')
parser.add_argument('--audio_feature', type=str, 
                    help='path for store audio features')
parser.add_argument('--text_feature',type=str, 
                    help='path for store text features')
parser.add_argument('--audio_wight', type=float, 
                    help='audio weight')
parser.add_argument('--text_weight',type=float, 
                    help='text weight')

args = parser.parse_args()

import numpy as np


audio = np.load(args.audio_feature)
text = np.load(args.text_feature)

fuse = audio * args.audio_wight + text * args.text_weight

predictions = np.argmax(fuse, axis=1)
with open('./out.csv', 'w') as out_f:
    for pred in predictions:
        print(pred, file=out_f)
