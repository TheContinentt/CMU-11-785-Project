

train_in = './knnw/train.txt'
val_in = './knnw/val.txt'
test_in = './knnw/test.txt'


train_out = './knnw/train_full.txt'
val_out = './knnw/val_full.txt'
test_out = './knnw/test_full.txt'

csv_file = './knnw/knnw_en_sub.csv'

class_to_idx_file = './knnw/class_to_idx.txt'
class_to_idx = {}
with open(class_to_idx_file, 'r') as  f:
    for line in f:
        line = line.strip().split(',')
        class_to_idx[line[0]] = line[1]


id_speaker_file = './knnw/id_speaker.txt'
id_speaker = {}
with open(id_speaker_file, 'r') as f:
    for line in f:
        line = line.strip().split(',')
        id_speaker[line[0]] = line[1].strip()


with open(csv_file, 'r', encoding='utf-8') as csv_f:
    line_dict = {}
    for line_idx, line in enumerate(csv_f):
        line = line.strip().split(';')
        if line_idx >= 2:
            line_dict[line[0]] = line[-1]


with open(test_in, 'r') as train_in_f:
    with open(test_out, 'w') as train_out_f:
        for line in train_in_f:
            line = line.strip().split(',')

            print(line)
            line.append(class_to_idx[id_speaker[line[0]]])
            txt = line_dict[line[0]]
            line.append(txt)
            print(';'.join(line), file=train_out_f)

