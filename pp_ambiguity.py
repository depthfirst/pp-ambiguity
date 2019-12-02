import re
import os

sent = ''
pp_ambiguity = []
star = re.compile(r'^\*')
all_sents = []
with open("nlvr/nlvr2/util/annotated_dev_examples.txt") as nlvr2_dev_in:
    for line in nlvr2_dev_in:
        s = line.strip()
        if len(s)<=1:
            continue
        m = re.match(star, s)
        if m is None:
            sent = s
            all_sents.append(sent)
        elif s == '* pp ambiguity':
            pp_ambiguity.append(sent)

with open("nlvr/nlvr2/util/pp_ambiguity.txt", "w") as nlvr2_pp_out:
    for pp in pp_ambiguity:
        nlvr2_pp_out.write("{}\n".format(pp))
with open("nlvr/nlvr2/util/all_sents.txt", "w") as nlvr2_all_out:
    for sent in all_sents:
        nlvr2_all_out.write("{}\n".format(sent))
