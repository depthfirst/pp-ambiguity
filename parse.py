import spacy
from benepar.spacy_plugin import BeneparComponent
import sys
import re


txtfile = sys.argv[1]
print("Parsing text from {}".format(txtfile))
sentfile = re.sub(r'\.[a-z0-9_]+$', '.sent', txtfile)
print("Writing sentences to {}".format(sentfile))
outfile = re.sub(r'\.[a-z0-9_]+$', '.parse', txtfile)
print("Writing parse trees to {}".format(outfile))

print("Loading spacy model")
nlp = spacy.load('en_core_web_lg')
print("Initializing benepar component")
nlp.add_pipe(BeneparComponent('benepar_en'))
all_parses = []
all_sents = []
with open(txtfile) as all_txt_in:
    for line in all_txt_in:
        s = line.strip()
        doc = nlp(s)
        parses = [sent._.parse_string for sent in list(doc.sents)]
        sents = [sent.text for sent in list(doc.sents)]
        for sent, parse in zip(sents,parses):
            all_sents.append(sent)
            all_parses.append(parse)
    with open(outfile, 'w') as all_parse_out, open(sentfile,'w') as pp_sent_out:
        for parse,sent in zip(all_parses,all_sents):
            all_parse_out.write('{}\n'.format(parse))
            pp_sent_out.write('{}\n'.format(sent))

