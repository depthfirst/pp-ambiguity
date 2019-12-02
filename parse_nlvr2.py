import spacy
from benepar.spacy_plugin import BeneparComponent

print("Loading spacy model")
nlp = spacy.load('en_core_web_lg')
print("Initializing benepar component")
nlp.add_pipe(BeneparComponent('benepar_en'))
all_parses = []
all_sents = []
with open('nlvr/nlvr2/util/pp_ambiguity.txt') as all_sent_in:
    print("Loading pp_ambiguity.txt")
    for line in all_sent_in:
        s = line.strip()
        doc = nlp(s)
        parses = [sent._.parse_string for sent in list(doc.sents)]
        sents = [sent.text for sent in list(doc.sents)]
        for sent, parse in zip(sents,parses):
            all_sents.append(sent)
            all_parses.append(parse)

    #text_full = [line.strip() for line in all_sent_in]
    #print("{} examples loaded.".format(len(text_full)))
    #doc = nlp('\n'.join(text_full))
    #sents_full = [sent.text for sent in list(doc.sents)]
    #print("{} sentences.".format(len(sents_full)))
    #parses_full = [sent._.parse_string for sent in list(doc.sents)]
    #print("{} parses loaded.".format(len(parses_full)))
    fnpp = 'pp_ambiguity_parses.txt'
    fnsn = 'pp_ambiguity_sents.txt'
    with open(fnpp,'w') as all_parse_out, open(fnsn,'w') as pp_sent_out:
        for parse,sent in zip(all_parses,all_sents):
            all_parse_out.write('{}\n'.format(parse))
            pp_sent_out.write('{}\n'.format(sent))

