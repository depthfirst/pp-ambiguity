import sys
import os
import re
import nltk
import json
import numpy as np
import sklearn
import spacy
import pandas as pd
import tensorflow as tf
import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score as kappa
from itertools import groupby
from xml.etree import ElementTree as ET
from sklearn import svm
from collections import Counter

from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from sklearn.neural_network import MLPClassifier

sys.path.append('/bridge/science/AI/nlp/bert')
import tokenization

BERT_BASEDIR = "/bridge/science/AI/nlp/corpora/BERT/wwm_uncased_L-24_H-1024_A-16"

def load_text_file(file):
    lines = []
    with open(file) as text_in:
        for line in text_in:
            lines.append(line.strip())
    return lines


def load_xml_files(xmldir):
    xmlfiles = sorted(os.listdir(xmldir))[:-1]
    sents = []
    wordidmap = {}
    dependencies = {}
    for xmlfile in xmlfiles:
        xmlpath = os.path.join(xmldir,xmlfile)
        #print(xmlpath)
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        batch_code = root.attrib['{http://www.w3.org/XML/1998/namespace}id']
        snodes = root.findall('{http://ilk.uvt.nl/folia}text')[0].findall('{http://ilk.uvt.nl/folia}s')
        alltext = [[t[0].text for t in [w.findall('{http://ilk.uvt.nl/folia}t') for w in s.findall('{http://ilk.uvt.nl/folia}w')]] for s in snodes]
        for s in snodes:
            sent = []
            for w in s.findall('{http://ilk.uvt.nl/folia}w'):
                word_id = w.attrib['{http://www.w3.org/XML/1998/namespace}id']
                if 'space' in w.attrib:
                    space = not (w.attrib['space']=="no")
                else:
                    space = True
                tnodes = w.findall('{http://ilk.uvt.nl/folia}t')
                text = tnodes[0].text
                posnodes = w.findall('{http://ilk.uvt.nl/folia}pos')
                if posnodes[0].attrib['set'] == 'https://raw.githubusercontent.com/proycon/folia/master/setdefinitions/spacy/spacy-en-pos':
                    pos = posnodes[0].attrib['class']
                elif posnodes[1].attrib['set'] == 'https://raw.githubusercontent.com/proycon/folia/master/setdefinitions/spacy/spacy-en-pos':
                    pos = posnodes[1].attrib['class']
                else:
                    pos = "UNK"
                lemma = w.findall('{http://ilk.uvt.nl/folia}lemma')[0].attrib['class']
                wtpl = (text,word_id,pos,lemma,space)
                wordidmap[word_id] = wtpl
                sent.append(wtpl)
            depsroots = s.findall('{http://ilk.uvt.nl/folia}dependencies')
            if depsroots is not None and len(depsroots)>0:
                depsroot = depsroots[0]
                for dep in depsroot.findall('{http://ilk.uvt.nl/folia}dependency'):
                    if not dep.attrib['class']=='prep':
                        continue
                    heads = dep.findall('{http://ilk.uvt.nl/folia}hd')
                    if heads is None or len(heads)==0:
                        continue
                    wrefs = heads[0].findall('{http://ilk.uvt.nl/folia}wref')
                    if wrefs is None or len(wrefs)==0:
                        continue
                    head = None
                    for wref in wrefs:
                        wref_id = wref.attrib['id']
                        t = wref.attrib['t']
                        wtpl = wordidmap[wref_id]
                        head = wtpl
                    deps = dep.findall('{http://ilk.uvt.nl/folia}dep')
                    if deps is None or len(deps)==0:
                        continue
                    wrefs = deps[0].findall('{http://ilk.uvt.nl/folia}wref')
                    if wrefs is None or len(wrefs)==0:
                        continue
                    prep = None
                    for wref in wrefs:
                        wref_id = wref.attrib['id']
                        t = wref.attrib['t']
                        wtpl = wordidmap[wref_id]
                        prep = wtpl
                    dependencies[prep] = head
            sents.append(sent)
    return sents, dependencies, wordidmap

def generate_tuples(sents):
    preplemmas = []
    pobjs = {}
    # Look for VNPN patterns
    # NN* => noun phrase
    # VB* => verb phrase
    # IN => prep
    # ADJ/RB/DT/CD/... => no change
    for s in sents:
        state = "start"
        prev = "none"
        head = None
        prep = None
        tpl = []
        for w in s:
            pos = w[2]
            prev = state
            if pos == 'IN':
                state = "prep"
                if prev in ["start","verb"]:
                    #print("S|V => P;",w)
                    tpl = []
                    continue
                if prev == "noun":
                    #print("N => P;",w)
                    if len(tpl)>0:
                        tpl.append(head)
                        if len(tpl)>3:
                            yield tuple(tpl)
                head = w
                prep = w
            elif pos in ['NN','NNS','NNP']:
                state = "noun"
                if prev == "verb":
                    #print("V => N;",w)
                    tpl.append(head)
                elif prev == "prep":
                    if len(tpl)>1:
                        #print("*P => N;",w)
                        tpl.append(head)
                head = w
            #elif w in chunkwords:
            #    if not prev == "noun":
            #        print("???",prev,w)
            elif pos in ['VB','VBZ','VBG','VBD','VBN','VBP']:
                if not prev=="verb":
                    if prev=="noun" and len(tpl)==3:
                        tpl.append(head)
                        yield tuple(tpl)
                    tpl = []
                    #print("V;",w)
                state = "verb"
                head = w
            else:
                #print("?;",w)
                if prev == "noun" and len(tpl)==3:
                    tpl.append(head)
                    yield tuple(tpl)
        if state=="noun" and len(tpl)==3:
            tpl.append(head)
            yield tuple(tpl)

def find_sentence_from_file(f, snum, sents):
    for s in sents:
        word_id_segments = s[0][1].split('.')
        sf = word_id_segments[0]
        ssnum = int(word_id_segments[3])
        #print(sf,ssnum)
        if f==sf and int(snum)==ssnum:
            return s

def find_sentence_from_word_id(word_id, sents):
    word_id_segments = word_id.split('.')
    sf = word_id_segments[0]
    ssnum = word_id_segments[3]
    return find_sentence_from_file(sf, ssnum, sents) 

tokify = lambda tpl : "%s "%tpl[0] if tpl[4] else tpl[0]
stext = lambda s : ''.join([tokify(tpl) for tpl in s])
wtpl_stext = lambda wtpl, sents : stext(sents[int(wtpl[1].split('.')[3])-1])
sentence_text = lambda f,snum,sents : stext(find_sentence_from_file(f,snum,sents))
find_sentence_from_4tpl = lambda t4tpl,sents : find_sentence_from_word_id(t4tpl[0][1], sents)

def generate_annotated4tpls(tpls, dependencies):
    for tpl in tpls:
        tprep = tpl[2]
        if tprep in dependencies:
            #tdeps[tprep] = dependencies[tprep]
            #continue
            yield tpl, dependencies[tprep]
        for prep,attachment in dependencies.items():
            if prep[1]==tprep[1]:
                yield tpl, attachment
                #tdeps[tprep] = attachment
                #continue
        #missed+=1
        ##if len(tpl)==4 and tpl[2] in tdeps:
        #    yield tpl

def load_folia_xml(folia_dir, cross_dependencies=None):
    sents, dependencies, wordidmap = load_xml_files(folia_dir)
    def generate_4tpls():
        dependency_check = lambda prep, deps: deps is None or prep in deps
        for tpl in generate_tuples(sents):
            if len(tpl)!=4:
                continue
            tprep = tpl[2]
            if not dependency_check(tprep,cross_dependencies):
                continue
            if dependency_check(tprep, dependencies):
                yield tpl, dependencies[tprep]
                continue
            for prep, attachment in dependencies.items():
                if prep[1]==tprep[1]:
                    yield tpl, attachment
                    continue
    return sents, generate_4tpls

def generate_sentences_from_4tpls(annotated4tpls, sents):
    for t4tpl in annotated4tpls:
        yield stext(find_sentence_from_word_id(t4tpl[0][1], sents))

def generate_google_instances(annotated4tpls, sents, jsonl_file, bert_basedir=BERT_BASEDIR,
    labels=[], omit_indexes=[]):
    sents_all = list(generate_sentences_from_4tpls(annotated4tpls,sents))
    #sents_all=[stext(find_sentence_from_word_id(t4tpl[0][1],sents)) for t4tpl in annotated4tpls]
    bertdicts_all = []
    with open(jsonl_file) as allin:
        for s in allin:
            bert_dict = json.loads(s)
            bertdicts_all.append(bert_dict)
    bertdicts_all = [bd for i,bd in enumerate(bertdicts_all) if i not in omit_indexes]
    assert len(sents_all)==len(bertdicts_all)
    num_instances = len(annotated4tpls)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_basedir,"vocab.txt"),
        do_lower_case=True)
    for i,(annotated4tpl,bertdict,label) in enumerate(zip(annotated4tpls,bertdicts_all,labels)):
        sent = stext(find_sentence_from_word_id(annotated4tpl[0][1],sents))
        sent_from_orig_tokenization = find_sentence_from_4tpl(annotated4tpl,sents)
        orig_tokens = [t[0] for t in sent_from_orig_tokenization]
        bert_tokens = []
        bert_tokens.append("[CLS]")
        orig_token_indexes = [int(tpl[1].split('.')[-1])-1 for tpl in annotated4tpl]
        orig_4tpl_tokens = [orig_tokens[i] for i in orig_token_indexes]
        orig_token_bert_arrays = []
        orig_tokens_pieces = []
        for orig_token in orig_tokens:
            #orig_to_tok_map.append(len(bert_tokens))
            word_pieces = tokenizer.tokenize(orig_token)
            num_word_pieces = len(word_pieces)
            layeridx=3
            # If token >1 piece, use layers from word pieces (4 total)
            # 4th-from-top layer from first piece...
            # top layer from 4th (or last) piece
            wpi=0
            if (len(bert_tokens)+num_word_pieces)>len(bertdict['features']):
                print("Too many pieces")
                bert_tokens.extend(word_pieces)
                continue
            token_layers_values = []
            orig_token_pieces=[]
            while layeridx>=0:
                token_bertdict = bertdict['features'][len(bert_tokens)+wpi]
                #tli = 3
                tli = layeridx
                #tli = 3-layeridx
                #tli = 0
                orig_token_pieces.append("{}[{}]".format(word_pieces[wpi],tli))
                if not token_bertdict['token']==word_pieces[wpi]:
                    print("Token mismatch: {}<>{}".format(token_bertdict['token'],word_pieces[wpi]))
                token_layer_values = token_bertdict['layers'][tli]['values']
                token_layers_values.extend([token_layer_values])
                layeridx-=1
                if wpi<(num_word_pieces-1):
                    wpi+=1
            flattened_layers = [xi for layer in token_layers_values for xi in layer]
            orig_token_bert_arrays.append(flattened_layers)
            bert_tokens.extend(word_pieces)
        bert_tokens.append("[SEP]")
        x = []
        orig_4tpl_pieces = []
        for orig_token_idx in orig_token_indexes:
            x.extend(orig_token_bert_arrays[orig_token_idx])
        yield np.array(x)
    # sentences, bertdict)

def generate_huggingface_instances(model,tokenizer,
                                  annotated4tpls, 
                                  sents,
                                  labels,
                                  max_length=128,
                                  pad_to_max_length=True,
                                  use_cuda=False):
    sents_all = list(generate_sentences_from_4tpls(annotated4tpls,sents))
    if use_cuda:
        model.to('cuda')
    for i,(annotated4tpl,sent,label) in enumerate(zip(annotated4tpls,sents_all,labels)):
        orig_tokens = [t[0] for t in find_sentence_from_4tpl(annotated4tpl,sents)]
        bert_tokens = ["[CLS]"]
        orig_token_indexes = [int(tpl[1].split('.')[-1])-1 for tpl in annotated4tpl]
        orig_4tpl_tokens = [orig_tokens[i] for i in orig_token_indexes]
        orig_bert_token_indexes = []
        word_pieces_array = []
        for orig_token in orig_tokens:
            #orig_to_tok_map.append(len(bert_tokens))
            orig_bert_token_indexes.append(len(bert_tokens))
            word_pieces = tokenizer.tokenize(orig_token)
            word_pieces_array.append(word_pieces)
            bert_tokens.extend(word_pieces)
        bert_tokens.append("[SEP]")
        indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = tokenizer.encode(indexed_tokens,
                                         max_length=max_length,
                                         pad_to_max_length=pad_to_max_length,
                                         return_tensors='pt')
        if use_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
        x = []
        y = []
        with torch.no_grad():
            # When output_hidden_states = True, 
            # the hidden states are output in the third value
            # in the tuple returned from the model. 
            # That value is itself a tuple of the embedding matrix and
            # hidden layers, 1-N (where N is the number of hidden layers)
            # We want the last 4 layers of 24, which will be found in 
            # elements 21-24 of the second return tuple (embedding matrix is
            # element 0). 
            hidden_layers = model(tokens_tensor)[2][21:25]
            for orig_token_idx in orig_token_indexes:
                word_pieces = word_pieces_array[orig_token_idx]
                num_word_pieces = len(word_pieces)
                # If token >1 piece, use layers from word pieces (4 total)
                # 4th-from-top layer from first piece...
                # top layer from 4th (or last) piece

                layeridx = 3
                wpi = 0
                token_layers_values = []
                orig_bert_token_index = orig_bert_token_indexes[orig_token_idx]
                while layeridx>=0:
                    #tli = 0
                    tli=layeridx
                    #tli = 3-layeridx
                    token_layer_values = hidden_layers[tli][0,orig_bert_token_index+wpi]
                    #token_layers_values.extend([token_layer_values])
                    #y.extend(token_layer_values)
                    x.extend(token_layer_values)
                    layeridx-=1
                    if wpi<(num_word_pieces-1):
                        wpi+=1
                #flattened_layers = [xi for layer in token_layers_values for xi in layer]
                #x.extend(flattened_layers)
        #assert np.array_equal(x,y)
        #print("{} => {}".format(orig_4tpl_tokens,orig_4tpl_pieces))
        yield x