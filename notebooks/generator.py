import os
import json
import torch
import numpy as np
import spacy
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

class Generator():
    def generate_instances(self, dataset, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        pass

    def generate_dataset(self, inputs, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        pass

class CountVectorizerGenerator(Generator):
    def __init__(self, binarize=True, tokenizer=None, vocabulary=None, lowercase=True, ngram_range=(1,1)):
        self.binarize=binarize
        self.vocabulary=vocabulary
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.vectorizer = CountVectorizer(tokenizer=tokenizer, vocabulary=vocabulary, 
                analyzer='word', min_df=1, 
                ngram_range=ngram_range, binary=binarize, lowercase=False)
        else:
            self.vectorizer = CountVectorizer(vocabulary=vocabulary, min_df=1, lowercase=lowercase, 
                binary=binarize, ngram_range=ngram_range)
            self.tokenizer = self.vectorizer.build_tokenizer()

    def instance_to_4tpl(self, rec):
        return " ".join([rec['V']['lemma'],rec['N']['lemma'],rec['P']['lemma'],rec['N2']['lemma']])
        
    def fit_transform(self, inputs):
        X = self.vectorizer.fit_transform([self.instance_to_4tpl(instance) for instance in inputs])
        #X = self.vectorizer.fit_transform([instance['sentence_text'] for instance in inputs])
        self.vocabulary = self.vectorizer.vocabulary_
        return X.toarray()

    def transform(self, inputs):
        if self.vocabulary is None:
            raise ValueError("Vocabulary not set")
        X = self.vectorizer.transform([self.instance_to_4tpl(instance) for instance in inputs])
        return X.toarray()

    def generate_instances(self, inputs, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        for instance in inputs:
            yield self.transform([instance])

    def generate_dataset(self, inputs, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        return np.array([x for x in self.generate_instances(inputs)])


class HuggingFaceGenerator(Generator):
    def __init__(self, model_name):
        self.model = self.load_bert_model(model_name)
        self.tokenizer = self.load_bert_tokenizer(model_name)

    def load_bert_model(self,model_name, bert_config=None):
        if bert_config is None:
            bert_config = BertConfig.from_pretrained(model_name)
            bert_config.output_hidden_states=True

        bert_model = BertModel.from_pretrained(model_name,config=bert_config)
        bert_model.eval()

        return bert_model

    def load_bert_tokenizer(self,model_name, bert_config=None):
        if bert_config is None:
            bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            bert_tokenizer = BertTokenizer.from_pretrained(model_name,config=bert_config)
        return bert_tokenizer

    def generate_dataset(self, inputs, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        return np.array([x for x in self.generate_instances(inputs,orig_tokenizer,max_length,pad_to_max_length,use_cuda)])

    def generate_instances(self,inputs,
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        sents_all = [instance['sentence_text'] for instance in inputs]
        if use_cuda:
            self.model.to('cuda')
        retuple = lambda word_attr : (word_attr['text'],
                                      word_attr['source'],
                                      word_attr['pos_tag'],
                                      word_attr['lemma'],
                                      word_attr['trail_space'])
        for instance in inputs:
            annotated4tpl = (retuple(instance['V']),
                             retuple(instance['N']),
                             retuple(instance['P']),
                             retuple(instance['N2']))
            sent = instance['sentence_text']
            label = instance['label']
            if 'tokenized_sentence' in instance:
                orig_tokens = [t[0] for t in instance['tokenized_sentence']]
            elif orig_tokenizer is not None:
                orig_tokens = [t.text for t in orig_tokenizer(sent)]
            else:  
                orig_tokens = sent
            bert_tokens = ["[CLS]"]
            orig_token_indexes = [int(tpl[1].split('.')[-1])-1 for tpl in annotated4tpl]
            orig_4tpl_tokens = [orig_tokens[i] for i in orig_token_indexes]
            orig_bert_token_indexes = []
            word_pieces_array = []
            for orig_token in orig_tokens:
                #orig_to_tok_map.append(len(bert_tokens))
                orig_bert_token_indexes.append(len(bert_tokens))
                word_pieces = self.tokenizer.tokenize(orig_token)
                word_pieces_array.append(word_pieces)
                bert_tokens.extend(word_pieces)
            bert_tokens.append("[SEP]")
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
            tokens_tensor = self.tokenizer.encode(indexed_tokens,
                                             max_length=max_length,
                                             pad_to_max_length=pad_to_max_length,
                                             return_tensors='pt')
            if use_cuda:
                tokens_tensor = tokens_tensor.to('cuda')
            x = []
            with torch.no_grad():
                # When output_hidden_states = True,
                # the hidden states are output in the third value
                # in the tuple returned from the model.
                # That value is itself a tuple of the embedding matrix and
                # hidden layers, 1-N (where N is the number of hidden layers)
                # We want the last 4 layers of 24, which will be found in
                # elements 21-24 of the second return tuple (embedding matrix is
                # element 0).
                hidden_layers = self.model(tokens_tensor)[2][21:25]
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
                        tli=layeridx
                        token_layer_values = hidden_layers[tli][0,orig_bert_token_index+wpi]
                        x.extend(token_layer_values)
                        layeridx-=1
                        if wpi<(num_word_pieces-1):
                            wpi+=1
            yield x

class NLVR2SentenceGenerator(Generator):
    def __init__(self, datadir):
        self.datadir = datadir

    def generate_inputs(self, settype="dev"):
        path_to_json_file=os.path.join(self.datadir, "%s.json"%settype)
        if not os.path.exists(path_to_json_file):
            raise FileNotFoundError("Not found: %s" % path_to_json_file)
        with open(path_to_json_file) as input_file:
            for jsonline in input_file:
                yield json.loads(jsonline)

    def generate_instances(self, inputs=None, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        if inputs is None:
            inputs = self.generate_inputs()
        sents_all = [instance['sentence'] for instance in inputs]
        for instance in Counter(sents_all):
            new_instance = {}
            new_instance['sentence_text'] = instance
            yield new_instance

    def generate_dataset(self, inputs=None, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        if inputs is None:
            inputs = self.generate_inputs()
        return [x for x in self.generate_instances(inputs,orig_tokenizer,max_length,pad_to_max_length,use_cuda)]

class MaskedPrepGenerator(HuggingFaceGenerator):
    
    def __init__(self,bert_model_name,spacy_model_name='en_core_web_lg', nlvr2_datadir="/bridge/data/compositional_semantics/nlvr2"):
        self.load_bert_model(bert_model_name)
        self.tokenizer = self.load_bert_tokenizer(bert_model_name)
        self.nlp = spacy.load(spacy_model_name)
        self.datadir = nlvr2_datadir

    def load_bert_model(self, model_name):
        self.model = BertForMaskedLM.from_pretrained(model_name)

    def transform(self, instances):
        return instances

    def generate_instances(self, inputs, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        sents_all = [instance['sentence_text'] for instance in inputs]
        if orig_tokenizer is None:
            orig_tokenizer = self.nlp
        if use_cuda:
            self.model.to('cuda')
        for instance in inputs:
            sentence = orig_tokenizer(instance['sentence_text'])
            '''
            if 'tokenized_sentence' in instance:
                orig_tokens = [(t[0],t[2]) in instance['tokenized_sentence']]
            else:
                if orig_tokenizer is None:
                    orig_tokenizer = spacy.load('en_core_web_lg')  
                orig_tokens = [(t.text,t.pos) for t in orig_tokenizer(sent)]
            '''
 # Problems: Need BERT Model, spaCy tokenizer, full dataset, 
 # https://huggingface.co/transformers/quickstart.html#bert-example
            bert_tokens = ["[CLS]"]
            orig_token_indexes = []
            orig_tokens = []
            orig_bert_token_indexes = []
            word_pieces_array = []
            masked_indexes = []
            correct_preps = []
            for i,orig_token in enumerate(sentence):
                #orig_to_tok_map.append(len(bert_tokens))
                orig_tokens.append(orig_token)
                orig_token_indexes.append(i)
                orig_bert_token_indexes.append(len(bert_tokens))
                word_pieces = self.tokenizer.tokenize(orig_token.text)
                num_word_pieces = len(word_pieces)
                if orig_token.tag_=='IN':
                    if orig_token.idx+len(orig_token.text)>len(sentence.text):
                        print("Warning: '{}' in sentence is past the end of the sentence: ".format(token.text))
                        print(sentence.text)
                        continue
                    correct_prep = orig_token.lemma_
                    correct_preps.append(correct_prep)
                    masked_index=len(bert_tokens)
                    masked_indexes.append(masked_index)
                bert_tokens.extend(word_pieces)
            bert_tokens.append("[SEP]")
            yield bert_tokens, masked_indexes, correct_preps

    def evaluate_dataset(self, settype="dev",
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        nlvr2_generator = NLVR2SentenceGenerator(datadir=self.datadir)
        nlvr2=list(nlvr2_generator.generate_inputs(settype=settype))
        nlvr2=list(nlvr2_generator.generate_instances(inputs=nlvr2))
        print("Evaluating {}, {} instances".format(settype,len(nlvr2)))
        num_correct = 0
        num_total = 0
        for bert_tokens, masked_indexes, correct_preps in self.generate_instances(nlvr2):
            for masked_index,correct_prep in zip(masked_indexes,correct_preps):
                bert_tokens[masked_index] = "[MASK]"
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
                #print(indexed_tokens)
                # Convert inputs to PyTorch tensors
                tokens_tensor = torch.tensor([indexed_tokens])
                #segments_tensors = torch.tensor([segments_ids])
                # If you have a GPU, put everything on cuda
                #tokens_tensor = self.tokenizer.encode(indexed_tokens,
                #                             max_length=max_length,
                #                             pad_to_max_length=pad_to_max_length,
                #                             return_tensors='pt')
                tokens_tensor = tokens_tensor.to('cuda')
                #segments_tensors = segments_tensors.to('cuda')
                self.model.to('cuda')

                # Predict all tokens
                with torch.no_grad():
                    outputs = self.model(tokens_tensor) #, token_type_ids=segments_tensors)
                    predictions = outputs[0]

                # confirm we were able to predict 'henson'
                predicted_index = torch.argmax(predictions[0, masked_index]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
                if predicted_token==correct_prep:
                    num_correct+=1
                #else:
                #    print("Pred: {}; Gold: {}; Sentence: {}".format(predicted_token,correct_prep,' '.join(bert_tokens)))
                num_total+=1
                if (num_total%100)==0:
                    #print("Torch Tensor Dimensions: {}".format(tokens_tensor.size()))
                    #print("BERT Tokens: {}".format(len(bert_tokens)))
                    print("({} correct / {} total)".format(num_correct, num_total))
        return num_correct, num_total

    def generate_dataset(self, inputs, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        if orig_tokenizer is None:
            orig_tokenizer = self.nlp
        return np.array([x for x in self.generate_instances(inputs,orig_tokenizer,max_length,pad_to_max_length,use_cuda)])


