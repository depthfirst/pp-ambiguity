import torch
from transformers import BertConfig, BertTokenizer, BertModel

class Generator():
    def generate_instances(self, dataset, 
                           orig_tokenizer=None,
                           max_length=128,
                           pad_to_max_length=True,
                           use_cuda=False):
        pass

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