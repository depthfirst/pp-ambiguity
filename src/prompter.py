import json
import sys
import numpy as np
import pandas as pd
import spacy
import re

from tqdm import tqdm as progress_bar, trange
from sklearn.metrics import accuracy_score
from collections import defaultdict


class Prompter(): 
    samples = ["There are dogs near the edge.", "There are dogs of water.", "."]
    def __init__(self):
        self.initialize()

    def initialize(self):
        pass

    # Can I implement generator and iterator patterns? 

    def preprocess(self, source_dict={}, promptcolprefix=None):
        if 'prompt' not in source_dict:
            source_dict['prompt'] = source_dict['sentence_text']
        source_dict['class'] = "orig"
        yield source_dict

    def from_file(self, input_file, promptcolprefix=None):
        context = [] # the context stores a conversation history. 
                     # you can use this to make the model more context aware
        examples = []
        if input_file[-3:]=='csv':
            df = pd.read_csv(input_file, header=0, index_col='annidx')
            #print(f"Processing {df.shape[0]} records.")
            #pbar = trange(, desc=input_file)
            for i in range(df.shape[0]):
                rec = df.iloc[i].to_dict()
                annidx = int(df.index[i])
                rec['annidx'] = annidx
                # Summary: input_file is processed, one record at a time. Each record is loaded 
                # into a dictionary. Then we call generate() with the record. 
                # generate() returns a dictionary. No. whatever generate() returns is saved 
                # to the dictionary as 'response'. 
                for newrec in self.preprocess(rec, promptcolprefix=promptcolprefix):
                    yield newrec

        elif input_file[-4:]=='json':
            input_recs = json.load(input_file)
            for rec in progress_bar(json.load(input_file)):
                for newrec in self.preprocess(rec, promptcolprefix=promptcolprefix):
                    yield newrec

        elif input_file[-9:]=='jsonlines':
            with open(input_file) as jsoninput:
                for line in jsoninput:
                    rec = json.loads(line.strip())
                    for newrec in self.preprocess(rec, promptcolprefix=promptcolprefix):
                        yield newrec
        else:
            with open(input_file, "r") as prompts:
                for prompt in prompts:
                    prompt = prompt.strip()
                    for newrec in self.preprocess(self, {"prompt": prompt}, promptcolprefix=promptcolprefix):
                        yield newrec

    def init_rec(self, source_dict):
        req_entries = ["annidx", "sentence_text", "X", "P1", "Y", "P2", "Z"]
        opt_entries = ["attachment"]
        rec = {}
        for entry in req_entries: 
            if entry not in source_dict:
                raise ValueError(f"Entry not found: '{entry}'")
            rec[entry] = source_dict[entry]
        for entry in opt_entries:
            if entry in source_dict:
                rec[entry] = source_dict[entry]
        return rec

    def interactive(self):
        '''
        Just modified to accept lines of the form
        #name: value
        which will have the effect of adding an entry for 'name'
        to the `source_dict` if it doesn't already exist
        and setting it to `value`. 
        Would it be better to do this in preprocess to be more 
        generally applicable? 
        '''
        prompt_again = True
        examples = []
        print(f"Enter prompts to send to model. ")
        print("Enter '.' on line by itself to end prompt; 'bye' to end session): ")
        while prompt_again:
            prompt=input(f"Please Enter Prompt: ")
            promptlines = []
            rec = {}
            while prompt!='.' and not prompt.lower()[:3]=='bye':
                if len(prompt)==0:
                    promptlines.append("\n")
                elif prompt[0]=='#':
                    if ':' in prompt:
                        name, value = prompt[1:].split(":")
                        rec[name] = value
                else:
                    promptlines.append(prompt)
                prompt = input()
            if prompt.lower()[:3]=='bye':
                prompt_again = False
            else:
                prompt = "\n".join(promptlines)
                print(f"You entered the following prompt:\n'{prompt}'.") 
                ack = input("Is that correct? [Y/n]")
                ack = "Y"
                if len(ack)==0 or ack.lower()[0]=='y':
                    rec["prompt"] = prompt
                    rec["sentence_text"] = prompt
                    for newrec in self.preprocess(rec):
                        yield newrec
                else:
                    print("OK, let's try again.")

    def test_mode(self, prompts=samples):
        for prompt in prompts:
            if len(prompt)<=1 or prompt.lower()[:3]=='bye':
                print("Bye!")
                break
            else:
                yield {"prompt": prompt}

class RawPrompter(Prompter):
    def preprocess(self, source_dict=None, promptcolprefix=None):
        yield source_dict

class NLPPrompter(Prompter):
    def initialize(self):
        self.nlp = spacy.load("en_core_web_trf")

    def parse_caption(self, caption):
        doc = self.nlp(caption)
        preps = []
        nps = []
        np = []
        for i, token in enumerate(doc):
            if token.tag_ == 'IN':
                preps.append({'text': token.text, 'index': i})
                nps.append(np)
                np = []
            else:
                np.append(token)
        nps.append(np)
        if len(preps)!=2 or len(nps)!=3:
            raise ValueError(f"What went wrong? {source_dict['sentence_text']}")
        x0 = nps[0][0]
        #if x0.tag_ == 'DT':
        #    X = ' '.join([nps[0][1].text.lower()]+[x.text for x in nps[0][2:]])
        #else:
        X = ' '.join([x0.text.lower()]+[x.text for x in nps[0][1:]])
        xhead = nps[0][-1]
        y0 = nps[1][0]
        #if y0.tag_ == 'DT':
        #    Y = ' '.join([y.text for y in nps[1][1:]])
        #else:
        Y = ' '.join([y.text for y in nps[1]])
        yhead = nps[1][-1]
        p1 = preps[0]['text']
        p2 = preps[1]['text']
        xdt = ""
        if xhead.tag_ in ['NNS','NNPS', 'VBZ']:
            xv = "are"
            if p2 == 'with':
                xv2 = "have"
            else:
                xv2 = "are"
        elif x0.tag_ in ['CD']:
            if x0.text.lower() == 'one':
                xv = "is"
                if p2 == 'with':
                    xv2 = "has"
                else:
                    xv2 = "is"
            else:
                xv = "are"
                if p2 == 'with':
                    xv2 = "have"
                else:
                    xv2 = "are"
        elif xhead.tag_ in ['NN', 'NNP', 'VBD', 'VB', 'VBN']:
            xv = "is"
            if x0.tag_ != 'DT':
                if nps[0][0].text[0] in ['a','e','i','o','u']:
                    xdt = "an"
                else:
                    xdt = "a"
            if p2 == 'with':
                xv2 = "has"
            else:
                xv2 = "is"
        else:
            print(caption)
            print(f"xhead.text has tag '{xhead.tag_}'")
            raise ValueError()
        last_token = nps[2][-1]
        if last_token.tag_ != '.':
            #print(f"Last token: {last_token.text} has tag {last_token.tag_}")
            Z = ' '.join([y.text for y in nps[2]])
        else:
            Z = ' '.join([y.text for y in nps[2][:-1]])
        if yhead.tag_ in ['NNS','NNPS', 'VBZ']:
            yv = "are"
            if p2 == 'with':
                yv2 = "have"
            else:
                yv2 = "are"
        elif y0.tag_ in ['CD']:
            if y0.text.lower() == 'one':
                yv = "is"
                if p2 == 'with':
                    yv2 = "has"
                else:
                    yv2 = "is"
            else:
                yv = "are"
                if p2 == 'with':
                    yv2 = "have"
                else:
                    yv2 = "are"
        elif yhead.tag_ in ['NN', 'NNP', 'VBD', 'PRP']:
            yv = "is"
            if p2 == 'with':
                yv2 = "has"
            else:
                yv2 = "is"
        else:
            print(caption)
            print(f"yhead.text has tag '{yhead.tag_}'")
            raise ValueError()
        params = dict(zip(['doc','X','p1', 'Y', 'p2', 'Z', 'xdt', 'xv', 'xv2', 'xhead', 'yv', 'yv2', 'yhead'],
            [doc, X, p1, Y, p2, Z, xdt, xv, xv2, xhead, yv, yv2, yhead]))
        return params

class PrepRelYesNoPrompter(Prompter):
    def initialize(self):
        with open("data/preprels.json") as relin:
            self.preprels = json.load(relin)
        self.keyrels = ['attribute', 'activity', 'agent', 'participant', 'cause', 'location', 'temporal', 'via']

    def make_prompt(self, sentence_text, prep, ppobj, rel):
        if rel=='temporal':
            option = "time period"
        else: # if rel in ['location', 'attribute', 'activity', 'destination', 'numeric']:
            option = rel
        art = "an" if option[0] in ['a','e','i','o','u'] else "a"
        prompt = f"""In "{sentence_text}", does "{prep} {ppobj}" specify {art} {option}? Yes or No? """
        return prompt

    def init_rec(self, source_dict):
        rec = super().init_rec(source_dict)
        rec["context"] = "You are a helpful assistant that answers questions with only 'yes' or 'no'."
        return rec

    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        sentence_text = source_dict['sentence_text']
        X = source_dict['X']
        p1 = source_dict['P1']
        Y = source_dict['Y']
        p2 = source_dict['P2']
        Z = source_dict['Z']

        if p1 not in self.preprels:
            self.preprels[p1] = ["other"]
        for rel in self.preprels[p1]:
            if rel not in self.keyrels:
                continue
            rec = self.init_rec(source_dict)
            rec['prompt'] = self.make_prompt(sentence_text, p1, Y, rel)
            rec['class'] = f"p1-{p1}-{rel}"
            yield rec

        if p2 not in self.preprels:
            self.preprels[p2] = ["other"]
        for rel in self.preprels[p2]:
            if rel not in self.keyrels:
                continue
            rec = self.init_rec(source_dict)
            rec['prompt'] = self.make_prompt(sentence_text, p2, Z, rel)
            rec['class'] = f"p2-{p2}-{rel}"
            yield rec


class PrepRelationPrompter(Prompter):
    def initialize(self):
        with open("data/preprels.json") as relin:
            self.preprels = json.load(relin)
        self.context = "You are a helpful assistant that answers questions with only the letter of the correct choice."
    
    def init_rec(self, source_dict):
        rec = super().init_rec(source_dict)
        rec["context"] = self.context
        return rec

    def make_prompt(self, source_dict, prep, ppobj, pclass="p1rel"):
        sentence_text = source_dict["sentence_text"]
        prompt_intro= f"""In the phrase "{sentence_text}", which of the following best describes the role of the relation "{prep} {ppobj}"? 
"""
        options = []
        choice = 'A'
        for rel in self.preprels[prep]:
            if type(rel)==dict: 
                rellab = rel["label"]
                prompt = rel["prompts"][pclass]
            elif type(rel)==str:
                rellab = rel
                prompt = rel
            else:
                raise TypeError
            for var in ["X","P1","Y","P2","Z"]:
                # Replaces occurrences of "{X}"" with value of X, etc.
                if var=="X":
                    srcvar = source_dict[var].lower()
                else:
                    srcvar = source_dict[var]

                prompt = re.sub("{{{}}}".format(var), srcvar, prompt)
            option = f"({choice}) {prompt}"
            options.append(option)
            choice = chr(ord(choice) + 1)
        return "\n".join([prompt_intro] + options)

    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        sentence_text = source_dict['sentence_text']
        X = source_dict['X']
        p1 = source_dict['P1']
        Y = source_dict['Y']
        p2 = source_dict['P2']
        Z = source_dict['Z']

        # Hacky way - don't filter by preposition here
        #if p1=="of":
        rec = self.init_rec(source_dict)
        rec['prompt'] = self.make_prompt(source_dict, p1, Y)
        rec['class'] = "p1rel"
        yield rec

        rec = self.init_rec(source_dict)
        rec['prompt'] = self.make_prompt(source_dict, p2, Z)
        rec['class'] = "p2rel"
        yield rec
'''

        if p2 in ['at','in','on','of','with','near']:
        if p2=="near":
            rec = self.init_rec(source_dict)
            rec['prompt'] = self.make_prompt(sentence_text, p2, Z)
            rec['class'] = "p2rel"
            yield rec
'''

class PrepSensePrompter(Prompter):
    def initialize(self):
        self.prep_senses = defaultdict(list)
        preps = ["in", "on", "at", "with", "near"]
        datadir = "../data"
        for prep in preps: 
            with open(f"{datadir}/{prep}.txt") as prepin:
                for line in prepin:
                    self.prep_senses[prep].append(line.strip())

    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        sentence_text = source_dict['sentence_text']
        X = source_dict['X']
        p1 = source_dict['P1']
        Y = source_dict['Y']
        p2 = source_dict['P2']
        Z = source_dict['Z']
        if p2 in self.prep_senses and p1!=p2:
            rec = self.init_rec(source_dict)
            senses = self.prep_senses[p2]
            promptlines = [f"Given the word \"{p2}\" in the input sentence, choose the correct meaning from the following:"] 
            for i,sense in enumerate(senses):
                promptlines.append(f"{chr(ord('A')+i)}) {sense}")
            promptlines.append("Generate only the letter of the correct option.")
            promptlines.append(f"Input: \"{sentence_text}\"\n")
            prompt = "\n".join(promptlines)
            rec['prompt'] = prompt
            rec['context'] = ""
            rec['class'] = "P2sense"
            yield rec


class OrigPrompter(Prompter):

    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        #params = self.parse_caption(source_dict['sentence_text'])
        X = source_dict['X']
        p1 = source_dict['P1']
        Y = source_dict['Y']
        p2 = source_dict['P2']
        Z = source_dict['Z']

        rec = self.init_rec(source_dict)
        #rec['prompt'] = source_dict['sentence_text']
        rec['prompt'] = f"{X} {p1} {Y} {p2} {Z}."
        rec['class'] = 'XpYpZ'
        rec['context'] = ''
        yield rec

        rec = self.init_rec(source_dict)
        rec['prompt'] = f"{X} {p1} {Y}."
        rec['class'] = 'XpY'
        rec['context'] = ''
        yield rec

        rec = self.init_rec(source_dict)
        rec['prompt'] = f"{X} {p2} {Z}."
        rec['class'] = 'XpZ'
        rec['context'] = ''
        yield rec

        rec = self.init_rec(source_dict)
        rec['prompt'] = f"{Y.capitalize()} {p2} {Z}."
        rec['class'] = 'YpZ'
        rec['context'] = ''
        yield rec

class AdjectivePrompter(Prompter):

    def preprocess(self, source_dict=None, promptcolprefix=None):
        '''
        '''

        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        doc = self.nlp(source_dict['sentence_text'])
        preps = []
        nps = []
        np = []
        for i, token in enumerate(doc):
            if token.tag_ == 'IN':
                preps.append({'text': token.text, 'index': i})
                nps.append(np)
                np = []
            else:
                np.append(token)
        nps.append(np)
        if len(preps)!=2 or len(nps)!=3:
            raise ValueError(f"What went wrong? {source_dict['sentence_text']}")
        x0 = nps[0][0]
        if x0.tag_ == 'DT':
            X = ' '.join([nps[0][1].text.lower()]+[x.text for x in nps[0][2:]])
        else:
            X = ' '.join([x0.text.lower()]+[x.text for x in nps[0][1:]])
        xhead = nps[0][-1]
        y0 = nps[1][0]
        if y0.tag_ == 'DT':
            Y = ' '.join([y.text for y in nps[1][1:]])
        else:
            Y = ' '.join([y.text for y in nps[1]])
        yhead = nps[1][-1]
        p1 = preps[0]['text']
        p2 = preps[1]['text']
        if xhead.tag_ in ['NNS','NNPS', 'VBZ']:
            xv = "are"
            if p2 == 'with':
                xv2 = "have"
            else:
                xv2 = "are"
        elif x0.tag_ in ['CD']:
            if x0.text.lower() == 'one':
                xv = "is"
                if p2 == 'with':
                    xv2 = "has"
                else:
                    xv2 = "is"
            else:
                xv = "are"
                if p2 == 'with':
                    xv2 = "have"
                else:
                    xv2 = "are"
        elif xhead.tag_ in ['NN', 'NNP', 'VBD', 'VB', 'VBN']:
            xv = "is a"
            if p2 == 'with':
                xv2 = "has"
            else:
                xv2 = "is"
        else:
            print(source_dict['sentence_text'])
            print(f"xhead.text has tag '{xhead.tag_}'")
            raise ValueError()
        last_token = nps[2][-1]
        if last_token.tag_ != '.':
            #print(f"Last token: {last_token.text} has tag {last_token.tag_}")
            Z = ' '.join([y.text for y in nps[2]])
        else:
            Z = ' '.join([y.text for y in nps[2][:-1]])
        context = f"There {xv} {X} {p1} {Y} {p2} {Z}."
        #prompt = f"There is a {source_dict['sentence_text']}. The fact that the {X} is {p1} {Y} is"
        if p2 == 'with':
            prompt = f"The fact that the {X} {xv2} {Z} is"
        else:
            prompt = f"The fact that the {X} {xv2} {p2} {Z} is"        
        varadjs = ["surprising", "irrelevant", "informative", "relevant", 
                   "interesting", "redundant", "typical", "silly", "funny",
                   "wrong", "unexpected", "expected"]
        for varadj in varadjs:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
            # rec = # copy source_dict
            rec["prompt"] = f"{prompt} {varadj}."
            rec["context"] = context
            rec["class"] = "XpZ"
            rec["variant"] = varadj
            yield rec

        if xhead.tag_ in ['NNS','NNPS', 'VBZ']:
            xv = "are"
        elif x0.tag_ in ['CD']:
            if x0.text.lower() == 'one':
                xv = "is"
            else:
                xv = "are"
        elif xhead.tag_ in ['NN', 'NNP', 'VBD']:
            xv = "is a"
        # what about 'themselves'? 
        if yhead.tag_ in ['NNS','NNPS', 'VBZ']:
            yv = "are"
            if p2 == 'with':
                yv2 = "have"
            else:
                yv2 = "are"
        elif y0.tag_ in ['CD']:
            if y0.text.lower() == 'one':
                yv = "is"
                if p2 == 'with':
                    yv2 = "has"
                else:
                    yv2 = "is"
            else:
                yv = "are"
                if p2 == 'with':
                    yv2 = "have"
                else:
                    yv2 = "are"
        elif yhead.tag_ in ['NN', 'NNP', 'VBD', 'PRP']:
            yv = "is a"
            if p2 == 'with':
                yv2 = "has"
            else:
                yv2 = "is"
        else:
            print(source_dict['sentence_text'])
            print(f"yhead.text has tag '{yhead.tag_}'")
            raise ValueError()
        last_token = nps[2][-1]
        if last_token.tag_ != '.':
            #print(f"Last token: {last_token.text} has tag {last_token.tag_}")
            Z = ' '.join([z.text for z in nps[2]])
        else:
            Z = ' '.join([z.text for z in nps[2][:-1]])

        #prompt = f"There is a {source_dict['sentence_text']}. The fact that the {X} is {p1} {Y} is"
        if p2 == 'with':
            prompt = f"The fact that the {Y} {yv2} {Z} is"
        else:
            prompt = f"The fact that the {Y} {yv2} {p2} {Z} is"        
        for varadj in varadjs:

            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}

            # copy source_dict
            rec["prompt"] = f"{prompt} {varadj}."
            rec["context"] = context
            rec["class"] = "YpZ"
            rec["variant"] = varadj
            yield rec

class HerePrompter(Prompter):
    def preprocess(self, source_dict=None, promptcolprefix=None):
        #if source_dict is None or 'sentence_text' not in source_dict:
        #    raise ValueError("Entry not found: 'sentence_text'")
        #params = self.parse_caption(source_dict['sentence_text'])
        X = source_dict['X'].lower()
        p1 = source_dict['P1']
        Y = source_dict['Y']
        p2 = source_dict['P2']
        Z = source_dict['Z']

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {X} {p1} {Y}."
        rec["class"] = "XpY"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {X} {p2} {Z}."
        rec["class"] = "XpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {Y} {p2} {Z}."
        rec["class"] = "YpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {X} {p1} {Y} {p2} {Z}."
        rec["class"] = "XpYpZ"
        yield rec

class HereNormPrompter(NLPPrompter):
    def preprocess(self, source_dict=None, promptcolprefix=None):
        if 'X' not in source_dict or 'Y' not in source_dict:
            if 'sentence_text' not in source_dict:
                raise ValueError("Entry not found: 'sentence_text'")
            params = self.parse_caption(source_dict['sentence_text'])
            X = params['X'].lower()
            Y = params['Y']
        else:
            X = source_dict['X'].lower()
            Y = source_dict['Y']
        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {X}."
        rec["class"] = "X"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {Y}."
        rec["class"] = "Y"
        yield rec

class OldTherePrompter(NLPPrompter):
    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        params = self.parse_caption(source_dict['sentence_text'])
        X = params["X"]
        p1 = params["p1"]
        Y = params["Y"]
        p2 = params["p2"]
        Z = params["Z"]
        xv = params["xv"]
        yv = params["yv"]
        xdt = params["xdt"]
        if len(xdt)>0:
            xdt = f"{xdt} "
        rec = self.init_rec(source_dict)
        rec["prompt"] = f"There {xv} {xdt}{X} {p1} {Y}."
        rec["class"] = "XpY"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"There {xv} {xdt}{X} {p2} {Z}."
        rec["class"] = "XpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"There {yv} {Y} {p2} {Z}."
        rec["class"] = "YpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"There {xv} {xdt}{X} {p1} {Y} {p2} {Z}."
        rec["class"] = "XpYpZ"
        yield rec

class TherePrompter(NLPPrompter):
    def nwords(self, phrase):
        return len([t for t in self.nlp(phrase)])

    def parse_caption(self, sentence_text):
        params = super().parse_caption(sentence_text)
        X = params["X"]
        p1 = params["p1"]
        Y = params["Y"]
        p2 = params["p2"]
        Z = params["Z"]
        xv = params["xv"]
        yv = params["yv"]
        xdt = params["xdt"]
        if len(xdt)>0:
            xdt = f"{xdt} "
        stext = sentence_text
        singulars = ['NN','NNP','PRP','VBN','VB']
        plurals = ['NNS','VBZ']
        pcap = params['doc']
        x = X.lower()
        nx = len(x.split())
        np1 = min(len(p1.split()),self.nwords(p1))
        ny = min(len(Y.split()),self.nwords(Y))
        np2 = len(p2.split()) #nwords(p2)
        nz = len(Z.split()) #nwords(z)
        xidx = 0
        p1idx = nx
        yidx = p1idx+np1
        p2idx = yidx+ny
        zidx = p2idx+np2
        tokens = [tok for tok in pcap]
        xhead = tokens[nx-1]
        yhead = tokens[p2idx-1]
        xdt2 = "The"
        ydt2 = "The"
        xdt1 = ""
        ydt1 = ""
        if tokens[0].tag_ in ['DT','PRP$']: #, 'JJ']:
            #print("Yay! I found a DT: {}".format(tokens[0]))
            xc = x[len(tokens[0].text)+1:]
            #xdt = tokens[0].text
        else:
            xc =x
            if tokens[0].text.lower()[0] in ['a','e','i','o','u']:
                xdt1 = "an "
            else:
                xdt1 = "a "
        if tokens[yidx].tag_ in ['DT', 'PRP$']: #, 'JJ']:
            #print("Yay! I found a DT: {}".format(tokens[yidx]))
            yc = Y[len(tokens[yidx].text)+1:]
            if tokens[yidx].tag_=='PRP$':
                ydt2 = tokens[yidx].text.capitalize()
        else:
            yc = Y
            if yhead.tag_ in singulars:
                if tokens[yidx].text.lower()[0] in ['a','e','i','o','u']:
                    ydt1 = "an "
                else:
                    ydt1 = "a "
        if xhead.tag_ in singulars:
            v1 = "is"
            if p2=='with':
                xv = "has"
                xp2z = " {}".format(Z)
            else:
                xv = "is"
                xp2z = " {} {}".format(p2,Z)
        elif xhead.tag_ in plurals:
            v1 = "are"
            xdt1 = ""
            if p2=='with':
                xv = "have"
                xp2z = " {}".format(Z)
            else:
                xv = "are"
                xp2z = " {} {}".format(p2,Z)
        else:
            raise ValueError("{} Unrecognized tag for X head '{}': {}".format(stext,xhead,xhead.tag_))
            
        if yhead.tag_ in singulars:
            if p2=='with':
                yv = "has"
                yp2z = " {}".format(Z)
            else:
                yv = "is"
                yp2z = " {} {}".format(p2, Z)
        elif yhead.tag_ in plurals:
            if p2=="with":
                yv = "have"
                yp2z = " {}".format(Z)
            else:
                yv = "are"
                yp2z = " {} {}".format(p2, Z)
        else:
            raise ValueError("{} Unrecognized tag for Y head '{}': {}".format(stext,yhead,yhead.tag_))

        newcaps1 = "There {} {}{} {} {}{}.".format(v1,xdt1,x,p1,ydt1,Y)
        newcapx = "{} {} {} {}".format(newcaps1, xdt2, xc, xv)
        newcapxz = "{}.".format(xp2z)
        newcapy = "{} {} {} {}".format(newcaps1, ydt2, yc, yv)
        newcapyz = "{}.".format(yp2z)
        params["v1"]   = v1
        params["xdt1"] = xdt1
        params["x"] = x
        params["ydt1"] = ydt1
        params["xc"]   = xc
        params["xv"]   = xv
        params["nc1"]  = newcaps1
        params["xdt2"] = xdt2
        params["ydt2"] = ydt2
        params["yc"]   = yc
        params["yv"]   = yv
        params["xp2z"] = xp2z
        params["yp2z"] = yp2z
        params["ncx"]  = newcapx
        params["ncxz"] = newcapxz
        params["ncy"]  = newcapy
        params["ncyz"] = newcapyz
        return params

    def preprocess(self, source_dict=None, promptcolprefix='pe'):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        params = self.parse_caption(source_dict['sentence_text'])
        X = params["X"]
        p1 = params["p1"]
        Y = params["Y"]
        p2 = params["p2"]
        Z = params["Z"]
        xv = params["xv"]
        yv = params["yv"]
        xdt = params["xdt"]
        if len(xdt)>0:
            xdt = f"{xdt} "
        stext = source_dict['sentence_text']
        newcapx = params["ncx"]
        newcapxz = params["ncxz"]
        newcapy  = params["ncy"]
        newcapyz  = params["ncyz"]

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"There {xv} {xdt}{X} {p1} {Y}."
        rec["class"] = "XpY"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = "{}{}".format(newcapx, newcapxz)
        rec["class"] = "XpZ"

        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = "{}{}".format(newcapy,newcapyz)
        rec["class"] = "YpZ"

        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"There {xv} {xdt}{X} {p1} {Y} {p2} {Z}."
        rec["class"] = "XpYpZ"
        yield rec

class DualPrompter(TherePrompter):
    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        params = self.parse_caption(source_dict['sentence_text'])
        X = params["X"]
        p1 = params["p1"]
        Y = params["Y"]
        p2 = params["p2"]
        Z = params["Z"]
        xv = params["xv"]
        yv = params["yv"]
        xdt = params["xdt"]
        if len(xdt)>0:
            xdt = f"{xdt} "
        newcapx = params["ncx"]
        newcapxz = params["ncxz"]
        newcapy  = params["ncy"]
        newcapyz  = params["ncyz"]

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {X} {p1} {Y}."
        rec["class"] = "XpY"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = "{}{}".format(newcapx, newcapxz)
        rec["class"] = "XpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = "{}{}".format(newcapy, newcapyz)
        rec["class"] = "YpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {X} {p1} {Y} {p2} {Z}."
        rec["class"] = "XpYpZ"
        yield rec

class MattersPrompter(TherePrompter):
    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        params = self.parse_caption(source_dict['sentence_text'])
        X = params["X"]
        p1 = params["p1"]
        Y = params["Y"]
        p2 = params["p2"]
        Z = params["Z"]
        xv = params["xv"]
        yv = params["yv"]
        x  = params["x"]
        xdt = params["xdt"]
        if len(xdt)>0:
            xdt = f"{xdt} "
        newcapx  = params["ncx"]
        newcapxz = params["ncxz"]
        newcapy  = params["ncy"]
        newcapyz = params["ncyz"]
        nc1      = params["nc1"]
        xp2z     = params["xp2z"]
        yp2z     = params["yp2z"]
        v1       = params["v1"]
        xc       = params["xc"]
        yc       = params["yc"]

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"There {v1} {xdt}{X} {p1} {source_dict['Y']}."
        rec["class"] = "XpY"
        yield rec

        XpYpZ = f"There {v1} {xdt}{X} {p1} {Y} {p2} {Z}."
        if yv=="has":
            yv2 = "does"
        elif yv=="have":
            yv2 = "do"
        else:
            yv2 = yv
        if xv=="has":
            xv2 = "does"
        elif xv=="have":
            xv2 = "do"
        else:
            xv2 = xv

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"{XpYpZ} That means {x} {xv}{xp2z} but the {yc} {yv2} not."
        #rec["prompt"] = f"{XpYpZ} That means {x} {xv}{xp2z} but {Y} {yv2} not."
        rec["class"] = "XpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"{XpYpZ} That means {X} {xv}{xp2z} and not the {yc}."
        rec["class"] = "XpZ2"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"{XpYpZ} That means {Y} {yv}{yp2z} but the {xc} {xv2} not."
        rec["class"] = "YpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"{XpYpZ} That means {Y} {yv}{yp2z} and not the {xc}."
        rec["class"] = "YpZ2"
        yield rec

        rec = self.init_rec(source_dict)
        if yv in ["has", "have"]:

            rec["prompt"] = f"{XpYpZ} That means {X} {xv}{xp2z} and {Y} also {yv}{yp2z}."
        else:
            rec["prompt"] = f"{XpYpZ} That means {X} {xv}{xp2z} and {Y} {yv} also{yp2z}."
        rec["class"] = "XpYpZ"
        yield rec

        rec = self.init_rec(source_dict)
        if yv in ["has","have"]:

            rec["prompt"] = f"{XpYpZ} That means both {X} and {Y} have{xp2z}."
        else:
            rec["prompt"] = f"{XpYpZ} That means both {X} and {Y} are{xp2z}."
        rec["class"] = "XpYpZ2"
        yield rec


class InfoStructPrompter(NLPPrompter):

    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        doc = self.nlp(source_dict['sentence_text'])
        preps = []
        nps = []
        np = []
        for i, token in enumerate(doc):
            if token.tag_ == 'IN':
                preps.append({'text': token.text, 'index': i})
                nps.append(np)
                np = []
            else:
                np.append(token)
        nps.append(np)
        if len(preps)!=2 or len(nps)!=3:
            raise ValueError(f"What went wrong? {source_dict['sentence_text']}")
        x0 = nps[0][0]
        if x0.tag_ == 'DT':
            X = ' '.join([nps[0][1].text.lower()]+[x.text for x in nps[0][2:]])
        else:
            X = ' '.join([x0.text.lower()]+[x.text for x in nps[0][1:]])
        xhead = nps[0][-1]
        y0 = nps[1][0]
        #if y0.tag_ == 'DT':
        #    Y = ' '.join([y.text for y in nps[1][1:]])
        #else:
        Y = ' '.join([y.text for y in nps[1]])
        yhead = nps[1][-1]
        p1 = preps[0]['text']
        p2 = preps[1]['text']
        if xhead.tag_ in ['NNS','NNPS', 'VBZ']:
            xv = "are"
        elif x0.tag_ in ['CD']:
            if x0.text.lower() == 'one':
                xv = "is"
            else:
                xv = "are"
        elif xhead.tag_ in ['NN', 'NNP', 'VBD', 'VB', 'VBN']:
            xv = "is a"
        else:
            print(source_dict['sentence_text'])
            print(f"xhead.text has tag '{xhead.tag_}'")
            raise ValueError()
        last_token = nps[2][-1]
        if last_token.tag_ != '.':
            #print(f"Last token: {last_token.text} has tag {last_token.tag_}")
            Z = ' '.join([y.text for y in nps[2]])
        else:
            Z = ' '.join([y.text for y in nps[2][:-1]])
        context = f"There {xv} {X} {p1} {Y} {p2} {Z}."
        #prompt = f"There is a {source_dict['sentence_text']}. The fact that the {X} is {p1} {Y} is"
        prompt = f"There {xv} {X} {p1} {Y}."
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"]}
        else:
            rec = {}
        # rec = # copy source_dict
        rec["prompt"] = prompt
        rec["context"] = ""
        rec["class"] = "XpY"

        yield rec
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"]}
        else:
            rec = {}
        rec["prompt"] = context
        rec["context"] = ""
        rec["class"] = "XpYpZ"
        yield rec
        context = f"There {xv} {X} {p1} {Y} {p2} {Z}."
        #prompt = f"There is a {source_dict['sentence_text']}. The fact that the {X} is {p1} {Y} is"
        prompt = f"There {xv} {X} {p2} {Z}."
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"]}
        else:
            rec = {}
        # rec = # copy source_dict
        rec["prompt"] = prompt
        rec["context"] = ""
        rec["class"] = "XpZ"
        yield rec

        if yhead.tag_ in ['NNS','NNPS', 'VBZ']:
            yv = "are"
        elif y0.tag_ in ['CD', 'PRP']:
            if y0.text.lower() == 'one':
                yv = "is"
            else:
                yv = "are"
        elif yhead.tag_ in ['NN', 'NNP', 'VBD', 'VB', 'VBN']:
            yv = "is"
        else:
            print(source_dict['sentence_text'])
            print(f"'{yhead.text}' has tag '{yhead.tag_}'")
            raise ValueError()
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"]}
        else:
            rec = {}

        rec["prompt"] = f"There {yv} {Y} {p2} {Z}."
        rec["context"] = ""
        rec["class"] = "YpZ"
        yield rec

class ThisIsPrompter(NLPPrompter):
    def preprocess(self, source_dict=None, promptcolprefix=None):
        if source_dict is None or 'sentence_text' not in source_dict:
            raise ValueError("Entry not found: 'sentence_text'")
        doc = self.nlp(source_dict['sentence_text'])
        preps = []
        nps = []
        np = []
        for i, token in enumerate(doc):
            if token.tag_ == 'IN':
                preps.append({'text': token.text, 'index': i})
                nps.append(np)
                np = []
            else:
                np.append(token)
        nps.append(np)
        if len(preps)!=2 or len(nps)!=3:
            raise ValueError(f"What went wrong? {source_dict['sentence_text']}")
        x0 = nps[0][0]
        if x0.tag_ == 'DT':
            X = ' '.join([nps[0][1].text.lower()]+[x.text for x in nps[0][2:]])
        else:
            X = ' '.join([x0.text.lower()]+[x.text for x in nps[0][1:]])
        xhead = nps[0][-1]
        y0 = nps[1][0]

        Y = ' '.join([y.text for y in nps[1]])
        yhead = nps[1][-1]
        p1 = preps[0]['text']
        p2 = preps[1]['text']
        if xhead.tag_ in ['NNS','NNPS', 'VBZ']:
            xv = "These are"
        elif x0.tag_ in ['CD']:
            if x0.text.lower() == 'one':
                xv = "This is"
            else:
                xv = "These are"
        elif xhead.tag_ in ['NN', 'NNP', 'VBD', 'VB', 'VBN']:
            xv = "This is a"
        else:
            print(source_dict['sentence_text'])
            print(f"xhead.text has tag '{xhead.tag_}'")
            raise ValueError()
        last_token = nps[2][-1]
        if last_token.tag_ != '.':
            Z = ' '.join([y.text for y in nps[2]])
        else:
            Z = ' '.join([y.text for y in nps[2][:-1]])
        context = f"{xv} {X} {p1} {Y} {p2} {Z}."
        prompt = f"{xv} {X} {p1} {Y}."
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"]}
        else:
            rec = {}
        # rec = # copy source_dict
        rec["prompt"] = prompt
        rec["context"] = ""
        rec["class"] = "XpY"
        yield rec
        
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"]}
        else:
            rec = {}
        rec["prompt"] = context
        rec["context"] = ""
        rec["class"] = "XpYpZ"
        yield rec
        
        context = f"{xv} {X} {p1} {Y} {p2} {Z}."
        prompt = f"{xv} {X} {p2} {Z}."
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"]}
        else:
            rec = {}
        # rec = # copy source_dict
        rec["prompt"] = prompt
        rec["context"] = ""
        rec["class"] = "XpZ"
        yield rec

        if yhead.tag_ in ['NNS','NNPS', 'VBZ']:
            yv = "These are"
        elif y0.tag_ in ['CD', 'PRP']:
            if y0.text.lower() == 'one':
                yv = "This is"
            else:
                yv = "These are"
        elif yhead.tag_ in ['NN', 'NNP', 'VBD', 'VB', 'VBN']:
            yv = "This is"
        else:
            print(source_dict['sentence_text'])
            print(f"'{yhead.text}' has tag '{yhead.tag_}'")
            raise ValueError()
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"]}
        else:
            rec = {}
        rec["prompt"] = f"{yv} {Y} {p2} {Z}."
        rec["context"] = ""
        rec["class"] = "YpZ"
        yield rec
