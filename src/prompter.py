import json
import sys
import numpy as np
import pandas as pd
import spacy

from tqdm import tqdm as progress_bar, trange
from sklearn.metrics import accuracy_score

class Prompter(): 
    samples = ["There are dogs near the edge.", "There are dogs of water.", "."]
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.nlp = spacy.load("en_core_web_trf")

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
        if "attachment" in source_dict:
            rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                   "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}
        elif "annidx" in source_dict:
            rec = {"annidx": source_dict["annidx"], "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                   "P2": source_dict["P2"], "Z": source_dict["Z"]}
        else:
            rec = {}
        return rec

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
            print(source_dict['sentence_text'])
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
            print(source_dict['sentence_text'])
            print(f"yhead.text has tag '{yhead.tag_}'")
            raise ValueError()
        params = dict(zip(['X','p1', 'Y', 'p2', 'Z', 'xdt', 'xv', 'xv2', 'xhead', 'yv', 'yv2', 'yhead'],
            [X, p1, Y, p2, Z, xdt, xv, xv2, xhead, yv, yv2, yhead]))
        return params

    def interactive(self):
        prompt_again = True
        examples = []
        while prompt_again:
            prompt=input(f"Please Enter Prompt: ")
            if len(prompt)<=1 or prompt.lower()[:3]=='bye':
                print("Bye!")
                prompt_again = False

            else:
                print(f"You said '{prompt}'.") 
                ack = input("Is that correct? [Y/n]")
                ack = "Y"
                if len(ack)==0 or ack.lower()[0]=='y':
                    rec = {"prompt": prompt, "sentence_text": prompt}
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

class HereNormPrompter(HerePrompter):
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

class TherePrompter(Prompter):
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

class DualPrompter(HerePrompter):
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
        rec["prompt"] = f"Here we have {X} {p1} {Y}."
        rec["class"] = "XpY"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = source_dict[f"{promptcolprefix}_x"]
        rec["class"] = "XpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = source_dict[f"{promptcolprefix}_y"]
        rec["class"] = "YpZ"
        yield rec

        rec = self.init_rec(source_dict)
        rec["prompt"] = f"Here we have {X} {p1} {Y} {p2} {Z}."
        rec["class"] = "XpYpZ"
        yield rec


class InfoStructPrompter(Prompter):

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

class ThisIsPrompter(Prompter):
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
