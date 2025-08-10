#!/bridge/science/laboratory/conda/envs/nlp310/bin/python
import json
import requests
import sys
import os
import pandas as pd
import argparse
import json
import torch
import transformers
import numpy as np
import spacy

from transformers import AutoTokenizer, AutoModelForCausalLM
from prompter import ModelPrompter
from torch.nn import functional as F

class Llama3Prompter(ModelPrompter):

    samples = ["There are dogs near the edge.", "There are dogs of water.", "."]
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B"):
        super().__init__(model_name)

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.nlp = spacy.load("en_core_web_trf")
    
    def generate(self, prompt, context=[], source_dict={}):
        #print(type(self.tokenizer))
        print(prompt)
        encodings = self.tokenizer(prompt)
        #print(encodings.keys())
        input_ids = torch.tensor([encodings["input_ids"]])
        #print("input_ids.shape=",input_ids.shape)
        #print(input_ids[0,:].tolist())
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            #print(type(outputs))
            logits = outputs.logits
            # Need labels to compute loss
            loss = outputs.loss
            if loss is None:
                print(prompt, "No loss returned.")
            if logits is None:
                print(prompt, "No logits returned.")
                raise ValueError(f"{prompt} No logits returned.")

        #print(f"logits.shape={logits.shape}")
        probabilities = F.softmax(logits, dim = -1)[0]
        #print(f"probabilities.shape={probabilities.shape}")
        token_probs = probabilities[range(probabilities.shape[0]-1), input_ids[0, 1:]]
        # -1 because we're discounting the start token '<|start|>'. 
        p2zlen = len(self.tokenizer(f"{source_dict['P2']} {source_dict['Z']}")['input_ids'])-1
        # -1 because we're discounting the '.'
        p2zprobs = token_probs[-1-p2zlen:-1]
        print(f"token_probs={token_probs}")
        prob = torch.exp(torch.sum(torch.log(p2zprobs))).item()
        #perplexity = float(torch.exp(-loss).numpy())

        return prob

    def get_prompts(self, source_dict=None, promptcolprefix=None):
        prompts = []
        if promptcolprefix is not None:
            prompts.append(f"{source_dict['sentence_text']}")
            prompts.append(source_dict[f"{promptcolprefix}_x"])
            prompts.append(source_dict[f"{promptcolprefix}_y"])
        else:
            x = source_dict['X']
            p1 = source_dict['P1']
            y = source_dict['Y']
            p2 = source_dict['P2']
            z = source_dict['Z']
            xhtag = self.nlp(x)[-1].tag_
            if xhtag in ['NNS','NNPS']:
                xv = 'are'
            else:
                xv = 'is'
            yhtag = self.nlp(y)[-1].tag_
            if yhtag in ['NNS','NNPS']:
                yv = 'are'
            else:
                yv = 'is'
            context = f"There {xv} {X} {p1} {Y} {p2} {Z}."
            for var in ['XpY', 'XpZ', 'YpZ']:
                rec = {"annidx": source_dict['annidx'], "sentence_text": source_dict["sentence_text"],
                       "X": source_dict["X"], "P1": source_dict["P1"], "Y": source_dict["Y"],
                       "P2": source_dict["P2"], "Z": source_dict["Z"], "attachment": source_dict["attachment"]}

                # copy source_dict
                if var == "XpY":

                    rec["prompt"] = f"There {xv} {x.lower()} {p1} {y}.\n"
                elif var == "XpZ":
                    rec["prompt"] = f"There {xv} {x.lower()} {p2} {z}.\n"
                elif var == "YpZ":
                    rec["prompt"] = f"There {yv} {y} {p2} {z}.\n"
                rec["context"] = context
                rec["class"] = var
                prompts.append(rec)
        return prompts

    def test_mode(self, prompts=samples):
        prompt_again = True
        examples = []
        for prompt in self.samples:
            if len(prompt)<=1 or prompt.lower()[:3]=='bye':
                print("Bye!")
                break
            else:
                response = self.generate(prompt, [])
                print(f"{prompt}: {response}")
                #print(f"{self.model_name}'s response: '{response}'.")
                examples.append({"prompt": prompt, "response": response})
        #if examples[0]['response'] > examples[1]['response']:
        #    print("Y")
        #else:
        #    print("X")
        return examples

    def process_response(self, response, source_dict={}):
        self.tokenizer(f"{source_dict['P2']} {source_dict['Z']}")
        return np.exp(np.sum(np.log(response)))

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--interactive', help="Interactive mode", action='store_true', required=False)
    group.add_argument("-f", '--input-file', metavar="<file>", help="Input file", required=False)
    group.add_argument('-t', '--test-mode', action='store_true', required=False, help="Submit sample prompts")                
    parser.add_argument("-o", '--output-file', metavar="<file>", help="Output file", 
        default=None, required=False)
    parser.add_argument("--promptcolprefix", metavar="<pcp>", help="Columns named <pcp>_x, <pcp>_y",
        default=None, required=False)
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    prompter = Llama3Prompter()

    if args.interactive:
        print("Entering interactive mode")
        prompter.interactive(args.output_file)
    elif args.input_file is not None:
        print("Reading input from",args.input_file)
        prompter.read_input(args.input_file, args.output_file, promptcolprefix=args.promptcolprefix)
    elif args.test_mode:
        print("Using sample prompts")
        prompter.test_mode()
    else:
        print("Invalid arguments")
        parser.print_help()
        sys.exit(2)

if __name__ == "__main__":
    main()
