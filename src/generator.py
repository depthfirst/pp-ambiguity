import sys
import json
import argparse
import transformers
import torch
import os

import pandas as pd
import numpy as np

from torch.nn import functional as F
from tqdm import tqdm as progress_bar
from transformers import AutoTokenizer, AutoModelForCausalLM

#from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr as correlation

from prompter import *
from evaluator import *

class Generator():

    '''
    What a mess. I was trying to get a single iterator to keep spitting out prompts
    and then another generator to spit out results. 
    '''
    def generate(self, inputs, outputfile=None):
        if outputfile is not None:
            output = open(outputfile, "w")

        for source_dict in inputs:
            if outputfile is None:
                print(f"{source_dict['prompt']}")
            response = self.process(source_dict)
            self.postprocess(source_dict, response=response)
            if outputfile is not None:
                json.dump(source_dict, output)
                output.write("\n")
            else:
                print(f"{response}")
            yield source_dict

    def process(self, source_dict={}):
        pass

    def postprocess(self, source_dict={}, response=None):
        source_dict["response"] = response

class ModelGenerator(Generator):
    model_name = "Unknown"

    def __init__(self, model_name="Model"):
        self.model_name = model_name
        self.initialize()

    def process(self, source_dict={}, response=None):
        pass

class Llama3Generator(ModelGenerator):

    samples = ["There are dogs near the edge.", "There are dogs of water.", "."]
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B"):
        super().__init__(model_name)

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_eos_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
    
    def process(self, source_dict={}):
        if 'prompt' not in source_dict:
            raise ValueError("'prompt' is required.")
        prompt = source_dict["prompt"]
        if 'context' in source_dict:
            context = source_dict['context']
        else:
            context = []
        if len(context)==0:
            input_text = f"{prompt}"
        else:
            input_text = f"{context} {prompt}"

        encodings = self.tokenizer(input_text)
        token_ids = encodings["input_ids"]+[self.tokenizer.eos_token_id]
        input_ids = torch.tensor([token_ids])
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            logits = outputs.logits
            # Need labels to compute loss
            loss = outputs.loss
            if logits is None:
                raise ValueError(f"{prompt} No logits returned.")

        probabilities = F.softmax(logits, dim = -1)[0]
        # Token probabilities are shifted left by 1
        # We want the probability that token t at index i follows tokens [0..t-1]
        token_probs = probabilities[range(probabilities.shape[0]-1), input_ids[0, 1:]]
        # -1 because we're discounting the start token '<|start|>'. 
        return self.aggregate_token_probs(source_dict, token_probs)

    def aggregate_token_probs(self, source_dict, token_probs):
        prob = token_probs[-1].item()
        return prob

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
        return examples

class Llama3P2ZGenerator(Llama3Generator):
    def aggregate_token_probs(self, source_dict, token_probs):
        if 'class' in source_dict and source_dict['class']=='XpY':
            offset = len(self.tokenizer(f"{source_dict['P1']} {source_dict['Y']}")['input_ids'])-1
        else:
            offset = len(self.tokenizer(f"{source_dict['P2']} {source_dict['Z']}")['input_ids'])-1
        probs = token_probs[-2-offset:-2]
        prob = torch.exp(torch.sum(torch.log(probs))).item()
        return prob

class Llama3ZPeriodGenerator(Llama3Generator):
    def aggregate_token_probs(self, source_dict, token_probs):
        zlen = len(self.tokenizer(f"{source_dict['Z']}.")['input_ids'])-1
        probs = token_probs[-1-zlen:-1]
        prob = torch.exp(torch.sum(torch.log(probs))).item()
        return prob


class Llama3P2ZLastTokenGenerator(Llama3Generator):

    def aggregate_token_probs(self, source_dict, token_probs):
        if "class" in source_dict and source_dict["class"] in ["XpZ", "YpZ"]:
            # -1 because we have a start token 
            p2zlen = len(self.tokenizer(f"{source_dict['P2']} {source_dict['Z']}")['input_ids'])-1
            probs = token_probs[-2-p2zlen:-2]
            prob = torch.exp(torch.sum(torch.log(probs))).item()
        else:
            prob = token_probs[-1].item()
        #print(f"token_probs={token_probs}")
        #perplexity = float(torch.exp(-loss).numpy())
        return prob

class Llama3P2ZPeriodGenerator(Llama3Generator):

    def aggregate_token_probs(self, source_dict, token_probs):
        if "class" in source_dict and source_dict["class"] in ["XpZ", "YpZ"]:
            # -1 because we have a start token 
            p2zlen = len(self.tokenizer(f"{source_dict['P2']} {source_dict['Z']}")['input_ids'])-1
            probs = token_probs[-2-p2zlen:-2]
            prob = torch.exp(torch.sum(torch.log(probs))).item()
        else:
            # corresponds to the predicted likelihood of the '.'
            prob = token_probs[-2].item()
        return prob

class Llama3LastTokenGenerator(Llama3Generator):
    # Identical to base implementation
    pass

class Llama3PadTokenGenerator(Llama3Generator):
    def process(self, source_dict={}):
        if 'prompt' not in source_dict:
            raise ValueError("'prompt' is required.")
        prompt = source_dict["prompt"]
        if 'context' in source_dict:
            context = source_dict['context']
        else:
            context = []
        if len(context)==0:
            input_text = f"{prompt}"
        else:
            input_text = f"{context} {prompt}"

        encodings = self.tokenizer(input_text)
        token_ids = encodings["input_ids"]+[self.tokenizer.pad_token_id]
        input_ids = torch.tensor([token_ids])
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            logits = outputs.logits
            # Need labels to compute loss
            loss = outputs.loss
            if logits is None:
                raise ValueError(f"{prompt} No logits returned.")

        probabilities = F.softmax(logits, dim = -1)[0]
        # Token probabilities are shifted left by 1
        # We want the probability that token t at index i follows tokens [0..t-1]
        token_probs = probabilities[range(probabilities.shape[0]-1), input_ids[0, 1:]]
        # -1 because we're discounting the start token '<|start|>'. 
        return self.aggregate_token_probs(source_dict, token_probs)

class Llama3FirstTokenGenerator(Llama3Generator):
    def aggregate_token_probs(self, source_dict, token_probs):
        prob = token_probs[0].item()
        return prob

class Llama3PeriodGenerator(Llama3Generator):
    def aggregate_token_probs(self, source_dict, token_probs):
        prob = token_probs[-2].item()
        return prob

class Llama3DualGenerator(Llama3P2ZPeriodGenerator):
    # Deprecated. Use P2ZPeriod
    pass

class VeraGenerator(ModelGenerator):

    def __init__(self, model_name="VERA"):
        super().__init__(model_name)

    def initialize(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera')
        self.model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera')
        self.model.D = self.model.shared.embedding_dim
        self.linear = torch.nn.Linear(self.model.D, 1, dtype=self.model.dtype)
        self.linear.weight = torch.nn.Parameter(self.model.shared.weight[32099, :].unsqueeze(0))
        self.linear.bias = torch.nn.Parameter(self.model.shared.weight[32098, 0].unsqueeze(0))
        self.model.eval()
        self.temperature = self.model.shared.weight[32097, 0].item() # temperature for calibration

    def process(self, source_dict={}):
        if 'prompt' not in source_dict:
            raise ValueError("'prompt' is required.")
        prompt = source_dict["prompt"]
        if 'context' in source_dict:
            context = source_dict['context']
        else:
            context = []
        if len(context)==0:
            input_text = prompt
        else:
            input_text = f"{context} {prompt}"

        input_ids = self.tokenizer.batch_encode_plus([input_text], return_tensors='pt', padding='longest', 
            truncation='longest_first', max_length=64).input_ids
        with torch.no_grad():
            output = self.model(input_ids)
            last_hidden_state = output.last_hidden_state
            hidden = last_hidden_state[0, -1, :]
            logit = self.linear(hidden).squeeze(-1)
            logit_calibrated = logit / self.temperature
            score_calibrated = logit_calibrated.sigmoid()
            return score_calibrated.item()
        raise SystemError("torch.no_grad() not available.")

def init_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--interactive', help="Interactive mode", action='store_true', required=False)
    group.add_argument("-f", '--input-file', metavar="<file>", help="Input file", required=False)
    group.add_argument('-s', '--sample-mode', action='store_true', required=False, help="Submit sample prompts")                
    parser.add_argument("-o", '--output-file', metavar="<file>", help="Output file", 
        default=None, required=False)
    parser.add_argument("-m", "--model", metavar="<model>", help="Model to use (llama3, vera)", 
        default="llama3", required=True)
    parser.add_argument("-p", '--prompter', metavar="<prompter>", help="Prompter to use (raw, here, herenorm, there, this, orig)", default="raw", required=True)
    parser.add_argument("-r", "--results_dir", metavar="<results_dir>", help="Directory for results files (jsonlines and html)", default="results")
    parser.add_argument("-t", '--token', metavar="<token>", help="Which token to use for probabilities (first,last, period, p2z)", required=False)
    parser.add_argument("--promptcolprefix", metavar="<pcp>", help="Columns named <pcp>_x, <pcp>_y",
        default=None, required=False)
    parser.add_argument("-b", "--boundary", type=str, choices={"all","default","svc"}, required=False, default="svc")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser

def init_prompter(prompter_name):
    # Initialize Prompter
    if prompter_name=="raw":
        prompter = RawPrompter()
    elif prompter_name=="there":
        prompter = TherePrompter()
    elif prompter_name=="this":
        prompter = ThisIsPrompter()
    elif prompter_name=="here":
        prompter = HerePrompter()
    elif prompter_name=="herenorm":
        prompter = HereNormPrompter()
    elif prompter_name=="orig":
        prompter = OrigPrompter()
    elif prompter_name=="dual":
        prompter = DualPrompter()
    else:
        prompter = Prompter()
    return prompter

def init_generator(model_name, prob_token="last"):
    # Initialize Generator (and Model)
    if model_name=="llama3":
        if prob_token=="period":
            generator = Llama3PeriodGenerator()
        elif prob_token=="first":
            generator = Llama3FirstTokenGenerator()
        elif prob_token=="last":
            generator = Llama3LastTokenGenerator()
        elif prob_token=="pad":
            generator = Llama3PadTokenGenerator()
        elif prob_token=="p2z":
            generator = Llama3P2ZGenerator()
        elif prob_token=="p2zlast":
            generator = Llama3P2ZLastTokenGenerator()
        elif prob_token=="p2zperiod":
            generator = Llama3P2ZPeriodGenerator()
        elif prob_token=="z.":
            generator = Llama3ZPeriodGenerator()
        elif prob_token=="dual":
            generator = Llama3DualGenerator()
        else: 
            generator = Llama3Generator()
    elif model_name=="vera":
        generator = VeraGenerator()
    else:
        raise IllegalArgument(f"Unknown model: '{model_name}'")
    return generator

def collect_inputs(args, prompter):
    # Initialize Prompter
    # Collect Inputs
    if args.interactive:
        print("Entering interactive mode")
        return prompter.interactive()
    elif args.input_file is not None:
        print("Reading input from",args.input_file)
        inputs = prompter.from_file(args.input_file, promptcolprefix=args.promptcolprefix)
        return inputs
    elif args.sample_mode:
        print("Using sample prompts")
        return prompter.test_mode()
        sys.exit(0)
    else:
        raise ValueError("Invalid arguments")



def main():

    parser = init_parser()
    args = parser.parse_args()
    plot_prefix = f"{args.results_dir}/info_vs_plausibility-{args.model.lower()}{args.token}-{args.prompter}"
    overwrite = False
    while args.output_file is not None and os.path.exists(args.output_file) and not overwrite:
        print(f"Output file {args.output_file} exists. ")
        print("What do you want to do?")
        print("(P)lot the results")
        print("(O)verwrite")
        print("(R)ename")
        print("(Q)uit")
        choice = input("Enter choice ([P]ORQ): ").strip()
        if choice is None or len(choice)==0:
            choice = "P"
        choice = choice[0].lower()
        if choice=='q':
            sys.exit(0)
        elif choice=='p':
            print("Plotting results.")
            with open(args.output_file) as jsonlines:
                journal = [json.loads(line.strip()) for line in jsonlines]
                fig01=plot_results(journal, 
                    model=args.model, prompter=args.prompter, token=args.token, boundary=args.boundary)
                fig01.write_html(f"{plot_prefix}.html")
                fig01.write_image(f"{plot_prefix}.png")        
            sys.exit(0)
        elif choice=='r':
            args.output_file = input("Enter new output_file: ").strip()
        elif choice=='o':
            overwrite = True
    prompter = init_prompter(args.prompter)
    generator = init_generator(args.model, args.token)
    inputs = collect_inputs(args, prompter)
    if args.interactive:
        journal = list(generator.generate(inputs, outputfile=args.output_file))
    else:
        journal = list(generator.generate(progress_bar(list(inputs)), outputfile=args.output_file))
        fig01=plot_results(journal, title="Information Structure vs Plausibility", 
            model=args.model.lower(), prompter=args.prompter, token=args.token,
            boundary=args.boundary)
        fig01.write_html(f"{plot_prefix}.html")
        fig01.write_image(f"{plot_prefix}.png")        
        if args.model.lower()=='vera':
            fig02=plot_results(journal, title="Information Structure vs Plausibility", 
                model=args.model.lower(), prompter=args.prompter, token=args.token,
                xcol="log_neg_log_ypz_over_xpz", ycol="log_neg_log_xpy_over_xpypz", 
                boundary=args.boundary)
            plot_prefix = f"{results_dir}/info_vs_plausibility-{args.model.lower()}-log-{args.prompter}"
            fig02.write_html(f"{plot_prefix}.html")
            fig02.write_image(f"{plot_prefix}.png")        

if __name__ == "__main__":
    main()
