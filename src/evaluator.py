import sys
import json
import argparse
import transformers
import torch
import os
import re

import pandas as pd
import numpy as np

from torch.nn import functional as F
from tqdm import tqdm as progress_bar
from transformers import AutoTokenizer, AutoModelForCausalLM

from matplotlib import pyplot as plt
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from scipy.stats import pearsonr as correlation

from plotly import express as px
from plotly import graph_objects as go
from plotly.io import to_html as fig_to_html
from plotly.offline import iplot

#prels   = json.load(open("../data/preprels.json"))

def results_to_dataframe(
    results, 
    index="annidx", 
    drop_columns=['sentence_text', 'X', 'P1', 'Y', 'P2', 'Z', 'attachment']
):
    
    df = pd.DataFrame(results)
    if index is not None:
        df = df.set_index(index)
    cols2drop = [col for col in drop_columns if col in df]
    df = df.drop(columns=cols2drop)
    return df

def fetch_labels(dev, prels={}): 
    labels = []
    for i in range(dev.shape[0]):
        di = dev.iloc[i]
        # Expand the choice that corresponds to p1 
        # and the response (letter of the choice)
        
        #p1_predrel = [None]*dev.shape[0]
    
        c = di["response"]
        if len(c)!=1:
            c = c[1]
        cd = ord(c)-ord('A')
        if cd<0:
            raise ValueError(f"Invalid choice '{c}'.")
        if di["class"]=="p1rel":
            pp = di["P1"]
        elif di["class"]=="p2rel":
            pp = di["P2"]
        plist = prels[pp]
        if type(plist)!=list:
            raise TypeError(f"prels[{pp}] is not a list.")
        if cd>=len(plist):
            raise KeyError(f"r={c};class={di['class']};pp={pp}")
        try:
            csel = prels[pp][cd]
        except IndexError:
            raise ValueError(f"r={c};class={di['class']};pp={pp};cd={cd}")
        if type(csel)==dict:
            if not csel["choice"]==c:
                raise KeyError
            label = csel["label"]
        else:
            label = csel
        labels.append(label)
    return labels


def clean_predrels(dev):
    if "predrel" not in dev.columns:
        return dev
    dev.loc[dev["predrel"]=="Topic : Attribute", "predrel"] = "Attribute"
    dev.loc[dev["predrel"]=="Topic : Setting", "predrel"] = "Setting"
    dev.loc[dev["predrel"]=="The setting at which {X} may be situated. ", "predrel"] = "Setting"
    dev.loc[dev["predrel"]=="Occupant : Vessel", "predrel"] = "Vessel"
    dev.loc[dev["predrel"]=="Agent : PhysicalSupport", "predrel"] = "Physical Support"
    dev.loc[dev["predrel"]=="Topic : Landmark", "predrel"] = "Landmark"
    dev.loc[dev["predrel"]=="Agent : Instrument", "predrel"] = "Instrument"
    dev.loc[dev["predrel"]=="Display : Medium", "predrel"] = "Medium"
    dev.loc[dev["predrel"]=="Activity : Participant", "predrel"] = "Participant"
    dev.loc[dev["predrel"]=="Topic : Spatial Arrangment", "predrel"] = "Spatial"
    dev.loc[dev["predrel"]=="A temporary condition or time of day", "predrel"] = "Temporal"
    dev.loc[dev["predrel"]=="Topic : Manner", "predrel"] = "Manner"
    dev.loc[dev["predrel"]=="Person : Clothing", "predrel"] = "Clothing"
    dev.loc[dev["predrel"]=="Topic : Source", "predrel"] = "Source"
    dev.loc[dev["predrel"]=="A special kind of spatial arrangement between {X} and {Y}. ", "predrel"] = "Spatial"
    dev.loc[dev["predrel"]=="Activity : Co-Participants", "predrel"] = "Co-Participants"
    dev.loc[dev["predrel"]=="An activity at which {X} may be engaged. ", "predrel"] = "Activity"    
    return dev

def load_results(
    f, 
    drop_columns=['sentence_text', 'X', 'P1', 'Y', 'P2', 'Z', 'attachment'], 
    index="annidx"
):
    examples=[]
    with open(f) as jsonl:
        for line in jsonl:
            example = json.loads(line.strip())
            examples.append(example)
    adf = results_to_dataframe(
        examples, 
        drop_columns=drop_columns, 
        index=index
    )
    return adf

def fill_choices(adf, response_mapper={}):
    adf["predrel"] = fetch_labels(adf, prels=response_mapper)
    adf = clean_predrels(adf)
    return adf

def collate_results(df):
    # Make generic - loop through classes
    if "class" not in df:
        return df
    classes = set(df["class"].values.tolist())
    if len(classes)==1:
        return df
    
    clsdfs = []
    for c in classes: 
        clsdf = df.loc[df['class']==c]
        clsdfs.append({"class": c, "data": clsdf})
    df = clsdfs[0]["data"].drop(columns=["class"])
    c  = clsdfs[0]["class"]
    clsdf = clsdfs[1]["data"]
    c2 = clsdfs[1]["class"]
    df = df.join(clsdf.drop(columns=["class"]), lsuffix=f"_{c}", rsuffix=f"_{c2}", how="inner")
    for i in range(2,len(clsdfs)):
        clsd = clsdfs[i]
        clsdf = clsd["data"]
        c    = clsd["class"]
        df = df.join(clsdf.drop(columns=["class"]), rsuffix=f"_{c}", how="inner")

    return df

def load_relations(f):
    prels   = json.load(open("../pp-ambiguity/data/preprels.json"))
    # Derive the choices from enumeration ('A', 'B', 'C', ...)
    # for each preposition in our inventory. 
    for prep,rels in prels.items():
        #print(f"prels[{prep}]  = {rels}")
        c = 'A'
        newrels = []
        for rel in rels:
            if type(rel)==str:
                newrel = {"label": rel}
                newrels.append(newrel)
                rel = newrel
                
            rel["choice"] = c
            c = chr(ord(c)+1)
    return prels

def plot_results(results, title="Information Structure vs Plausibility", 
    xcol="plausibility", ycol="structure",
    xlabel="Plausibility X", ylabel="Info Y",
    model="", prompter="", token="", boundary='default'):
    df = results_to_dataframe(results)
    fig01 = plot_dataframe(df, title=title, xcol=xcol, ycol=ycol, xlabel=xlabel, ylabel=ylabel, 
        model=model, prompter=prompter, token=token, boundary=boundary)
    return fig01

def plot_dataframe(df, title="Information Structure vs Plausibility", 
    xcol="plausibility", ycol="structure",
    xlabel="Plausibility X", ylabel="Info Y",
    model="", prompter="", token="", boundary='default'):
    default_boundary = 1
    if model=="vera":
        if xcol=="log_neg_log_ypz_over_xpz":
            token="log"
            model_name = "VERA:log"
            default_boundary = 0
        else:
            model_name = "VERA"
    elif model=="llama3":
        model_name = f"Llama 3.1 8B" # ":{token}"
    else:
        model_name = model
    if token is None:
        token=""
    else:
        token=f"-{token}"
    if prompter is not None and len(prompter)>0:
        prompter = f":{prompter}"
    fig01 = px.scatter(df, x=xcol, y=ycol, 
                       color="attachment", 
                       symbol="attachment",
                       symbol_map={"X": "x", "Y": "triangle-down"},                    
                       title=f"{title} ({model_name})", #{prompter})",
                       labels={xcol: xlabel, 
                               ycol: ylabel, 
                               "attachment": "Attachment"},
                       hover_data=['sentence_text',
                                 'prompt_xpz', 'prompt_ypz',
                                 'prompt_xpy', 'prompt_xpypz',
                                 'neg_log_xpz', 'neg_log_ypz',
                                 'neg_log_xpy', 'neg_log_xpypz',
                                 'attachment'])
    if df[xcol].min(axis=0)<0:
        hxmin = min(int(df[xcol].min(axis=0))-1, 0)
    else:
        hxmin = min(int(df[xcol].min(axis=0)), 0)
    hxmax = int(df[xcol].max(axis=0))+1
    if df[ycol].min(axis=0)<0:
        vymin = min(int(df[ycol].min(axis=0))-1, 0)
    else:
        vymin = min(int(df[ycol].min(axis=0)), 0)
    vymax = int(df[ycol].max(axis=0))+1
    if boundary in ['default', 'all']:
        fig01.add_shape(type="line", x0=default_boundary, y0=vymin, x1=default_boundary, y1=vymax, 
            line=dict(color="black", dash="dash", width=1))
        fig01.add_shape(type="line", x0=hxmin, y0=default_boundary, x1=hxmax, y1=default_boundary, 
            line=dict(color="black", dash="dash", width=1))
    if boundary in ['svc', 'all']:
        eval_dict = eval_results(df, plaus_col=xcol, struct_col=ycol)
        m = eval_dict['combo']['m']
        b = eval_dict['combo']['b']

        mp = eval_dict[xcol]['m']
        bp = eval_dict[xcol]['b']

        ms = eval_dict[ycol]['m']
        bs = eval_dict[ycol]['b']

        x_int = -bp/mp
        y_int = -bs/ms

        x0 = max(hxmin, (vymin-b)/m)
        x1 = min(hxmax, (vymax-b)/m)
        fig01.add_shape(type="line", x0=x0, y0=m*x0+b, x1=x1, y1=m*x1+b, line=dict(color="red", dash="dash", width=1))
        fig01.add_shape(type="line", x0=x_int, y0=vymin, x1=x_int, y1=vymax, line=dict(color="purple", dash="dash", width=1))
        fig01.add_shape(type="line", x0=hxmin, y0=y_int, x1=hxmax, y1=y_int, line=dict(color="purple", dash="dash", width=1))
    return fig01

def compute_metrics(df):
    df['structure'] = np.log(df['response_xpy']/df['response_xpypz'])
    df['plausibility'] = np.log(df['response_xpz']/df['response_ypz'])
    df['neg_log_xpy'] = -np.log(df['response_xpy'].values)
    df['neg_log_xpypz'] = -np.log(df['response_xpypz'].values)
    df['neg_log_xpz'] = -np.log(df['response_xpz'].values)
    df['neg_log_ypz'] = -np.log(df['response_ypz'].values)
    df['neg_log_ypz_over_xpz'] = df['neg_log_ypz'] / df['neg_log_xpz']
    df['neg_log_xpy_over_xpypz'] = df['neg_log_xpy'] / df['neg_log_xpypz']    
    df['log_neg_log_ypz_over_xpz'] = np.log(df['neg_log_ypz_over_xpz'])
    df['log_neg_log_xpy_over_xpypz'] = np.log(df['neg_log_xpy_over_xpypz'])
    return df

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("resultsfile", type=str, help="results jsonlines file")
    parser.add_argument("--plaus_col", type=str, default="neg_log_ypz_over_xpz", required=False)
    parser.add_argument("--struct_col", type=str, default="neg_log_xpy_over_xpypz", required=False)
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser

def cross_validation(results, plaus_col='plausibility', struct_col='structure', folds=5):
    X = results[[plaus_col, struct_col]].values
    y = results.attachment

    X_plaus = X[:,:1]
    X_info = X[:,1:]

    N = X.shape[0]

def compute_accuracy(X_train, X_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), svm.LinearSVC(dual=True))
    clf.fit(X_train, y_train)

    if X_train.shape[1]==2:
        scaler = clf.steps[0][1]
        model = clf.steps[1][1]

        w = model.coef_[0] / scaler.scale_
        b = model.intercept_[0] - np.dot(scaler.mean_, w)

        m = -w[0]/w[1]
        b = -b / w[1]
    elif X_train.shape[1]==1:
        w = clf.steps[1][1].coef_[0] / clf.steps[0][1].scale_
        b = clf.steps[1][1].intercept_[0] - np.dot(clf.steps[0][1].mean_, w)
        m = -w[0]
        b = -b

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return preds, acc, m, b

def eval_results(results, plaus_col='plausibility', struct_col='structure', y_col='attachment', folds=5):
    X = results[[plaus_col, struct_col]].values
    y = results[y_col].values

    X_plaus = X[:,:1]
    X_info = X[:,1:]

    kf = KFold(n_splits=folds, shuffle=False) #, random_state=917716873)

    split_combo = []
    split_info  = []
    split_plaus = []
    clfkeys = ['preds', 'accuracy_score', 'm', 'b']

    y_test_cv = ['N']*X.shape[0]
    lastj = 0
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        for j in range(len(test_index)):
            y_test_cv[lastj+j] = y_test[j]
        lastj += len(test_index)
        split_combo.append(dict(zip(clfkeys, compute_accuracy(X_train, X_test, y_train, y_test))))
        split_info.append(dict(zip(clfkeys, compute_accuracy(X_info[train_index], X_info[test_index], y_train, y_test))))
        split_plaus.append(dict(zip(clfkeys, compute_accuracy(X_plaus[train_index], X_plaus[test_index], y_train, y_test))))

    ppreds = []
    for splitres in split_plaus:
        ppreds += splitres['preds'].tolist()
    acc_plaus = np.mean([splitres['accuracy_score'] for splitres in split_plaus])
    mp = np.mean([splitres['m'] for splitres in split_plaus])
    bp = np.mean([splitres['b'] for splitres in split_plaus])

    print(f"Accuracy (plausibility): {acc_plaus*100.0:.3f}")
    spreds = []
    for splitres in split_info:
        spreds += splitres['preds'].tolist()
    acc_info = np.mean([splitres['accuracy_score'] for splitres in split_info])
    ms = np.mean([splitres['m'] for splitres in split_info])
    bs = np.mean([splitres['b'] for splitres in split_info])

    print(f"Accuracy (structure): {acc_info*100.0:.3f}")
    preds = []
    for splitres in split_combo:
        preds += splitres['preds'].tolist()
    acc_combo = np.mean([splitres['accuracy_score'] for splitres in split_combo])
    alt_acc_combo = accuracy_score(y_test_cv, preds)
    m = np.mean([splitres['m'] for splitres in split_combo])
    b = np.mean([splitres['b'] for splitres in split_combo])

    print(f"Accuracy (combo): {acc_combo*100.0:.3f} alt={alt_acc_combo*100.0:.3f}")
    retval = {}
    retval[plaus_col] = {'preds': ppreds, 'accuracy_score': acc_plaus, 'm': mp, 'b': bp}
    retval[struct_col] = dict(zip(clfkeys, [spreds, acc_info, ms, bs]))
    retval['combo'] = dict(zip(clfkeys, [preds, acc_combo, m, b]))
    return retval

def eval_results_nofolds(results, plaus_col='plausibility', struct_col='structure'):
    X = results[[plaus_col, struct_col]].values
    y = results.attachment

    X_plaus = X[:,:1]
    X_info = X[:,1:]

    clfs = make_pipeline(StandardScaler(), svm.LinearSVC(dual=True))
    clfs.fit(X_info, y)
    preds = clfs.predict(X_info)
    acc_info = accuracy_score(y, preds)

    clfp = make_pipeline(StandardScaler(), svm.LinearSVC(dual=True))
    clfp.fit(X_plaus, y)
    preds = clfp.predict(X_plaus)
    acc_plaus = accuracy_score(y, preds)

    clf = make_pipeline(StandardScaler(), svm.LinearSVC(dual=True))
    clf.fit(X, y)
    preds = clf.predict(X)
    acc_combo = accuracy_score(y, preds)

    print(f"Structure: {acc_info*100.0:.3f}%")
    print(f"Plausibility: {acc_plaus*100.0:.3f}%")
    print(f"Combined: {acc_combo*100.0:.3f}%")

    return clf, clfp, clfs

def get_bucket(df, criteria={}):
    ''' 
    Given a dict of name/value pairs, return a slice of the DataFrame
    matching the criteria given. Each name must match the name of a column. 
    If the value is a list, `df[name].isin(val)` is used, otherwise `==`. 
    ''' 
    filters = []
    for criterion in criteria:
        if criterion=="*":
            continue
        val = criteria[criterion]
        if type(val)==list:
            filters.append(df[criterion].isin(val))    
        else:
            filters.append(df[criterion]==val)
    npfilters = np.array(filters)
    return df.loc[npfilters.all(axis=0)]

def get_proportion(df, criteria={}, subcrit={}):
    stuff = get_bucket(df, criteria=criteria).shape[0]
    criteria.update(subcrit)
    stuff_sub = get_bucket(df, criteria=criteria).shape[0]
    return stuff_sub/stuff

def load_results_pprel(f):
    examples=[]
    with open(f) as jsonl:
        for line in jsonl:
            example = json.loads(line.strip())
            examples.append(example)
    adf = results_to_df_pprel(examples) 
    return adf

def results_to_df_pprel(res):
    df = pd.DataFrame(res)
    # Make generic - loop through classes
    p1rel = df.loc[df['class']=='p1rel'].set_index('annidx')
    
    extra_columns=['sentence_text', 'X', 'P1', 'Y', 'P2', 'Z', 'attachment']
    drop_columns = [col for col in extra_columns if col in df]
    p2rel = df.loc[df['class']=='p2rel'].drop(columns=drop_columns).set_index('annidx')
    df = p1rel.join(p2rel, lsuffix='_p1rel', rsuffix='_p2rel', how='inner')
    return df

def compute_acc(df, col1="p1_relation", col2="predrel_p1rel", label="overall"):
    if df.shape[0]==0:
        return 0, 0
    acc1 = accuracy_score(df[col1], df[col2])
    n1 = df.shape[0]
    return acc1, n1

def summarize_results(df, prep=None):
    if prep is None:
        acc1, n1 = compute_acc(df, col1="p1_relation", col2="predrel_p1rel")
        acc2, n2 = compute_acc(df, col1="p2_relation", col2="predrel_p2rel")
        label = "overall"
    else:        
        acc1, n1 = compute_acc(df.loc[df["P1"]==prep], col1="p1_relation", col2="predrel_p1rel")
        acc2, n2 = compute_acc(df.loc[df["P2"]==prep], col1="p2_relation", col2="predrel_p2rel")
        label = prep
    if (n1+n2)==0:
        return
    oacc = ((n1*acc1)+(n2*acc2))/(n1+n2)
    print(f"{label}: acc={oacc*100.0:.1f}% (N={n1+n2}); acc={acc1*100.0:.2f}% (P1;N={n1}); acc={acc2*100.0:.2f}% (P2;N={n2}); ")
def main():

    parser = init_parser()
    args = parser.parse_args()
    results = load_results(args.resultsfile)
    eval_results(results, plaus_col=args.plaus_col, struct_col=args.struct_col)

if __name__ == "__main__":
    main()
