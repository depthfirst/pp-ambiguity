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

def load_results(f):
    examples=[]
    with open(f) as jsonl:
        for line in jsonl:
            example = json.loads(line.strip())
            examples.append(example)
    adf = results_to_dataframe(examples) 
    return adf

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


def results_to_dataframe(res):
    df = pd.DataFrame(res)
    # Make generic - loop through classes
    xpydf = df.loc[df['class']=='XpY'].set_index('annidx')
    
    drop_columns=['sentence_text', 'X', 'P1', 'Y', 'P2', 'Z', 'attachment']
    xpypzdf = df.loc[df['class']=='XpYpZ'].drop(columns=drop_columns).set_index('annidx')
    xpzdf = df.loc[df['class']=='XpZ'].drop(columns=drop_columns).set_index('annidx')
    ypzdf = df.loc[df['class']=='YpZ'].drop(columns=drop_columns).set_index('annidx')    
    df1 = xpydf.join(xpypzdf, lsuffix='_xpy', rsuffix='_xpypz', how='inner')
    df2 = xpzdf.join(ypzdf, lsuffix='_xpz', rsuffix='_ypz', how='inner')
    df = df1.join(df2, how='inner')
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

def eval_results(results, plaus_col='plausibility', struct_col='structure', folds=5):
    X = results[[plaus_col, struct_col]].values
    y = results.attachment.values

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

def main():

    parser = init_parser()
    args = parser.parse_args()
    results = load_results(args.resultsfile)
    eval_results(results, plaus_col=args.plaus_col, struct_col=args.struct_col)

if __name__ == "__main__":
    main()
