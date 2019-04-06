# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:03:41 2019

@author: Angelo
@description: Trains our classifier on processedco.csv and saves to classifier.sav
    Call with flags for visualizations/debugging

"""

###############################################################################
### Imports, Defines, Globals
###############################################################################

#General
import time
import datetime
import pickle
import configparser
#import os
#import psutil
#process = psutil.Process(os.getpid())

config = configparser.ConfigParser()
config.read('config.ini')

#Math/datastructs/visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp

#Machine learning tools
from sklearn import model_selection
from sklearn import metrics
#from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_selection import SelectFromModel   #Only needed if I am going to use RF for feature selection

#import dev_vis

###############################################################################
### Main
###############################################################################

#Global settings
#TODO2 - can I be using warm_start = true?

#Optimized Mar 22 - 1000ftwrd
#default_params = {'n_estimators' : 275, 'max_depth' : 50, 'max_features':20, 'min_samples_split':200, 'n_jobs' : -1, 'random_state' : 42 }
#Optimized Mar 25 - 1000ftwrd
default_params = {'n_estimators' : 310, 'max_depth' : 60, 'max_features':18, 'min_samples_split':25, 'n_jobs' : -1, 'random_state' : 42 }
##Optimized Mar 25 - 2000ft
#default_params = {'n_estimators' : 500, 'max_depth' : 110, 'max_features':18, 'min_samples_split':20, 'n_jobs' : -1, 'random_state' : 42 }

#default_params = {'n_estimators' : 25}## fast testing purposes

def main():
    totalruntime = time.time()
    print("")
    print (str(datetime.datetime.now())) 
    print("""
    Overnight CV run (round 2) with 2000ft words. Getting iffy results from our shady classifier.
                
    """)
    print("Defaults: ")
    print(str(default_params))
    print('drop_cols=["__p_removed","__score"]')
    print("---")
    print("")
    
    global thresholds#temp testing
    load_data(drop_cols=["__p_removed","__score","__p_score"]) #drop columns that are somewhat misleading
    
    #WARNING - finding thresholds from this set to test on later will overfit the tested set.
    clf, thresholds = one_run()
    
    # Show important features (could we use more or less feature words?)
    feature_importance(clf,print_num=250)
    
    #print_extreme_comments(clf)
    #print_mistakes(clf, thresholds[1], 250,1,1) #thresholds[x] is threshold @ fpr = x%
    
    #do just 1 CV
    #cv_test('n_estimators', np.unique(np.geomspace(200,500,6,dtype=int)), num_runs=3)
    
    
    '''Overnight test block ... use to vary multiple hyperparams
    cv_test('n_estimators', np.unique(np.geomspace(100,750,6,dtype=int)),updates = 0)
    cv_test('max_depth', np.unique(np.geomspace(20,150,6,dtype=int)), updates = 0)
    cv_test('max_features', np.unique(np.geomspace(10,60,6,dtype=int)), updates = 0)
    cv_test('min_samples_split', np.unique(np.geomspace(2,300,6,dtype=int)), updates = 0)
    #'''
    
    
    '''
    clf = RandomForestClassifier(**default_params)
    plot_learning_curves()
    #can we increase the learning rate with more estimators? ... maybe.
    default_params['n_estimators'] = 400
    clf = RandomForestClassifier(**default_params)
    plot_learning_curves()
    #'''
    
    #'''#Todo2 - allow other split types... maybe doesn't matter much.
    #cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    #plot_ROC_Curves(clf, X, Y, cv=cv)
    #'''
    
    # Save the classifier to disk
    #TODO2 switch to cPickle? joblib?
    #TODO2 save the headers too
    ft = pd.read_csv("pdf.csv", nrows=0).drop(columns=['Unnamed: 0'])
    ft_words = ft.drop(columns=ft.filter(like='__'))
    pickle.dump([clf,thresholds,ft_words], open('classifier.sav', 'wb'))
    #config["Execution"]["thresholds"] = thresholds
    #with open('config.ini', 'w') as configfile:
    #    config.write(configfile)
    print("Testing completed in " + str((time.time() - totalruntime)/60) + "minutes")
    return

###############################################################################
### Load/split data
###############################################################################
#
# truncate_len - load a subset of the csv. 0 means no truncation
# force_reload - bool forcing reload
    
def load_data(truncate_len=0,force_reload=0,drop_cols=[],rand_split=0):
    #use globals
    global processeddf; global X; global Y; global X_te; global X_tr; global Y_te; global Y_tr
    
    # Don't reload if db is still in memory from previous run
    # Warning. This is a risky way to use data!!!
    if 'processeddf' not in globals() or force_reload:
        print("Loading data...")
        tt = time.time()
        processeddf = pd.read_csv("pdf.csv")
        print("Loaded in " + str((time.time() - tt)) + "seconds")
    else:
        print("Skipping DB load, already in memory")
    
    # split data, toss index
    X = processeddf.drop(columns=['__rem','Unnamed: 0'])
    Y = processeddf['__rem']
        
    # Testing on a truncated set
    if truncate_len:
        X = X.iloc[:truncate_len]
        Y = Y.iloc[:truncate_len]

    # Testing with score
    if drop_cols:
        X = X.drop(columns=drop_cols)
    
    if rand_split:
        #Random test/train split
        #Risk - may overfit without lots of random splits
            # Imagine a chain of 10 identical stupid comments, 
            # if this is trained on 2 and knows to remove the rest... it has info the real program won't have
        X_tr, X_te, Y_tr, Y_te = model_selection.train_test_split(X,Y, test_size = .2, random_state = 42, stratify = Y)
    else:
        #Test/train split so that the test is the most recent 20% of entries
        #Risk - may overfit to this dataset
        #Benefit - more realistic test ... no chance to capture recent trends/threads
        X_te = X.iloc[0:int(len(X)*0.2)]
        X_tr = X.iloc[int(len(X)*0.2):] 
        Y_te = Y.iloc[0:int(len(Y)*0.2)] 
        Y_tr = Y.iloc[int(len(Y)*0.2):]
        
    print("----")
    print("")
    return
    

###############################################################################
### Trial Run
###############################################################################

def one_run(show_roc=1):
    tt = time.time()
    print("Training our model on " + str(len(X)) + " comments (with " + str(len(X_te.columns)) + " features).")
    
    #God speed magic machine learning algorithm.
    clf = RandomForestClassifier(**default_params)
    
    # :( so sad that building a custom loss function in sklearn seems near impossible. mse will have to do. But the problem isn't symmetrical :/
    #TODO3 - fix this
    clf = clf.fit(X_tr, Y_tr)
    print("Trained in " + str((time.time() - tt)) + "seconds")
    
    ## Show some stats on our model.
    y_pred = clf.predict(X_te)
    y_pred_proba = clf.predict_proba(X_te)
    fpr, tpr, thresholds = metrics.roc_curve(Y_te, y_pred_proba[:, 1])
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    mean_thresholds = interp(mean_fpr, fpr, thresholds)
    
    print("Cur model accuracy: " + str(metrics.accuracy_score(Y_te, y_pred)))
    #print("Cur oobscore: " + str(clf.oob_score_))
    print("Cur model F1 score: " + str(metrics.f1_score(Y_te, y_pred)))
    print("Cur model R2 score: " + str(metrics.r2_score(Y_te, y_pred)))
    print("Cur model Avg Prec score: " + str(metrics.average_precision_score(Y_te, y_pred)))
    print("Cur model ROCAUC score: " + str(metrics.roc_auc_score(Y_te, y_pred)))
    print("Cur model pAUC (max FPR = .1) score: " + str(metrics.roc_auc_score(Y_te, y_pred_proba[:, 1],max_fpr=.1)))
    print("Cur model pAUC (max FPR = .01) score: " + str(metrics.roc_auc_score(Y_te, y_pred_proba[:, 1],max_fpr=.01)))
    print("TPR @ .01 FPR: " + str(mean_tpr[1]))
    print("TPR @ .1 FPR: " + str(mean_tpr[10]))
    
    #Print a basic ROC
    if show_roc:
        plt.figure(figsize=(7,7))
        plt.plot(fpr, tpr, lw=1)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.2)
        plt.plot([0.1, 0.1], [0, 1], color='g', alpha=.2)
        plt.plot([0.01, 0.01], [0, 1], color='black', alpha=.2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('1 off ROC')
        plt.show()
    
    print("----")
    print("")
    return clf, mean_thresholds

###############################################################################
### Crossvalidation testing
###############################################################################

# Tuning hyperparamaters!

#cvhyperparam: n_estimators, max_depth, max_features, min_samples_split
#cvparam_range: a range of values to try. ie: np.unique(np.geomspace(2,500,8,dtype=int))

#Returns: pauccv
def cv_test(cvhyperparam, cvparam_range, graphs=True, updates=True, num_runs=3):
    print("---")
    print("Running CV testing for " + str(cvhyperparam) + "...")
    f1cv = np.zeros([len(cvparam_range)])
    #oobcv = np.zeros([len(cvparam_range)])
    bscv = np.zeros([len(cvparam_range)])
    #r2cv = np.zeros([len(cvparam_range)])
    avpreccv = np.zeros([len(cvparam_range)])
    rocauccv = np.zeros([len(cvparam_range)])
    pauccv = np.zeros([len(cvparam_range)])
    cvtimes = np.zeros([len(cvparam_range)])
    params = default_params.copy()
    for zz in range(1,num_runs+1):
        #TODO1 - should there be a shuffle something?
        #X_tr, X_te, Y_tr, Y_te = model_selection.train_test_split(X,Y, test_size = .2, stratify = Y)
        i = 0
        for testvals in cvparam_range:
            tt = time.time()
            params[cvhyperparam] = testvals
            cvclf = RandomForestClassifier(**params)
            cvclf = cvclf.fit(X_tr, Y_tr)
            y_pred = cvclf.predict(X_te)
            y_pred_proba = cvclf.predict_proba(X_te)
            
            f1cv[i] = f1cv[i] + metrics.f1_score(Y_te, y_pred)
            #oobcv[i] = oobcv[i] + cvclf.oob_score_
            bscv[i] = bscv[i] + (1 - metrics.brier_score_loss(Y_te, y_pred_proba[:, 1]))
            #r2cv[i] = r2cv[i] + metrics.r2_score(Y_te, y_pred)
            avpreccv[i] = avpreccv[i] + metrics.average_precision_score(Y_te, y_pred)
            rocauccv[i] = rocauccv[i] + metrics.roc_auc_score(Y_te, y_pred_proba[:, 1])
            pauccv[i] = pauccv[i] + metrics.roc_auc_score(Y_te, y_pred_proba[:, 1],max_fpr=.1)
            cvtimes[i] = cvtimes[i] + time.time() - tt
            #TODO2 - ram visualization is diabled since the spyder process makes values misleading
            #process.memory_info().rss
            i = i+1
            
        
        if updates or zz == num_runs:
            #print updates each run
            print("Run #" + str(zz) + "/" + str(num_runs))
            print("CV for " + str(cvhyperparam) + ":")
            print(cvparam_range)
            print("F1 scores:        " + str((f1cv / zz)))
            #print("Oob scores:       " + str((oobcv / zz)))
            print("1-BS scores:      " + str((bscv / zz)))      #Using 1-Brier Score to make it easier to graph
            #print("R2 scores:        " + str((r2cv / zz)))
            print("Avg Prec scores:  " + str((avpreccv / zz)))
            print("AUC scores:       " + str((rocauccv / zz)))
            print("pAUC (.1) scores: " + str((pauccv / zz)))
            print("Runtimes:         " + str(cvtimes / zz))
            
            #Plot our CV curves each run
            fig, ax1 = plt.subplots()
            fig.set_figheight(7)
            fig.set_figwidth(10)
            
            ax1.set_title("CV Curves - " + str(params))
            ax1.set_xlabel(str(cvhyperparam))
            ax1.set_ylabel("Score")        
            
            ax1.plot(cvparam_range, f1cv / zz, 'o-', color="red", label="F1")
            #ax1.plot(cvparam_range, oobcv / zz, 'o-', color="black", label="Oob")
            ax1.plot(cvparam_range, bscv / zz, 'o-', color="gray", label="1-BS")
            #ax1.plot(cvparam_range, r2cv / zz, 'o-', color="brown", label="R2")
            ax1.plot(cvparam_range, avpreccv / zz, 'o-', color="green", label="Avg Prec")
            ax1.plot(cvparam_range, rocauccv / zz, 'o-', color="orange", label="AUC")
            ax1.plot(cvparam_range, pauccv / zz, 'o-', color="pink", label="pAUC (.1)")
            ax1.grid()
            ax1.legend(loc="best")
            
            ax2 = ax1.twinx()
            ax2.set_ylabel("Runtime (s)")
            ax2.plot(cvparam_range, cvtimes / zz, color="black")
                    
            fig.tight_layout()
            plt.show()
            
            print("----")
            print("")
    return

#'''

###############################################################################
### Print missed comments
###############################################################################

#clf - a trained classifier
#count - how many sample comments to print
        #TODO2 - allow a 0 value here to print all comments...  or ppl just put a big number, w/e
#fp,fn - boolflags to toggle printing false negs, false pos

def print_mistakes(clf, threshold=.5, count=25,show_fp=1,show_fn=1):
    

    
    y_proba = clf.predict_proba(X_te)
    y_pred = (y_proba[:,1] >= threshold).astype(bool) # set threshold as 0.3
    
    print("Printing FP/FN on the basis of threshold = " + str(threshold))
    
    #Todo1 - only works with unshuffled data! Fix!    
    #Todo1 - possibly related ... this doesn't seem to work after 130 or so. Maybe because I manually fucked with the df though.
    rawdf = pd.read_csv("rawdf.csv", keep_default_na=False)
    fneg = fpos = 0
    
    for i in range(len(Y_te)):
        if Y_te.iloc[i] == 1:
            if y_pred[i] == 0:
                fneg += 1
                if fneg <= count and show_fn: 
                    #Print false negatives (bad comments the algo missed)
                    print("False negative #" + str(fneg) + ":")
                    print("Proba score: " + str(y_proba[i]))
                    print("Score: " + str(rawdf.iloc[i]['score']))
                    print("Author: " + str(rawdf.iloc[i]['author']))
                    print(rawdf.iloc[i]['comment'])
                    print("https://www.reddit.com" + str(rawdf.iloc[i]['permalink']))
                    print("---")

    for i in range(len(Y_te)):
        if Y_te.iloc[i] == 0 and show_fp:
            if y_pred[i] == 1:
                fpos += 1
                if fpos <= count and show_fp:
                    #Print false postives (good comments the algo reported)
                    print("False postive #" + str(fpos) + ":")
                    print("Proba score: " + str(y_proba[i]))
                    print("Score: " + str(rawdf.iloc[i]['score']))
                    print("Author: " + str(rawdf.iloc[i]['author']))
                    print(rawdf.iloc[i]['comment'])
                    print("https://www.reddit.com" + str(rawdf.iloc[i]['permalink']))
                    print("---")
                
    print("Found " + str(fneg) + " false negatives (" +str(int(sum(Y_te)-fneg))+ "true positives). Of " + str(int(sum(Y_te))) + " positives.")         
    print("Found " + str(fpos) + " false positives (" +str(int(len(Y_te) - sum(Y_te)-fpos))+ "true negatives). Of " + str(int(len(Y_te) - sum(Y_te))) + " negatives.")         
    
    return

def print_extreme_comments(clf):
    
    rawdf = pd.read_csv("rawdf.csv", keep_default_na=False)
    
    y_probas = clf.predict_proba(X)[:, 1].tolist()
    
    print("Shittiest comment of the past " + str(len(X)) + "(any) comments:")
    print(rawdf.iloc[y_probas.index(max(y_probas))]['score'])
    print(rawdf.iloc[y_probas.index(max(y_probas))]['author'])
    print(rawdf.iloc[y_probas.index(max(y_probas))]['comment'])
    print("https://www.reddit.com" + str(rawdf.iloc[y_probas.index(max(y_probas))]['permalink']))
    print("----")
    print("")
    print("Most innocent comment of the past " + str(len(X)) + "(any) comments:")
    print(rawdf.iloc[y_probas.index(min(y_probas))]['score'])
    print(rawdf.iloc[y_probas.index(min(y_probas))]['author'])
    print(rawdf.iloc[y_probas.index(min(y_probas))]['comment'])
    print("https://www.reddit.com" + str(rawdf.iloc[y_probas.index(min(y_probas))]['permalink']))
    print("----")
    print("")
    y_probas = clf.predict_proba(X_te)[:, 1].tolist()
    print("Shittiest comment of the past " + str(len(X_te)) + "(test) comments:")
    print(rawdf.iloc[y_probas.index(max(y_probas))]['score'])
    print(rawdf.iloc[y_probas.index(max(y_probas))]['author'])
    print(rawdf.iloc[y_probas.index(max(y_probas))]['comment'])
    print("https://www.reddit.com" + str(rawdf.iloc[y_probas.index(max(y_probas))]['permalink']))
    print("----")
    print("")
    print("Most innocent comment of the past " + str(len(X_te)) + "(test) comments:")
    print(rawdf.iloc[y_probas.index(min(y_probas))]['score'])
    print(rawdf.iloc[y_probas.index(min(y_probas))]['author'])
    print(rawdf.iloc[y_probas.index(min(y_probas))]['comment'])
    print("https://www.reddit.com" + str(rawdf.iloc[y_probas.index(min(y_probas))]['permalink']))
    print("----")
    print("")
    return
###############################################################################
### Learning Curves
###############################################################################
#
#n_jobs - how many cpu cores to use default is 1. -1 uses all.
#n_points - how smooth to make the curve...
    
#TODO1 - show training curves too
#TODO1 - add option for averaging runs
def plot_learning_curves(n_points=5,n_jobs=None):
    
    train_sizes = [int(k) for k in len(X) * np.linspace(0.1, 1.0, n_points)] 
    
    print("---")
    print("Finding Learning Curves...")
    f1 = np.zeros([len(train_sizes)])
    #oob = np.zeros([len(train_sizes)])
    bs = np.zeros([len(train_sizes)])
    #r2 = np.zeros([len(param_range)])
    avprec = np.zeros([len(train_sizes)])
    rocauc = np.zeros([len(train_sizes)])
    pauc = np.zeros([len(train_sizes)])
    runtimes = np.zeros([len(train_sizes)])
    
    clf = RandomForestClassifier(**default_params)
    
    zz= 1 ##temp obviously.
    i = 0
    for train_size in train_sizes:
        ##X_te = X.iloc[:int(train_size*0.2)]
        ##Y_te = Y.iloc[:int(train_size*0.2)]
        ##X_tr = X.iloc[int(train_size*0.2):train_size] 
        ##Y_tr = Y.iloc[int(train_size*0.2):train_size]
        
        X_te = X.iloc[:int(len(X)*0.2)]
        X_tr = X.iloc[int(len(X)*0.2):int((train_size * .8) + (len(X)*0.2))] 
        Y_te = Y.iloc[:int(len(X)*0.2)]
        Y_tr = Y.iloc[int(len(X)*0.2):int((train_size * .8) + (len(X)*0.2))] 
        
        tt = time.time()
        clf = clf.fit(X_tr, Y_tr)
        y_pred = clf.predict(X_te)
        y_pred_proba = clf.predict_proba(X_te)
        
        f1[i] = f1[i] + metrics.f1_score(Y_te, y_pred)
        #oob[i] = oob[i] + clf.oob_score_
        bs[i] = bs[i] + (1 - metrics.brier_score_loss(Y_te, y_pred_proba[:, 1]))
        #r2[i] = r2[i] + metrics.r2_score(Y_te, y_pred)
        avprec[i] = avprec[i] + metrics.average_precision_score(Y_te, y_pred)
        rocauc[i] = rocauc[i] + metrics.roc_auc_score(Y_te, y_pred_proba[:, 1])
        pauc[i] = pauc[i] + metrics.roc_auc_score(Y_te, y_pred_proba[:, 1],max_fpr=.1)
        runtimes[i] = runtimes[i] + time.time() - tt
        i += 1
        
    print("Scores for training sizes:")
    print(train_sizes)
    print("F1 scores:        " + str((f1 / zz)))
    #print("Oob scores:       " + str((oob / zz)))
    print("1-BS scores:      " + str((bs / zz)))      #Using 1-Brier Score to make it easier to graph
    #print("R2 scores:        " + str((r2 / zz)))
    print("Avg Prec scores:  " + str((avprec / zz)))
    print("AUC scores:       " + str((rocauc / zz)))
    print("pAUC (.1) scores: " + str((pauc / zz)))
    print("Runtimes:         " + str(runtimes / zz))
    
    #Plot our  curves each run
    #plt.figure()
    fig, ax1 = plt.subplots()
    fig.set_figheight(7)
    fig.set_figwidth(10)
    
    ax1.set_title("Learning Curves")
    ax1.set_xlabel("#Training examples")
    ax1.set_ylabel("Score")        
    
    ax1.plot(train_sizes, f1 / zz, 'o-', color="red", label="F1")
    #ax1.plot(train_sizes, oob / zz, 'o-', color="black", label="Oob")
    ax1.plot(train_sizes, bs / zz, 'o-', color="gray", label="1-BS")
    #ax1.plot(train_sizes, r2 / zz, 'o-', color="brown", label="R2")
    ax1.plot(train_sizes, avprec / zz, 'o-', color="green", label="Avg Prec")
    ax1.plot(train_sizes, rocauc / zz, 'o-', color="orange", label="AUC")
    ax1.plot(train_sizes, pauc / zz, 'o-', color="pink", label="pAUC (.1)")
    plt.grid()
    plt.legend(loc="best")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Runtime (s)")
    ax2.plot(train_sizes, runtimes / zz, color="black")
            
    fig.tight_layout()
    plt.show()
    
    print("----")    
    return

def plot_ROC_Curves(classifier, X, y, cv=None):

    cv = model_selection.StratifiedKFold(n_splits=6)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(7,7))
    
    i = 0
    for train, test in cv.split(X, y):
        print('')
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ##plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    print("ROC data:")
    print("TPR @ .01 FPR: " + str(mean_tpr[1]))
    print("TPR @ .02 FPR: " + str(mean_tpr[2]))
    print("TPR @ .05 FPR: " + str(mean_tpr[5]))
    print("TPR @ .1 FPR: " + str(mean_tpr[10]))
    print("TPR @ .3 FPR: " + str(mean_tpr[30]))
    print('----')
    print('')
    return

def feature_importance(clf,graph_num=50, print_num=2000):
    
    importances = clf.feature_importances_
    ##std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    
    s = sorted(zip(importances, X.columns), reverse=True)
    svals = [i[0] for i in s]
    snames = [i[1] for i in s]
    
    # Print the feature ranking
    print("Feature ranking:")
    for i in s[:print_num]: 
        print(i)
    
    # Plot the feature importances of the forest
    plt.figure(figsize=(12,7))
    plt.title("Feature importances")
    plt.bar(range(graph_num), svals[:graph_num], color="r", align="center") ##yerr=std[indices][:graph_num]
    plt.xticks(range(graph_num), snames[:graph_num], rotation='vertical')
    plt.margins(0.2)
    plt.xlim([-1, graph_num])
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    print('----')
    print('')    
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
main()