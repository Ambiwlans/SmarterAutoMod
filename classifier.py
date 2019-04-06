# -*- coding: utf-8 -*-
"""
@author: Angelo
@description: 
"""
###############################################################################
### Imports, Defines
###############################################################################

#General
import sys
import re
import time
import configparser
import pickle
from collections import deque

#Math/datastructs/visualizations
import pandas as pd

##
show_negatives = 1
from collections import Counter
coftwords = Counter()

#Machine learning tools
from sklearn.ensemble import RandomForestClassifier

#Other py files
from login import login
from data_processing import proc_one_co

def get_conf(p,thresholds):
    for i in range(len(thresholds))[::-1]:
        if thresholds[i] >= p:
            return (100-i)
        
###############################################################################
### Config/Startup
###############################################################################

starttime = time.time()

#read in settings from config file
config = configparser.ConfigParser()
config.read('config.ini')

#Load data
try:
    [clf,thresholds,ft_words] = pickle.Unpickler(open('classifier.sav','rb')).load()
except:
    print("Failed to load classifier/learner data.")
    sys.exit(1)
    
#Login
try:
    r = login(config['Login']['username'], config['Login']['password'], config['Login']['client_id'],
                     config['Login']['client_secret'], config['Login']['refresh_token'],
                     config['Login']['user_agent'])
except:
    print("Login Crashed")
    sys.exit(1)
if r.user.me() == None:
    print("No login credentials or invalid login.")
    sys.exit(1)
    
###############################################################################
### Main loop
###############################################################################
 
subreddit = r.subreddit(config['General']['subreddit'])

co_count = 0
co_record = deque([], 1000)

for co in subreddit.stream.comments():
    
    # Collect a comment
    p = co.parent()
    author = co.author
    try:
        author_karma = author.comment_karma
        author_age = co.created_utc - author.created_utc
    except:
        author_karma = 0
        author_age = 0    
    
    s = co.submission    
    rawco = pd.Series({'removed':co.removed,
                                     'comment':co.body.encode('ascii', 'ignore').decode(), 
                                     'score':co.score,
                                     'permalink':co.permalink, 
                                     'time':co.created_utc,
                                     'author':author,
                                     'author_karma':author_karma,
                                     'author_age':author_age,
                                     'top_lev':int(co.parent_id[0:2] == 't3'), #'t3' means parent is a submission, not comment
                                     'p_removed':p.removed,
                                     'p_score':p.score,
                                     'subm_fullname':s.name,
                                     'subm_title':s.title.encode('ascii', 'ignore').decode(),
                                     'subm_score':s.score,
                                     'subm_approved_at_utc':s.approved_at_utc,
                                     'subm_num_co':s.num_comments})
    
    #
    #Skip invalid comments
    #
    ignore_message = ""
    #ignore repeats (reddit bug)
    if rawco.permalink in co_record:
        ignore_message = "Comment " + str(rawco.permalink) + "has already been processed."
    else:
        co_record.appendleft(rawco.permalink)
        
    #Ignore submissions
    if(re.search(config['General']['ignore_submission_title'],s.title)):
        ignore_message = ("Ignored by title: \"" + s.title +"\"")
    if(re.search(config['General']['ignore_submission_flair'],str(s.link_flair_text))):
        ignore_message = ("Ignored by flair: \"" + s.title +"\"")
    if s.approved_at_utc is None:
        ignore_message = "Ignoring not yet approved thread"
    #ignore comments
    if(re.search(config['General']['ignore_user'],str(co.author))):
        ignore_message = "Ignored comment by user"
    if(re.search(config['General']['ignore_comment_content'],str(co.body))):
        ignore_message = ("Ignored comment by content: \"" + co.body +"\"")
    if ignore_message != "":
        if config['Debug']['verbose']: print(ignore_message)
        continue
    co_count += 1
    
    #
    # Process comment
    #
    pco = proc_one_co(rawco,ft_words)
    pco = pco.drop(columns=["__rem","__p_removed","__score","__p_score"])
    ##pco = pco.drop('__rem')
    
    #check the pco with our classifier
    y_proba = clf.predict_proba(list(pco.values.reshape(1,-1)))
    
    #Report or removed based on thresholds in config file.
    
    # If bad, handle it. If error, report the comment w/ error
    print("Comment #: " + str(co_count))
    print("Proba score: " + str(y_proba))
    print("Score: " + str(co.score))
    print("Author: " + str(co.author))
    print("https://www.reddit.com" + str(co.permalink))
    print("Body: " + str(co.body))
    if y_proba[0][1] >= thresholds[int(config["Execution"]['remove_fpr'])]:
        print('\x1b[1;31;41m' + 'REMOVED!' + '\x1b[0m')
        print("Confidence level of violation: " + str(get_conf(y_proba[0][1],thresholds)))
        #Do actual report to subreddit
        
        co.report("BeepBoop -  Robot REALLY no like! (Conf:" + str(get_conf(y_proba[0][1],thresholds)) + "%)")
    elif y_proba[0][1] >= thresholds[int(config["Execution"]['report_fpr'])]:
        print('\x1b[2;30;43m' + 'REPORTED!' + '\x1b[0m')
        print("Confidence level of violation: " + str(get_conf(y_proba[0][1],thresholds)) + "%")
        co.report("BeepBoop -  Robot no like! (Conf:" + str(get_conf(y_proba[0][1],thresholds)) + "%)")
    else:
        if show_negatives:
            print('\x1b[1;37;42m' + 'ACCEPTED!' + '\x1b[0m')
            print("Confidence level of violation: " + str(get_conf(y_proba[0][1],thresholds)))
    print("---")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    