# -*- coding: utf-8 -*-
"""
@author: u/Ambiwlans
@general: Autoautomod learns from old comment data/mod actions and can automatically
    report/remove comments once trained.
@description: Stage 1. Data collection. Implemented by classifier.py
@credit: u/CAM-Gerlach - Tireless consultant, ML wizard
    r/SpaceX mod team - let me slack off on modding while I built this
"""

#TODO1 - for 1.0 release (internal)
#TODO2 - for 2.0 release (public)
    # add logging?
    
###############################################################################
### Imports, Defines
###############################################################################

#General
import sys
import time
import configparser
import re

#Math/datastructs/visualizations
import numpy as np
import pandas as pd

#Other py files
from login import login


###############################################################################
### Config/Startup
###############################################################################

starttime = time.time()

#read in settings from config file
config = configparser.ConfigParser()
config.read('config.ini')

print("Comment collector saves every 10th sumbissions and can be stopped safely at any time.")
print("Should collect around 6000 comments/hour ... expect it to take a while.")
#TODO2 - find out if there is a way to lower # of api calls. Checking parent commenter is COSTLY
    #1 call per 100 comments + at least 1 call for every parent comment + 2 to check accnt karmas.
    #could keep track of all comments in a submission to skip c.parent() calls
    #c.author basically seems fucked though.
print("----")


###############################################################################
### Login
###############################################################################

try:
    r = login(config['Login']['username'], config['Login']['password'], config['Login']['client_id'],
                     config['Login']['client_secret'], config['Login']['refresh_token'],
                     config['Login']['user_agent'])
except:
    print("Login Crashed")
    sys.exit(1)
if str(type(r)) != "<class 'praw.reddit.Reddit'>":
    print("Login Failed")
    sys.exit(1)

###############################################################################
### Load old data if set/available
###############################################################################

#TODO2 - add front end updating (adding in recent threads to the top of the df)
#TODO3 - use psraw (pushshift ) or similar to go past the 1000 subm limit /r/pushshift/
    
subreddit = r.subreddit(config['General']['subreddit'])     # read comments from sub in config    

#load and continue from existing csv
if config['General']['use_old_df']:
    try:
        rawdf = pd.read_csv("rawdf.csv", keep_default_na=False)
        rawdf = rawdf.drop(columns=['Unnamed: 0'])
        continue_param = {'after':rawdf.iloc[-1]['subm_fullname']}
        print("Continuing collection from old data file.")
        #attempt to continue from last submission in the csv, if broken, try from the one before
        #required since reddit doesn't update 'next' pointers to submissions that are deleted
        for i in range(20):
            #does the next comment exist?
            if not len([bool(s) for s in subreddit.new(limit=1,params=continue_param)]):
                print("Unable to continue from last submission, trying another (" + str(i+1) +")...")
                rawdf = rawdf.drop(rawdf[rawdf.subm_fullname == rawdf.iloc[-1]['subm_fullname']].index)
                continue_param = {'after':rawdf.iloc[-1]['subm_fullname']}
            else:
                print("Found a good comment to continue from.")
                break
            if i == 19:
                print("Failed continuing 20x in a row. Something is wrong.")
                print("Keep in mind that the current system is limited to 1000 submissions.")
                print("If you are close to that, you are probably at the end of the line.")
                print("Otherwise, check the data file. Or start from scratch.")
                sys.exit(2)
        co_count = len(rawdf)
        s_count = len(np.unique(rawdf['subm_fullname']))
                
        print("Start from scratch by disabling in config.ini or by deleting rawdf.csv")
    except FileNotFoundError:
        rawdf = pd.DataFrame()
        s_count = 0
        co_count = 0
        continue_param = {}
        print("Old data file not found. Creating new one.")
    except:
        print(sys.exc_info()[0])
        sys.exit(2)

print("----")

###############################################################################
### Main Data Collection Loop
###############################################################################

for s in subreddit.new(limit=int(config['General']['num_submissions']),params=continue_param):
    try:
        #Ignore certain submissions
        ignore_message = ""
        if s.num_comments < int(config['General']['ignore_min_co']):
            ignore_message = ("Ignored dead thread: \"" + s.title +"\"")
        if s.num_comments > int(config['General']['ignore_max_co']):
            ignore_message = ("Ignored massive thread: \"" + s.title +"\"")
        if (s.banned_by):   #removed threads
            ignore_message = ("Ignored removed thread: \"" + s.title +"\"")
        if(re.search(config['General']['ignore_submission_title'],s.title)):
            ignore_message = ("Ignored by title: \"" + s.title +"\"")
        if(re.search(config['General']['ignore_submission_flair'],str(s.link_flair_text))):
            ignore_message = ("Ignored by flair: \"" + s.title +"\"")
        if (time.time() - s.approved_at_utc)  < int(config['General']['ignore_recent_submission_age']) * 3600:
            ignore_message = ("Ignored recent: \"" + s.title +"\"")
        
        if ignore_message != "":
            if config['Debug']['verbose']: print(ignore_message)
            continue
            
        s_count = s_count + 1
        print("Working on submission #:" + str(s_count) + "       Comment #:" + str(co_count))
              
        s.comments.replace_more(limit=0)
        comments = s.comments.list()
    except Exception as ex:
        #TODO2 - switch this to a more normal python error handling setup
        if str(type(ex).__name__) == 'BdbQuit':     #This is sadly needed to cancel running code in python. wtf
            sys.exit(0)
        print("ERROR: Failed reading submission:")
        try: print("   " + str(s.permalink))
        except: print("   Corrupt data/connection error.")
        if str(type(ex).__name__) == 'NotFound':
            print("404 Error")
            continue
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        continue
    for c in comments:
        try:
            co_count = co_count + 1
            #Ignore certain comments
            if(re.search(config['General']['ignore_user'],str(c.author))):
                continue
            if s.approved_at_utc is not None:
                if((c.created_utc - s.approved_at_utc) > (int(config['General']['ignore_late_comment_age']) * 3600 )):
                    #print("Ignored late comer comment: \"" + c.body +"\"")
                    continue
            if c.body == "[deleted]" or c.body == "" or c.body == "[removed]":
                #print("Ignored deleted comment: \"" + c.body +"\"")
                continue
            if(re.search(config['General']['ignore_comment_content'],str(c.body))):
                print("Ignored comment by content: \"" + c.body +"\"")
                continue
            
            #TODO2 - check that we got back valid information ... error checking better than the try/catch?
            #TODO2 - using approved_at_utc over created_utc. This may not word for other subs
            
            p = c.parent()
            author = c.author
            if author is None:      #comments from users that have deleted their account
                author_karma = 0
                author_age = 0
            else:
                author_karma = author.comment_karma
                author_age = c.created_utc - author.created_utc
                
            #Add our collected comment to the dataframe
            raw_comment = pd.Series({'removed':c.removed,
                                     'comment':c.body.encode('ascii', 'ignore').decode(), 
                                     'score':c.score,
                                     'permalink':c.permalink, 
                                     'time':c.created_utc,
                                     'author':author,
                                     'author_karma':author_karma,
                                     'author_age':author_age,
                                     'top_lev':int(c.parent_id[0:2] == 't3'), #'t3' means parent is a submission, not comment
                                     'p_removed':p.removed,
                                     'p_score':p.score,
                                     'subm_fullname':s.name,
                                     'subm_title':s.title.encode('ascii', 'ignore').decode(),
                                     'subm_score':s.score,
                                     'subm_approved_at_utc':s.approved_at_utc,
                                     'subm_num_co':len(comments)})
            rawdf = rawdf.append(raw_comment, ignore_index=True)
        except Exception as ex:
            #Unable to read a comment. This happens for reddit-wide shadowbanned users... 
            #and connection issues
            if str(type(ex).__name__) == 'BdbQuit':     #This is sadly needed to cancel running code in python. wtf
                sys.exit(0)
            print("ERROR: Failed reading comment:")
            try: print("   " + str(c.permalink))
            except: print("   Corrupt data/connection error.")
            if str(type(ex).__name__) == 'NotFound':
                print("404 Error or shadowbanned user.")
                continue
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            continue
        
    #save the csv every 10th submission scanned
    if s_count % 10 is 0:
        try:
            #TODO2 - could be just appending to save more efficiently but it probably doesn't matter much.
            rawdf.to_csv("rawdf.csv")
            print("Saved raw data.")
            print("Elapsed Time: " + str(round(((time.time() - starttime)/60),2)) + "m")
        except:
            print("ERROR: Saving to file failed.")
        
        
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    