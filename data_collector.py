# -*- coding: utf-8 -*-
"""
@author: u/Ambiwlans
@general: SmarterAutoMod learns from old comment data/mod actions and can automatically
    report/remove comments once trained.
@description: Stage 1. Raw data collection.
@credit: r/SpaceX mod team - Thanks to everyone who let me slack on mod duties while I made this
    u/CAM-Gerlach - Tireless consultant, ML wizard
"""

#TODO1 - add front end updating (adding in recent threads to the top of the df)

#TODO2 - Reconstruct comment tree structures -> allows more meta data on locations of comments ('parent of parent = me' implies a conversation)
#TODO2 - Ignore quarantined threads (not used in r/Spacex)
#TODO2 - Add support for username/pass login + script that will return OAuth token

#TODO3 - use psraw (pushshift ) or similar to go past the 1000 subm limit /r/pushshift/
#TODO3 - logging?
#TODO3 - Add an option to save on quit?
#TODO3 - Could be saving more efficiently but it probably doesn't matter
#TODO3 - Be more efficient with api calls. Checking parent commenter is COSTLY
    #1 call per 100 comments + at least 1 call for every parent comment + 2 to check accnt karmas.
    #could keep track of all comments in a submission to skip c.parent() calls
    #and track all users to avoid calling the same users repeatedly.
        #This would need an 'update users' flag. Or throw out user data between runs.
    
    
###############################################################################
### Imports, Defines
###############################################################################

#General 
import sys
import time
import shutil
import configparser
import re

#Math/datastructs/visualizations
import numpy as np
import pandas as pd

#Other py files
from login import login

from prawcore import BadRequest    #Praw 400 Error Code

###############################################################################
### Config/Startup
###############################################################################

starttime = time.time()

#read in settings from config file
config = configparser.ConfigParser()

if config.read('config.ini') == []:
    try:
        shutil.copyfile('config.ini.default', 'config.ini')
        print("Modify config.ini with your information before continuing")
    except:
        print("Error: No File write access")
        sys.exit(1)
    sys.exit(0)
    
print("Comment collector saves every 10th sumbissions and can be stopped safely at any time.")
print("Should collect around 6000 comments/hour ... expect it to take a while.")

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
if r.user.me() == None:
    print("No login credentials or invalid login.")
    sys.exit(1)
    
###############################################################################
### Load old data if set/available
###############################################################################


subreddit = r.subreddit(config['General']['subreddit'].rsplit('/')[-1])     # read comments from sub in config    

try:
    subreddit.id = subreddit.id
    print("Preparing to collect comments from: " + subreddit.display_name)
except BadRequest:
    print ("BadRequest: received 400 HTTP response")
    print("Malformed subreddit name, check config file.")
except Exception as ex:
    print("Error during subreddit name read: " + str(type(ex).__name__))
    
#load and continue from existing csv
if config['General']['use_old_df'] == "True":
    try:
        rawdf = pd.read_csv("rawdf.csv.gz", compression='gzip', keep_default_na=False)
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
                
        print("Start from scratch by disabling in config.ini or by deleting/renaming rawdf.csv.gz")
    except FileNotFoundError:
        rawdf = pd.DataFrame()
        s_count = 0
        co_count = 0
        continue_param = {}
        print("Old data file not found. Creating new one.")
    except:
        print("Error: " + str(sys.exc_info()[0]))
        sys.exit(2)
else:
    rawdf = pd.DataFrame()
    s_count = 0
    co_count = 0
    continue_param = {}
    print("Creating new DataFile.")
print("----")

###############################################################################
### Main Data Collection Loop
###############################################################################

for s in subreddit.new(limit=int(config['General']['num_submissions']),params=continue_param):
    try:
        #Get the start time from config
        if config['General']['post_start_type'] == "0":
            #Using s.created_utc
            s_start_time = s.created_utc
        elif config['General']['post_start_type'] == "1":
            #Using s.approved_at_utc
            if(s.approved_at_utc != None):
                s_start_time = s.approved_at_utc
            else:
                s_start_time = 0
                ignore_message = ("Ignored unapproved thread: \"" + s.title +"\"")
        else:
            print ("Error: Config file option 'post_start_type' should be 0 or 1.")
            sys.exit(0)
        
        #Ignore certain submissions
        ignore_message = ""
        if s.num_comments < int(config['General']['ignore_min_co']):
            ignore_message = ("Ignored dead thread: \"" + s.title +"\"")
        if s.num_comments > int(config['General']['ignore_max_co']):
            ignore_message = ("Ignored massive thread: \"" + s.title +"\"")
        if (s.banned_by):   #removed threads   ##This doesn't seem to be necessary in this configuration (new won't return removed threads)
            ignore_message = ("Ignored removed thread: \"" + s.title +"\"")
        if(re.search(config['General']['ignore_submission_title'],s.title)):
            ignore_message = ("Ignored by title: \"" + s.title +"\"")
        if(re.search(config['General']['ignore_submission_flair'],str(s.link_flair_text))):
            ignore_message = ("Ignored by flair: \"" + s.title +"\"")
        if (time.time() - s_start_time)  < int(config['General']['ignore_recent_submission_age']) * 3600:
            ignore_message = ("Ignored recent: \"" + s.title +"\"")
        
        #Print ignore message and skip submission
        if ignore_message != "":
            if config['Debug']['verbose'] == "True": print(ignore_message)
            continue
            
        s_count = s_count + 1
        print("Submission #: " + str(s_count) + "\t\tComment #: " + str(co_count) + "\t\tDate: " + str(time.ctime(s_start_time)))
              
        s.comments.replace_more(limit=0)
        comments = s.comments.list()
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
    except Exception as ex:
        print("ERROR: Failed reading submission:")
        try: print("   www.reddit.com" + str(s.permalink))
        except: print("   Corrupt data/connection error.")
        if str(type(ex).__name__) == 'NotFound':
            print("404 Error")
            continue
        print("Error: " + str(type(ex).__name__))
        print("Skipping Submission...")
        continue
    for co in comments:
        try:
            co_count = co_count + 1
            #Ignore certain comments
            if(re.search(config['General']['ignore_user'],str(co.author))):
                continue
            if((co.created_utc - s_start_time) > (int(config['General']['ignore_late_comment_age']) * 3600 )):
                #print("Ignored late comer comment: \"" + c.body +"\"")
                continue
            if co.body == "[deleted]" or co.body == "" or co.body == "[removed]":
                #print("Ignored deleted comment: \"" + c.body +"\"")
                continue
            if(re.search(config['General']['ignore_comment_content'],str(co.body))):
                print("Ignored comment by content: \"" + co.body +"\"")
                continue
            
            #TODO2 - check that we got back valid information ...
            
            p = co.parent()
            author = co.author
            if author is None:      #comments from users that have deleted their account
                author_karma = 0
                author_age = 0
            else:
                author_karma = author.comment_karma
                author_age = co.created_utc - author.created_utc
                
            #Add our collected comment to the dataframe
            raw_comment = pd.Series({'removed':co.removed,
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
                                     'subm_time':s_start_time,
                                     'subm_num_co':len(comments)})
            rawdf = rawdf.append(raw_comment, ignore_index=True)
        except KeyboardInterrupt:
            print("Exiting...")
            sys.exit(0)
        except Exception as ex:
            print("ERROR: Failed reading comment:")
            try: print("   www.reddit.com" + str(co.permalink))
            except: print("   Corrupt data/connection error.")
            if str(type(ex).__name__) == 'NotFound':
                print("404 Error")
                continue
            print("Error: " + str(type(ex).__name__))
            print("Skipping Comment...")
            continue
        
    #save the csv every 10th submission scanned
    if s_count % 10 == 0:
        try:
            rawdf.to_csv("rawdf.csv.gz", compression='gzip')
            print("Saved raw data.")
            print("Elapsed Time: " + str(round(((time.time() - starttime)/60),2)) + "m")
        except:
            print("ERROR: Unable to access 'rawdf.csv.gz'. Check that you have permissions and it isn't in use.")
            input("Press Enter to try again...")
            
#Done checking all submissions
try:
    rawdf.to_csv("rawdf.csv.gz", compression='gzip')
    print("Saved raw data.")
    print("Total Elapsed Time: " + str(round(((time.time() - starttime)/60),2)) + "m")
    print("----")
    print("Run data_processing.py next to process the raw data.")
except:
    print("ERROR: Unable to access 'rawdf.csv.gz'. Check that you have permissions and it isn't in use.")
    input("Press Enter to try again...")    
        
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    