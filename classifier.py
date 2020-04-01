# -*- coding: utf-8 -*-
"""
@author: u/Ambiwlans
@general: SmarterAutoMod learns from old comment data/mod actions and can automatically
    report/remove comments once trained.
@description: Stage 3. This is the active classifier that performs mod actions on comments as they are made.
@credit: r/SpaceX mod team - Thanks to everyone who let me slack on mod duties while I made this
    u/CAM-Gerlach - Tireless consultant, ML wizard
"""

#TODO2 - switch to more fine tuneable thresholds (99.9%)
#TODO2 - move drop_cols to config

#TODO2 - save the list of checked comments to allow retarts without duplicated effort
#TODO2 - allow data_collection, flagged as 'collected_live'
    # - Also save report/remove/proba status (to allow us to see the real TPR, FPR)
    # - goal is to train on live collected data to enable more use of meta data (parent comment data, num_siblinges, etc.)
#TODO2 - Give removal/report reasons.
    # - 1) show what key words/features appeared in the comment
        # - http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/
    # - 2) Train a second model (unsupervised clustering) to find removal 'classes'
        # - manually label those classes after the fact
#TODO2 - logging to file option
#TODO2 - Collect stats
    # - make stats available on reddit (wiki page) for mods
    # - via pm to bot?
#TODO3 - Improve feedback messages on report/removal
#TODO3 - Re-run on comments after 6 hrs to catch age related features
    
###############################################################################
### Imports, Defines
###############################################################################

#General
import sys
import re
import time
import pickle
from collections import deque

#Settings/Config
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

#Math/datastructs
import pandas as pd
from collections import Counter
coftwords = Counter()

#Machine learning tools
from sklearn.ensemble import RandomForestClassifier

#Praw
from prawcore.exceptions import RequestException

#Other py files
from login import login
from data_processing import proc_one_co
 
###############################################################################
### Helper Functions
###############################################################################

def get_conf(p,thresholds):
    for i in range(len(thresholds))[::-1]:
        if thresholds[i] >= p:
            return (100-i)
        
def reformat_notice(rawco,notice_type,conf):
    #notice_type 0 -> removal notice
    #notice_type 1 -> screening notice
    if notice_type == 0:
        msg = config['Execution']['removal_notice']
    if notice_type == 1:
        msg = config['Execution']['screening_notice']    
    msg = msg.replace("{{author}}",str(rawco.author))
    msg = msg.replace("{{subreddit}}",config['General']['subreddit'])
    msg = msg.replace("{{permalink}}","https://www.reddit.com" + str(rawco.permalink))
    msg = msg.replace("{{title}}",str(rawco.subm_title))
    msg = msg.replace("{{body}}",str(rawco.comment))
    msg = msg.replace("{{conf}}",conf)
    msg = msg.replace("{{nl}}","\n")
    return msg

###############################################################################
### Config/Startup
###############################################################################

tstart = time.time()

#Print our running state before getting to work
print()
print("Smarter AutoMod")
print("---")
print("Subreddit: " + config['General']['subreddit'])
if config['Execution']['classifier_mode'] == "0":
    print("Mode: TEST MODE")
if config['Execution']['classifier_mode'] == "1":
    print("Mode: REPORT ONLY MODE")
if config['Execution']['classifier_mode'] == "2":
    print("Mode: FULL MODE")
print("Report FPR: " + config['Execution']['report_fpr'])
print("Remove FPR: " + config['Execution']['remove_fpr'])
print("Send removal notice (to user): " + config['Execution']['send_removal_notice'])
if config['Execution']['send_removal_notice'] == "True":
    print("Notice format: ")
    print(config['Execution']['removal_notice'])
    print("")
print("Send removal notice (to modteam): " + config['Execution']['send_screening_notice'])
if config['Execution']['send_screening_notice'] == "True":
    print("Notice format: ")
    print(config['Execution']['screening_notice'])
    print("")
print("---")
print("")


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

print("Starting classification...")

###############################################################################
### Main loop
###############################################################################
 
subreddit = r.subreddit(config['General']['subreddit'])

co_count = 0
co_record = deque([], 1000)

#Keep trying every 5 minutes if there are connection issues to avoid downtime.
keeptrying = True
while keeptrying:
    keeptrying = False
    try:
        for co in subreddit.stream.comments():

            
            ###########################################################################
            # Collect a comment
            ###########################################################################
            
            p = co.parent()
            author = co.author
            try:
                author_karma = author.comment_karma
                author_age = co.created_utc - author.created_utc
            except:
                author_karma = 0
                author_age = 0    
            
            s = co.submission    
			
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
                             'subm_time':s_start_time,
                             'subm_num_co':s.num_comments})
            
            ###########################################################################
            #Skip invalid comments
            ###########################################################################
            
            ignore_message = ""
            #ignore repeats (reddit bug)
            if rawco.permalink in co_record:
                ignore_message = "Comment " + str(rawco.permalink) + " has already been processed."
            else:
                co_record.appendleft(rawco.permalink)
                
            #Ignore submissions
            if(re.search(config['General']['ignore_submission_title'],s.title)):
                ignore_message = ("Ignored by title: \"" + s.title +"\"")
            if(re.search(config['General']['ignore_submission_flair'],str(s.link_flair_text))):
                ignore_message = ("Ignored by flair: \"" + s.title +"\"")
            if s.approved_at_utc is None and config['General']['post_start_type'] == "1":
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
            
            ###########################################################################
            # Process comment
            ###########################################################################
            pco = proc_one_co(rawco,ft_words)
            pco = pco.drop(columns=["__rem","__p_removed","__score","__p_score"])
            
            #check the pco with our classifier
            y_proba = clf.predict_proba(list(pco.values.reshape(1,-1)))
            
            ###########################################################################
            # Handle comment, Printouts
            ###########################################################################
            #
            # Report or removed based on thresholds in config file.
            
            conflvl = str(get_conf(y_proba[0][1],thresholds))
            
            #Print comment if it hits a threshold OR if we are printing accepted comments            
            if (config['Execution']['show_accepted'] == "True") or (y_proba[0][1] >= thresholds[int(config["Execution"]['report_fpr'])]):
                print("Elapsed time: " + str(int((time.time() - tstart)/86400)) + "d, " +
                      str(int(((time.time() - tstart) % 86400)/3600)) + "h, " +
                      str(int(((time.time() - tstart) % 3600)/60)) + "m")
                print("Comment #: " + str(co_count))
                print("Probability: " + str(y_proba[0][0]))
                print("Comment Score: " + str(co.score))
                print("Author: " + str(co.author))
                print("https://www.reddit.com" + str(co.permalink))
                print("Body: " + str(co.body))
                
            #Handle Remove, Report, and Accept threshold comments
            if y_proba[0][1] >= thresholds[int(config["Execution"]['remove_fpr'])]:
                # Remove threshold comment!!!
                if (config['Execution']['classifier_mode']) == "0":
                    #TEST MODE - no reddit actions
                    print('\x1b[1;30;41m' + 'REMOVE THRESHOLD!' + '\x1b[0m -- (No action taken - Test Mode)')
                else:
                    if (config['Execution']['classifier_mode']) == "1":
                        #REPORT ONLY MODE - report instead of removing
                        co.report("BeepBoop -  🔥🔥🔥 Robot REALLY no like! 🔥🔥🔥 (Conf:" + conflvl + "%)")
                        print('\x1b[1;30;41m' + 'REMOVE THRESHOLD!' + '\x1b[0m -- (Reported - Report Only Mode)')
                    elif (config['Execution']['classifier_mode']) == "2":
                        #Full Mode - remove bad comment
                        co.mod.remove()
                        print('\x1b[1;31;41m' + 'REMOVED!' + '\x1b[0m')
                        if (config['Execution']['send_removal_notice']) == "True":
                            print("Sending user removal notification...")
                            co.mod.send_removal_message(reformat_notice(rawco,0,conflvl), title='Removal Notification', type='private')
                        if (config['Execution']['send_screening_notice']) == "True":
                            r.redditor(config['Execution']['screening_workaround_user']).message('Removal Notification', 
                                      reformat_notice(rawco,1,str(get_conf(y_proba[0][1],thresholds))),
                                      from_subreddit=config['General']['subreddit'])
                            print("Sending modteam removal notification...")
                print("Confidence level of violation: " + conflvl)        
            elif y_proba[0][1] >= thresholds[int(config["Execution"]['report_fpr'])]:
                # Report threshold comment!
                if (config['Execution']['classifier_mode']) == "0":
                    #TEST MODE - no reddit actions
                    print('\x1b[2;30;43m' + 'REPORT THRESHOLD!' + '\x1b[0m -- (No action taken - Test Mode)')
                elif (config['Execution']['classifier_mode']) in ("1","2"):
                    #FULL OR REPORT ONLY Mode or  - report bad comment
                    co.report("BeepBoop -  Robot no like! (Conf:" + conflvl + "%)")
                    print('\x1b[2;30;43m' + 'REPORTED!' + '\x1b[0m')
                print("Confidence level of violation: " + conflvl + "%")
            else:
                # Acceptable comment. - no actions required
                if (config['Execution']['show_accepted']) == "True":
                    print('\x1b[1;37;42m' + 'ACCEPTED!' + '\x1b[0m')
                    print("Confidence level of violation: " + conflvl)
            print("---")
            
    except RequestException:
        time.sleep(300)
        print("Connection error. Waiting 5 minutes before trying again.")
        keeptrying = True
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    