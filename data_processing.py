# -*- coding: utf-8 -*-
"""
@author: u/Ambiwlans
@general: SmarterAutoMod learns from old comment data/mod actions and can automatically
    report/remove comments once trained.
@description: Stage 2. Processes comments into features to train on.
@credit: r/SpaceX mod team - Thanks to everyone who let me slack on mod duties while I made this
    u/CAM-Gerlach - Tireless consultant, ML wizard
"""
#TODO1 - BUG - Emotes not being captured?
#TODO1 - tidy main, proc_one_co
    # rem unneeded virtcsv stuff
#TODO1 - ask first when saving over existing file.

#TODO2 - REGEX for subreddit/username mentions, redditstyle shortlinks
    # fix "Unable to find domain for: [important stuff](/r/spacex/comments/b29md0/elon_musk_on_twitter_testing_starship_heatshield/eir9pjz/)"
#TODO2 - Allow/use custom regex matches. One-hot them.
#TODO2 - Add verbose flag for debugging. Show comments at each stage of processing.
#TODO2 - More Features
    # percentages (like %caps)
    # is parent of parent comment me? (convo detection)
    # is parent comments a question?
    # more detailed caps info (1st char cap, all caps words)
    # try frequency values instead of raw counts of ft_words, or do both?
        # Highly correllated data is OKish in RFs, but can introduce a bias
#TODO2 - reorganize the stage2 virtual csv thing to save on memory (only important for like 1m + comments). Cur system is ugly
#TODO3 - trim the counters every 25k comments for a speed boost?
#TODO3 - clip the index columns rather than saving them?
#TODO3 - switch feature word selection to use a RF or elastic net/lasso rather than simple stats.... 
        # could also include some words by raw rate (rather than the difference in rates) then cut down in learn. Or a blend
#TODO3 - squeeze everything into uint8s? No point if it works fast enough already
#TODO3 - Risk of bobbytables abuses (say _domain_twittercom in a comment to confuse bot) is pretty minor
#TODO3 - Implement Automod as a feature (useless for subs without an advanced automod, costly, lots of code)
    
###############################################################################
### Imports, Defines, Globals
###############################################################################

#General
import sys
import re
import time

#Math/datastructs
import pandas as pd
from collections import Counter

#Natural Language Toolkit
import nltk

#Create and use a virtual csv because adding rows to dataframes is insanely slow (~5000x as slow at best)
from io import StringIO
from csv import writer 
virtcsv = StringIO()
csv_writer = writer(virtcsv)

#Config
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
num_ft_words = int(config['Processing']['num_ft_words'])      #how many feature words to collect

#Globals for finding our set of word features (ft_words)
cnt_pos_words = Counter({})     #counter for all the words in the positive cases (bad/removed comments)
cnt_neg_words = Counter({})     #counter for all the words in the negative cases (good/allowed comments)
cnt_co_words = Counter()        #current comment's feature words

#Regex borrowed from NLTK.casual Shown here for convenience
URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""
    
###############################################################################
### Load (the raw data)
###############################################################################

        
def load_data():
    try:        
        rawdf = pd.read_csv("rawdf.csv.gz", keep_default_na=False, compression='gzip').drop(columns=['Unnamed: 0'])
        print("Loaded raw comments.")
        return rawdf
    except:
        print("Failed to load csv.")
        sys.exit(1)
    

###############################################################################
### Helper Functions
###############################################################################

# Takes raw tokens and will bin certain features to improve learning rates        
def token_processing(tokens):
    
    #TODO1 can we use more efficient regex? (incl start char, use match rather than search) ... though this is a small part of compute time
    for tk in tokens:
        #convert links
        if re.search(URLS,tk):
            #bust out the domain and add it as another token
            try:
                tokens.append('_d_' + re.search(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]",tk).group().replace('.',''))
            except AttributeError:
                print("Unable to find domain for: " + tk)            
            if re.search(r'.gifv?',tk): 
                tk = '_giflink'
            elif re.search(r'.gif|.png|.jpg|.mp4|imgur',tk): 
                tk = '_imagelink'
            #add a rawlink feature as well to help learning rates
            tokens.append('_rawlink')
        #convert times (very liberal)
        if re.search(r'\d\:\d\d',tk):
            tk = '_time'
        #convert dates (very liberal)
        if re.search(r'\d(/|-)\d\d',tk):
            tk = '_date'
        #convert numbers > 1 digit
        if re.search(r'^\d\d',tk):
            tk = '_number'
        #convert emotes
        if re.search(EMOTICONS,tk):
            tk = '_emoticon'
        #convert quotation marks
        if re.search(r"""^('|"|`|â€)""",tk):
            tk = '_quote'
        #convert commas (just to avoid potential bugs with csvs)
        if tk == ",":
            tk = '_comma'
        #convert other special chars
        if re.search(r'^(\(|\)|\[|\]|{|})',tk):
            tk = '_bracket'
        if re.search(r'^(\\|\||/)',tk):
            tk = '_slashpipe'
        
        #TODO2 - stemming? 
        
        ##TODO3 - collapse ">" quotes ? Probably not worth it
    return tokens
            

# Convert "[a link](http://blah.ca)" -> "a link _redditlink _domain_blah.ca"
def format_redditlinks(text):
    try:
        ret = re.search(r"(?<=\[).+?(?=\])",text.group()).group()
        url = re.search(r"(?<=\]\().+?(?=\))",text.group()).group()
    except AttributeError:
        ret = ""
    ret+=" _redditlink "
    if re.search(r'\.gifv?',text.group()): 
        ret = ret + "_giflink"
    elif re.search(r'\.gif|\.png|\.jpg|\.mp4|imgur',text.group()): 
        ret = ret + '_imagelink'
    #add the domain as another token
    try:
        ret += ' _d_' + re.search(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]",url).group().replace('.','')
    except (AttributeError, UnboundLocalError):
        print("Unable to find domain for: " + text.group())
    return ret


###############################################################################
### STAGE 1
###############################################################################
# Do some comment formating/processing, find rates of appearance of each token and create our ft word list

def stage_1(rawdf):
    #setup
    global cnt_pos_words, cnt_neg_words
    csvheader = 0
    t1000co = time.time()
    for i in range(len(rawdf)):
        #DEV - clock ourselves every 1000 comments
        if i % 1000 is 0 and i != 0:
            print("------------------------------------")
            print("Handling comment #" + str(i) + "   Stage: 1")
            print("Elapsed Time: " + str((time.time() - tstart)/60) + "m")
            print("Time (last 1000 comments): " + str((time.time() - t1000co)))
            t1000co = time.time()
        # load up the row we're working on into ppco[x]
        ppco = pd.Series({"comment":rawdf.iloc[i]["comment"],
            "tokens":[],
            "__rem":rawdf.iloc[i]["removed"], 
            "__score":rawdf.iloc[i]["score"],
            "__age":(rawdf.iloc[i]["time"] - rawdf.iloc[i]["subm_approved_at_utc"]),
            "__num_words":0,
            "__num_uniques":0,
            "__num_chars":0,
            "__pct_caps":0,
            "__max_cons_reps":0,
            "__max_cons_rep_ch":0,  #int from ord()
            "__longest_word":1,
            "__author_karma":rawdf.iloc[i]["author_karma"],
            "__author_age":rawdf.iloc[i]["author_age"],
            "__top_lev":rawdf.iloc[i]["top_lev"],
            "__p_removed":rawdf.iloc[i]["p_removed"],
            "__p_score":rawdf.iloc[i]["p_score"],
            "__subm_score":rawdf.iloc[i]["subm_score"],
            "__subm_num_co":rawdf.iloc[i]["subm_num_co"]
            })
    
        #######################################################################
        ### Formatting
        #######################################################################
        
        tformat = time.time()
        
        #count caps and lower them
        ppco['__num_caps'] = sum(1 for ch in ppco['comment'] if ch.isupper())
        ppco['comment'] = ppco['comment'].lower()
        
        #count characters
        ppco['__num_chars'] = len(ppco['comment'])
    
        ppco['__pct_caps'] = 0
        if ppco['__num_chars'] is not 0:
            ppco['__pct_caps'] = ppco['__num_caps']/ppco['__num_chars']
        
        
        #find/count most consecutively repeated character
        cur_cons_reps = 0
        cur_cons_rep_ch = ' '
        for ch in ppco['comment']:
            #ignore whitespace 
            if re.match('\s',ch):
                continue
            if ch == cur_cons_rep_ch:
                cur_cons_reps = cur_cons_reps + 1
            else:
                cur_cons_reps = 1
                cur_cons_rep_ch = ch
            if cur_cons_reps > ppco['__max_cons_reps']:
                ppco['__max_cons_reps']  = cur_cons_reps
                ppco['__max_cons_rep_ch'] = ord(ch)
        
    
        #TODO2 - do all custom regex matches
        
        #Convert newlines
        ppco['comment'] = re.sub(r'\n|\r|\r\n|\&#x200B;|\u200b',' _newline ',ppco['comment'])
        
        #Catch some reddit special formatting
        ppco['comment'] = ppco['comment'].replace('~~', " _strikethrough ")
        ppco['comment'] = ppco['comment'].replace('**', " _bold ")
        ppco['comment'] = ppco['comment'].replace('*', " _italic ")
        
        ppco['comment'] = re.sub(r'\[.*?\]\(.*?\)',format_redditlinks,ppco['comment'])        
            
        
        #Process comment as tokens
        ppco['tokens'] = nltk.tokenize.casual_tokenize(ppco['comment'])
        ppco['tokens'] = token_processing(ppco['tokens'])
        
        #Count tokens
        ppco['__num_words'] = len(ppco['tokens'])
        ##for tk in range(ppco['__num_words']):
          ##  ppco['tokens'][tk] = token_processing(ppco,ppco['tokens'][tk])
        
        try:
            ppco['__longest_word'] = len(max(ppco['tokens'], key=len))
        except:
            ppco['__longest_word'] = 0
            
          
        ## Debugging printout
        ''' Print off a bunch of comments to see them processed ... 
        if i < 500:
            print("Raw comment:")
            print(rawdf.iloc[i]["comment"])
            print('')
            print("Partly processed:")
            print(ppco['tokens'])
            print('')
            print('----')
            print('')
        #else: sys.exit()    #force an early exit during debugging
        #'''
        if i % 1000 is 0 and i != 0: print("Time to format last comment: " + str((time.time() - tformat))) 
        
        #######################################################################
        ### Setup to help select feature set
        #######################################################################
    
        tsum = time.time() ##Time (to sum our word counters)
        #convert freq counts of each token as int value
        cnt_co = Counter(nltk.FreqDist(ppco['tokens']))
        ppco['__num_uniques'] = len(cnt_co)  
    
        ##TODO3 - This step is Annoyingly slow. (>>half the processing time)
        ##Unfortunately it is also hard to make faster. Batches may allow a minor speed up
        # Add the counters from this comment to the total counts
        if ppco["__rem"]:
            cnt_pos_words = cnt_pos_words + cnt_co
        if not ppco["__rem"]:
            cnt_neg_words = cnt_neg_words + cnt_co
            
        if i % 1000 is 0 and i != 0: print("Time to add last comment's words to our counter: " + str((time.time() - tsum))) 
    
        #Write our Stage1 data to the virtual csv
        #write our temp (Stage1) header for our data set (the pp header)
        if not csvheader:
            csvheader = True
            csv_writer.writerow(ppco.index)
        csv_writer.writerow(ppco)


#######################################################################
### Find the feature word/token set
#######################################################################
        
def find_ft_words():
    # Convert raw counts into % rates of total # words used
    t = sum(cnt_pos_words.values())
    for k in cnt_pos_words: cnt_pos_words[k] = cnt_pos_words[k] / t
    t = sum(cnt_neg_words.values())
    for k in cnt_neg_words: cnt_neg_words[k] = cnt_neg_words[k] / t
    
    # Select our feature words.
    # Sort by the raw difference in rates of appearance of words between pos and neg examples
    ft_words = cnt_pos_words.copy()
    ft_words.subtract(cnt_neg_words)
    ft_words = sorted(ft_words.items(), key=lambda k: -abs(k[1]))[0:num_ft_words]
    print('\n----')
    print("Feature Word Set (with weights):")
    print(ft_words)
    return [ k[0] for k in ft_words ]

###############################################################################
### STAGE 2
###############################################################################

def stage_2(ppdf,ft_words):
    csvheader = 0
    t1000co = time.time()
    for i in range(len(ppdf)):
        ppco = ppdf.iloc[i]
        if i % 1000 is 0 and i != 0:
            print("------------------------------------")
            print("Handling comment #" + str(i) + "   Stage: 2")
            print("Elapsed Time: " + str((time.time() - tstart)/60) + "m")
            print("Elapsed Time (Stage 2): " + str((time.time() - tstart2)/60) + "m")
            print("Time (last 1000 comments): " + str((time.time() - t1000co)))
            t1000co = time.time()
         
        #TODO2 - fix this god awful hack before anyone else sees it
        tokens = ppco['tokens'][2:-2].split("', '")
        
        #TODO3 - Redundant? Can put in S1?
        cnt_co = Counter(nltk.FreqDist(tokens))
        
        #Fill our fake csv with final values to be saved
        for k in ft_words: cnt_co_words[k] = cnt_co[k]

        pco = pd.Series({"__rem":ppco["__rem"], 
                                    "__score":ppco["__score"],
                                    "__age":ppco["__age"],
                                    "__num_words":ppco["__num_words"],
                                    "__num_uniques":ppco["__num_uniques"],
                                    "__num_chars":ppco["__num_chars"],
                                    "__pct_caps":ppco["__pct_caps"],
                                    "__max_cons_reps":ppco["__max_cons_reps"],
                                    "__max_cons_rep_ch":ppco["__max_cons_rep_ch"],
                                    "__longest_word":ppco["__longest_word"],
                                    "__author_karma":ppco["__author_karma"],
                                    "__author_age":ppco["__author_age"],
                                    "__top_lev":ppco["__top_lev"],
                                    "__p_removed":ppco["__p_removed"],
                                    "__p_score":ppco["__p_score"],
                                    "__subm_score":ppco["__subm_score"],
                                    "__subm_num_co":ppco["__subm_num_co"]
                                    })
        pco = pco.append(pd.Series(cnt_co_words))
        
        if not csvheader:
            csvheader = True
            csv_writer.writerow(pco.index)
        csv_writer.writerow(pco)

    
###############################################################################
### Finalization/Saving
###############################################################################

def save():
    virtcsv.seek(0)
    savedf = pd.read_csv(virtcsv)
    
    #Save loop (keep trying while file is in use)
    saved = 0
    while saved == 0:
        try:
            savedf.to_csv("pdf.csv.gz", compression='gzip')
            saved = 1
        except PermissionError:
            print("There was an error accessing 'pdf.csv.gz' check that you have permissions and it isn't in use.")
            input("Press Enter to try again...")
  
###############################################################################
### Main
###############################################################################
            
def main():
    global tstart, tstart2
    #Load
    tload = time.time()
    rawdf = load_data()
    print("Loaded csv in " + str((time.time() - tload)) + "seconds")
    print("Processing " + str(len(rawdf))+ "comments...")
    
    #Stage 1
    tstart = time.time()
    print("Warning: This will take ~10m per 10k comments (not totally linear) and cannot be paused. Ensure you have adequate time.")
    stage_1(rawdf)
    
    del rawdf
    
    #Find Features
    tft= time.time()
    ft_words = find_ft_words()    
    print("----")
    print("----")
    print("Time to select the ft word set: " + str((time.time() - tft))) 
    print("----")
    print("Stage 1 Complete.")
    print("----")
    
    #Restart the csv
    virtcsv.seek(0)
    #TODO2 - fix this monstrous waste of memory
        #Could read from this, write directly to drive
        ##pd.read_csv("processeddf.csv", nrows=0)
    ppdf = pd.read_csv(virtcsv)#, nrows=0) #.drop(columns=['Unnamed: 0'])
    virtcsv.seek(0)

    
    tstart2 = time.time()
    print("Stage 2 Starting:")
    stage_2(ppdf,ft_words)    
    
    #Save
    #convert our virtual csv into a dataframe then save as a real csv.
    print('--------')
    print('--------')
    print('Data all processed!')
    print('Saving...')
    tsave = time.time()
    save()
    print("Saved csv in " + str((time.time() - tsave)) + "seconds")
    print("Complete! Ready for machine learning after only " + str((time.time() - tstart)/60) + "minutes!")

###############################################################################
### Process a single comment
###############################################################################
#
# For use by the live classifier
    
def proc_one_co(rawco,ft_words):
    #Handle the virtcsv
    global virtcsv, csv_writer
    virtcsv = StringIO()
    csv_writer = writer(virtcsv)
    
    #Convert to a df
    ##TODO3 - do this the right way (have stage1 accept single raw comments)
    rawdf = pd.DataFrame([rawco])
    
    #Stage 1
    stage_1(rawdf)
    
    #Restart the csv
    virtcsv.seek(0)
    ppdf = pd.read_csv(virtcsv)#, nrows=0) #.drop(columns=['Unnamed: 0'])
    virtcsv.seek(0)
    
    #Stage 2
    stage_2(ppdf,ft_words)    
    
    virtcsv.seek(0)
    pdf = pd.read_csv(virtcsv)
    
    return pdf
    
if __name__ == "__main__":
    main()
    ##print(proc_one_co(rawco,ft_words))
    ##print(proc_one_co(rawco,ft_words))
















    