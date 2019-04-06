# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:01:21 2019

@author: Angelo
@decription: Processes rawdf.csv into pdf.csv (our actual features/X,y vals to train on)

"""

###############################################################################
### Imports, Defines, Globals
###############################################################################

#General
import sys
import re
import time
import configparser
from collections import Counter

#Create and use a virtual csv because adding rows to dataframes is insanely slow (~5000x as slow)
from io import StringIO
from csv import writer 
virtcsv = StringIO()
csv_writer = writer(virtcsv)
csvheader = 0

#Math/datastructs/visualizations
import pandas as pd
import nltk

#read in settings from config file
config = configparser.ConfigParser()
config.read('config.ini')

#Globals for finding our set of word features (cntftwords)
num_co = 1                  #the number of comments we're processing
num_ft_words = int(config['Processing']['num_ft_words'])      #how many feature words to collect
cntpos = Counter({})        #counter for all the words in the positive cases (bad/removed comments)
cntneg = Counter({})        #counter for all the words in the negative cases (good/allowed comments)
coftwords = Counter()       #current comment's feature words
cntftwords = Counter()      #a list of the feature words (not actually a list for technical reasons)

#Regex borrowed from NLTK
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
#
#TODO2 - remove from ram when done w/ it
        
def load_data():
    global num_co
    
    try:        
        rawdf = pd.read_csv("rawdf.csv", keep_default_na=False)#[0:1000] ##temp for testing
        rawdf = rawdf.drop(columns=['Unnamed: 0'])
        num_co = len(rawdf)
        return rawdf
    except:
        print("Failed to load csv.")
        sys.exit(1)
    

###############################################################################
### Helper Functions
###############################################################################

# Takes raw tokens and will group certain things to help the learner out
def token_processing(ppco,token_text):
    #TODO1 can we use more efficient regex? (incl start char, use match rather than search)
    #convert links to identifiable token
    if re.search(URLS,token_text):
        if re.search(r'\.gifv?',token_text): 
            token_text = '_giflink'
        elif re.search(r'\.gif|\.png|\.jpg|\.mp4|imgur',token_text): 
            token_text = '_imagelink'
        #add a rawlink feature regardless to help learning rates
        ppco['tokens'].append('_rawlink')
    #TODO2 - fix these
    #convert times (very liberal)
    #if re.search('(\d:\d)|T\-|T\+',token_text):
    #    token_text = '_time'
    #convert dates (very liberal)
    #if re.search('\d(/|-)\d',token_text):
    #    token_text = '_date'
    #convert numbers > 1 digit
    if re.search('\d\d',token_text):
        token_text = '_number'
    ##TODO3 collapse ">" quotes ?
     ##probably not worth it
    #convert emotes
    if re.search(EMOTICONS,token_text):
        token_text = '_emoticon'
    if re.search("^('|\"|`|â€)",token_text):
        token_text = '_apostrophe'
    if token_text == ",":
        token_text = '_comma'
    #convert other special chars
    if re.search(r'\(|\)|\[|\]',token_text):
        token_text = '_bracket'
    if re.search(r'\\|\||/',token_text):
        token_text = '_slashpipe'
    
    #TODO2 - stemming? 
    return token_text
            

# Convert "[a link](http://blah.ca)" -> "a link _redditlink"
def format_redditlinks(text):
    try:
        ret = re.search("(?<=\[).+?(?=\])",text.group()).group()
    except AttributeError:
        ret = ""
    ret+=" _redditlink "
    if re.search(r'\.gifv?',text.group()): 
        ret = ret + "_giflink"
    elif re.search(r'\.gif|\.png|\.jpg|\.mp4|imgur',text.group()): 
        ret = ret + '_imagelink'
    return ret


###############################################################################
### STAGE 1
###############################################################################
# Do some comment formating/processing, find rates of appearance of each token and create our ft word list

def stage_1(rawdf):
    #setup
    global csvheader, cntpos, cntneg
    t1000co = time.time()
    for i in range(num_co):
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
        #TODO1 more detailed caps info (1st char cap, all caps words)
        ppco['comment'] = ppco['comment'].lower()
        
        #count characters
        ppco['__num_chars'] = len(ppco['comment'])
    
        ##TODO1 - are percentages redundant features? If so, add more. If not, remove this one.
        ppco['__pct_caps'] = 0
        if ppco['__num_chars'] is not 0:
            ppco['__pct_caps'] = ppco['__num_caps']/ppco['__num_chars']
        
        
        #find/count most consecutively repeated character
        cur_cons_reps = 0
        cur_cons_rep_ch = ' '
        for ch in ppco['comment']:
            #ignore whitespace 
            ##TODO1 check if this actually works properly
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
        ppco['__num_words'] = len(ppco['tokens'])
        for tk in range(ppco['__num_words']):
            ppco['tokens'][tk] = token_processing(ppco,ppco['tokens'][tk])
        
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
        cntco = Counter(nltk.FreqDist(ppco['tokens']))
        ppco['__num_uniques'] = len(cntco)  
    
        ##TODO3 - This step is Annoyingly slow. (>>half the processing time)
        ##Unfortunately it is also hard to make faster. Batches may allow a minor speed up
        # Add the counters from this comment to the total counts
        if ppco["__rem"]:
            cntpos = cntpos + cntco
        if not ppco["__rem"]:
            cntneg = cntneg + cntco
            
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
##TODO3 - switch feature word selection to use a RF rather than simple stats.... 
        
def find_ft_words():
    # Convert raw counts into % rates of total # words used
    t = sum(cntpos.values())
    for k in cntpos: cntpos[k] = cntpos[k] / t
    t = sum(cntneg.values())
    for k in cntneg: cntneg[k] = cntneg[k] / t
    
    # Select our feature words.
    # Sort by the raw difference in rates of appearance of words between pos and neg examples
    cntftwords = cntpos.copy()
    cntftwords.subtract(cntneg)
    cntftwords = sorted(cntftwords.items(), key=lambda k: -abs(k[1]))[0:num_ft_words]
    return cntftwords

###############################################################################
### STAGE 2
###############################################################################

def stage_2(ppdf):
    global csvheader
    t1000co = time.time()
    for i in range(num_co):
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
        
        ##Redundant? Can put in S1?
        cntco = Counter(nltk.FreqDist(tokens))
        
        #Fill our fake csv with final values to be saved
        for k in cntftwords.keys(): coftwords[k] = cntco[k]
        #TODO1 - try frequency values instead of raw counts
        
        #TODO2 squueze everything into uint8s? (0~255)
        ##TODO2 - Risk of bobbytables abuses is pretty minor but maybe worth fixing
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
        pco = pco.append(pd.Series(coftwords))
        
        if not csvheader:
            csvheader = True
            csv_writer.writerow(pco.index)
        csv_writer.writerow(pco)

    
#
# Finalization/Saving
# 

def save():
    virtcsv.seek(0)
    savedf = pd.read_csv(virtcsv)
    
    saved = 0
    while saved == 0:
        try:
            savedf.to_csv("pdf.csv")
            saved = 1
        except PermissionError:
            print("There was an error accessing 'pdf.csv' check that you have permissions and it isn't in use.")
            input("Press Enter to try again...")
            
def main():
    global tstart, tstart2, csvheader, cntftwords
    #Load
    timeload = time.time()
    rawdf = load_data()
    print("Loaded csv in " + str((time.time() - timeload)) + "seconds")
    print("Processing " + str(len(rawdf))+ "comments...")
    
    #Stage 1
    tstart = time.time()
    print("Warning: This will take ~30m per 100k comments and cannot be paused. Ensure you have adequate time")
    stage_1(rawdf)
    
    #Find Features
    tft= time.time()
    cntftwords = find_ft_words()    
    print("----")
    print("----")
    print("Time to select the ft word set: " + str((time.time() - tft))) 
    print("----")
    print("Stage 1 Complete.")
    print("----")
    
    #Restart the csv
    virtcsv.seek(0)
    #TODO2 - fix this monstrous waste of memory
        #Read from this, write directly to drive!
    ##pd.read_csv("processeddf.csv", nrows=0)
    ppdf = pd.read_csv(virtcsv)#, nrows=0) #.drop(columns=['Unnamed: 0'])
    virtcsv.seek(0)
    csvheader = 0
    
    tstart2 = time.time()
    print("Stage 2 Starting:")
    stage_2(ppdf)    
    
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
    
def proc_one_co(rawco,ft_words):
    global csvheader, cntftwords,virtcsv, csv_writer
    virtcsv = StringIO()
    csv_writer = writer(virtcsv)
    csvheader = 0
    #Convert to a df
    ##TODO3 - do this the right way (have stage1 accept single raw comments)
        ##rawdf = pd.DataFrame()
        ##rawdf = rawdf.append(rawco, ignore_index=True)
    rawdf = pd.DataFrame([rawco])
    
    #Stage 1
    stage_1(rawdf)
    
    cntftwords = Counter(ft_words.columns.tolist())
    
    #Restart the csv
    virtcsv.seek(0)
    ppdf = pd.read_csv(virtcsv)#, nrows=0) #.drop(columns=['Unnamed: 0'])
    virtcsv.seek(0)
    csvheader = 0
    
    #Stage 2
    stage_2(ppdf)    
    
    virtcsv.seek(0)
    pdf = pd.read_csv(virtcsv)
    
    return pdf
    
if __name__ == "__main__":
    #main()
    print(proc_one_co(rawco,ft_words))
    print(proc_one_co(rawco,ft_words))
















    