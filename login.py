# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:23:47 2019

@author: Angelo
@description: Login script for autoautomod
"""

#TODO1 - for 1.0 release (internal)
#TODO2 - for 2.0 release (public)

###############################################################################
### Imports, Defines
###############################################################################

#Python Reddit API Wrapper
import praw                      

###############################################################################
###  Functions
###############################################################################

def login(username="", password="", client_id="",client_secret="",refresh_token="",user_agent=""):
    
#TODO2 - add logic that uses login details that are available (or gives an error)
    # should guide user through to getting a refresh_token and removing their password
    # at minimum give guide on how to get a token and error out if not available
    # should gracefully error out when info isn't present
    
    return praw.Reddit(client_id=client_id, client_secret=client_secret, refresh_token=refresh_token, user_agent=user_agent)


#TODO2 check if mod, else fail out














