# SmarterAutoMod

Machine learning script that can automatically report/remove bad comments in a reddit community. Two part process that examines moderation history to learn what types of comments typically get removed, learning from that, it is able to make predictions on new comments made and will report bad comments or remove them (if certainty is high enough).

### Prerequisites

No programming, advanced stats, or machine learning knowledge is needed to use this bot! 

You will need:

- A computer or server that can run continuously with a few hundred MB of ram and drive space.
- Experience running scripts, tweaking config files or a willingness to figure it out
- Some time to spend tuning the settings for your sub reddit (though it should work OK out of the box with defaults)
- Mod access to a subreddit that gets a [few hundred comments a day](https://subredditstats.com/) (to ensure enough data to learn from)

This is a python 3.7 project. In addition, you will need to ensure you have the following packages installed (via pip or otherwise):

- praw
- scikit-learn
- numpy
- pandas
- matplotlib
- nltk
- configparser

```
  ie. pip install praw scikit-learn numpy pandas matplotlib nltk configparser
```


## Initial setup/Config

* Rename config.ini.default to config.ini then start to edit it

* Login - Put in your oauth bot's login credentials. Go here: https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example if you're not sure how to get an oauth refresh token. The bot's account must of course have mod access to the subreddit. This is probably the most tedious step in the whole process. 

* General - You'll need to put in your subreddit name, and change any other settings here you like, most of the regex can be taken from your existing automod.

* Processing - The number of words here is important. If you're just trialing this software, 500 or 1000 will do a decent job. r/SpaceX is using 5000 currently which gets ~10% better accuracy than 1000.

* Training - Nothing in here needs to be changed unless you're retuning the hyperparameters for your subreddit

* Execution - "FPR" refers to false positive rate. Effectively, at what confidence level do you want the bot to report/remove comments at? Or what rate of false positives are you willing to accept. If you set it too high, like to 75, it'll report most of the comments in the subreddit! Keep it low to start.

  * To avoid issues, there is a test_mode (on by default) so that the bot will only print out what it *would* do without ever doing anything on reddit itself. When you've got the settings you want and after running it in this mode for a while, the bot appears to be correctly flagging bad comments, you can switch this off. For an extra layer of flexibility, when you turn off test_mode, you can leave on report_only_mode which will have the bot do automated reports, but not removals.

  * This also allows for removal notifications (to users whose comments have been removed). And screening notifications (to the modteam so that you cans still see what the bot has removed, just to keep an eye on it). These use the same formatting options as automod so you should be able to basically copy paste yours if desired

Test mode on console:
![Test mode on console](https://raw.githubusercontent.com/Ambiwlans/SmarterAutoMod/master/images/classifier%202019-05-11.png)


### Data collection

Run data_collector.py. Wait. This can be stopped and started at any time since it regularly saves while data is being collected. The process of collecting 1000 submissions takes 6 ~ 8 hours. Creates a ~15MB file.

### Data processing

Run data_processing.py. This takes significantly more computer resources and can take a few hours. It cannot be stopped midway. I suggest you run it overnight to not get in the way. Creates a ~15MB file.

### Learning

Train.py can be run as is with no tweaking. It will produce a ROC graph and save a model to disk (10~500MB file depending on the number of features, estimators etc.). This should complete in ~5 minutes.

If you want to examine what the model is doing, feature importance, want to retune your hyperparameters or otherwise want to debug the model, you'll need to edit the main() function in train.py. I've set up a number of useful codeblocks you can just uncomment to get them to run without you having to understand what the code is doing. Note that some of these functions can take a long time to complete.

Sample showing [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and Feature Importance:
![Sample showing ROC and Feature Importance](https://raw.githubusercontent.com/Ambiwlans/SmarterAutoMod/master/images/50%20FT%20words%202019-05-11.png)


## Deployment

Once you have trained a model, running classifier.py will start the bot classifying new comments. Note that when you first start the bot, it will immediately run through the past few hours of comments, in no particular order, this is a function of how reddit provides bots new comments.

That's it! Going through the earlier steps every 6 months or so would probably be valuable in catching new trends in rule-breaking comments. But otherwise, you're done aside from tuning it.

Running in report only mode:
![Running in report only mode](https://raw.githubusercontent.com/Ambiwlans/SmarterAutoMod/master/images/mod%20queue%202019-05-05.png)

## Bugs! Help! Feature Requests! Job Offers

Please contact u/Ambiwlans here or on reddit if you need help with setup, run into any bugs, or otherwise have a request. 

## Contributing

Please contact u/Ambiwlans here or on reddit if you're interested in contributing. 

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Authors

Created by Ambiwlans for [r/SpaceX](https://www.reddit.com/r/SpaceX). Find me on [github](https://github.com/Ambiwlans) or [reddit](https://www.reddit.com/u/Ambiwlans)

See also the list of [contributors](https://github.com/Ambiwlans/SmarterAutoMod/contributors) who participated in this project.

## Acknowledgments

* r/SpaceX mod team - Thanks to everyone who let me slack on mod duties while I made this
* Christopher Gerlach - Tireless consultant, ML wizard. Find him here on [github](https://github.com/CAM-Gerlach) or [reddit](https://www.reddit.com/u/CAM-Gerlach)
