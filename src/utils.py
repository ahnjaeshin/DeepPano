import argparse
import socket
import os
import datetime
import time
import logging

from slacker import Slacker

def logger(func):
    """decorator to log arguments passed into the function
    
    Arguments:
        func {function} -- function that you want to log the arguments
    
    """
    
    def wrapper(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        print('[{}] result args - {}, kwargs - {}'.format(timestamp, args, kwargs))
        result = func(*args, **kwargs)
        print('result: {}'.format(result))
        return result

    return wrapper

def timer(func):
    """decorator to time functions (for profiling)
    
    Arguments:
        func {function} -- function you want to time
    """
    pass

def slack_message(text, channel=None):
    """send slack message
    
    Arguments:
        text {string} -- any string want to send
    
    Keyword Arguments:
        channel {string} -- the name of channel to send to (default: channel #botlog)
    """

    
    slack_token = os.environ["SLACK_API_TOKEN"]

    slack = Slacker(slack_token)
    host = socket.gethostname() + '@bot'
    
    if not channel:
        channel = 'C9ZKLPGBV' # channel id of #botlog channel

    slack.chat.post_message(channel, text, as_user=False, username=host)

def main():
    slack_message('앞으로 aws ec2 instance에서 학습 돌리는건 여기로 logging이 될 것입니다, 원래 이 함수에 exception도 잡아야 하는데 귀찮..', '#ct')

if __name__ == '__main__':
    main()