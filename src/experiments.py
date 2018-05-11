#!/usr/bin/env python3

import os
import errno
from utils import slack_message
import subprocess
import signal
import argparse
import fcntl
from torch import cuda

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", "-m", type=str, help="add, watch, end", default="add")
parser.add_argument("--gpu", "-g", nargs='+', help="gpu ids to use")
parser.add_argument("--config", "-c", type=str, help="path to config file")
parser.add_argument("--log", "-l", type=str, help="optional stdout")

JOBS = 'experiments'
signal.signal(signal.SIGCHLD, signal.SIG_IGN)

class Experiment():
    """a experiment object
    gpu : number of gpus to use (list)
    config : path to config (str)
    """

    def __init__(self, gpu, config, log):
        self.config = config
        self.log = log

        if gpu is None:
            gpu = ','.join(range(cuda.device_count())) if cuda.is_available() else None
        elif gpu != "None":
            gpu = ','.join(gpu)
                
        self.gpu = gpu
        

    @staticmethod
    def makeExperiment(e):
        e = e.split('|')
        return Experiment(*e)

    def __str__(self):
        return "{}|{}|{}\n".format(self.gpu, self.config, self.log)
        
def run(experiment):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = experiment.gpu
    command = ['python3', 'driver.py', '--config', experiment.config]

    if experiment.log is None:
        p = subprocess.Popen(command, env=env)
        p.wait()
    else:
        p = subprocess.Popen(command, env=env, stdout=subprocess.PIPE)
        slack_message('new experiment pid({})'.format(p.pid))
        with open(experiment.log, 'wb') as log:
            while True:
                line = p.stdout.readline()
                if line != b'':
                    log.write(line)
                else: 
                    break
            


def watch():
    experimentNum = 0

    while True:
        print("Ready for new experiment")
        with open(JOBS, 'r') as fifo:
            while True:
                e = fifo.readline().strip()
                if len(e) == 0:
                    break
                if e == 'end':
                    exit(0)

                print("New experiment start: {}".format(experimentNum))
                slack_message("New experiment start: {}".format(experimentNum)) #, "#driver-py")
                experimentNum += 1
            
                experiment = Experiment.makeExperiment(e)
                run(experiment)
                

def add(args):
    with open(JOBS, 'w', ) as fifo:
        # fd = fifo.fileno()
        # fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
        fifo.write(str(Experiment(args.gpu, args.config, args.log)))
        print("Added new experiment")

def end():
    with open(JOBS, 'w', ) as fifo:
        fd = fifo.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
        fifo.write('end')

def main(args):
    try:
        os.mkfifo(JOBS)
    except OSError as oe:
        if oe.errno != errno.EEXIST:
            raise

    if args.mode == 'watch':
        watch()
    elif args.mode == 'add':
        add(args)
    elif args.mode == 'end':
        end()
    else:
        return

if __name__ == "__main__":
    main(parser.parse_args())

