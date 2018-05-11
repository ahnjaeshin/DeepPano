#!/usr/bin/env python3

import os
import errno
from utils import slack_message
import subprocess
import signal
import argparse
import fcntl
from torch import cuda
from threading import Lock
import threading

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", "-m", type=str, help="add, watch, end", default="add")
parser.add_argument("--gpu", "-g", nargs='+', help="gpu ids to use")
parser.add_argument("--config", "-c", type=str, help="path to config file")
parser.add_argument("--log", "-l", type=str, help="optional stdout")

JOBS = 'experiments'


running = {}
gpu_locks = [Lock() for c in cuda.device_count()] if cuda.is_available() else None
# gpu_locks = [Lock(), Lock()]

class Experiment():
    """a experiment object
    gpu : number of gpus to use (list)
    config : path to config (str)
    """

    def __init__(self, gpu, config, log):
        self.config = config
        self.log = log

        if gpu is None:
            gpu = [str(idx) for idx in range(cuda.device_count())] if cuda.is_available() else None

        self.gpu = gpu

    @staticmethod
    def makeExperiment(e):
        gpu, config, log = e.split('|')
        gpu = gpu.split(',')
        return Experiment(gpu, config, log)

    def __str__(self):
        gpu = ','.join(self.gpu)
        return "{}|{}|{}\n".format(gpu, self.config, self.log)
        
def run(experiment):
    
    env = os.environ.copy()

    if experiment.gpu is not None: # and cuda.is_available():
        
        gpu = experiment.gpu
        gpu.sort()
        for idx in gpu:
            gpu_locks[int(idx)].acquire()

        env['CUDA_VISIBLE_DEVICES'] = ' '.join(experiment.gpu)

    command = ['python3', 'driver.py', '--config', experiment.config]

    if experiment.log is None:
        p = subprocess.Popen(command, env=env, universal_newlines=True)
        slack_message('new experiment pid({})'.format(p.pid))
        running[p.pid] = experiment, None
    else:
        log = open(experiment.log, 'w')
        p = subprocess.Popen(command, env=env, universal_newlines=True, stdout=log)
        slack_message('new experiment pid({})'.format(p.pid))
        running[p.pid] = experiment, log


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
                experimentNum += 1
            
                experiment = Experiment.makeExperiment(e)
                thread = threading.Thread(target=run, args=(experiment,))
                thread.start()
                

def add(args):
    with open(JOBS, 'w', ) as fifo:
        fifo.write(str(Experiment(args.gpu, args.config, args.log)))
        print("Added new experiment")

def end():
    with open(JOBS, 'w', ) as fifo:
        fd = fifo.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
        fifo.write('end')

def done(signum, p):
    
    print('experiment done')

    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG|os.WUNTRACED|os.WCONTINUED)
            experiment, log = running[pid]

            print('pid {} gpu {}----------'.format(pid, experiment.gpu))
            
            if log is not None:
                log.close()

            gpu = experiment.gpu
            gpu.sort()
            for idx in gpu:
                gpu_locks[int(idx)].release()
        except:
            break

def main(args):
    signal.signal(signal.SIGCHLD, done)

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

