#!/usr/bin/env python3

import os
import errno
from utils import slack_message
import subprocess
import signal
import argparse
import fcntl
from torch import cuda
# import cuda # for testing
from threading import Lock, Condition
import threading
import shutil


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", "-m", type=str, help="add, watch, end", default="add")
parser.add_argument("--gpu", "-g", nargs='+', help="gpu ids to use")
parser.add_argument("--config", "-c", type=str, help="path to config file")
parser.add_argument("--log", "-l", type=str, help="optional stdout")
parser.add_argument("--pid", "-p", type=int, help="if delete process")

JOBS = 'experiments'


running = {}
gpu_lock = Condition()
gpu_acquired = [False for idx in range(cuda.device_count())] if cuda.is_available() else None

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
        if gpu != "None":
            gpu = gpu.split(',')
        return Experiment(gpu, config, log)

    def __str__(self):
        gpu = self.gpu
        if gpu is not None:
            gpu = ','.join(self.gpu)
        return "{}|{}|{}\n".format(gpu, self.config, self.log)
        
def canRunGPU(gpu):
    for idx in gpu:
        if gpu_acquired[int(idx)]:
            return False
    return True
    
def run(experiment):
    
    env = os.environ.copy()
    
    if experiment.gpu is not None: # and cuda.is_available():
        
        if cuda.is_available():
            gpu = experiment.gpu
            gpu.sort()

            gpu_lock.acquire()

            while not canRunGPU(gpu):
                gpu_lock.wait()

            for idx in gpu:
                gpu_acquired[int(idx)] = True
        
            gpu_lock.release()

        env['CUDA_VISIBLE_DEVICES'] = ', '.join(experiment.gpu)

    command = ['python3', 'driver.py', '--config', experiment.config]

    if experiment.log is None:
        p = subprocess.Popen(command, env=env, universal_newlines=True)
        running[p.pid] = experiment, None, p
    else:
        log = open(experiment.log, 'w')
        p = subprocess.Popen(command, env=env, universal_newlines=True, stdout=log)
        running[p.pid] = experiment, log, p
    
    slack_message('new experiment pid({})'.format(p.pid))
    print('new experiment pid({})'.format(p.pid))


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

                if 'delete' in e:
                    pid = int(e.split(':')[1])
                    if pid not in running:
                        print("process with pid {} not found".format(pid))
                        return
                    
                    _, _, p = running[pid]
                    p.kill()
                    slack_message("process with pid {} deleted".format(pid))
                    print("process with pid {} deleted".format(pid))
                    return

                print("New experiment queued: {}".format(experimentNum))
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
            pid, status = os.waitpid(-1, os.WNOHANG|os.WUNTRACED)
            experiment, log, p = running[pid]

            print('pid {} gpu {}----------'.format(pid, experiment.gpu))

            if log is not None:
                log.close()

            if cuda.is_available():
                gpu = experiment.gpu
                gpu.sort()
                
                gpu_lock.acquire()

                for idx in gpu:
                    gpu_acquired[int(idx)] = False
                
                gpu_lock.notifyAll()
                gpu_lock.release()
        except:
            break

    

def delete(pid):
    with open(JOBS, 'w', ) as fifo:
        fifo.write('delete:{}'.format(pid))

    

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
    elif args.mode == 'delete':
        delete(args.pid)
    elif args.mode == 'end':
        end()
    else:
        return

if __name__ == "__main__":
    main(parser.parse_args())

