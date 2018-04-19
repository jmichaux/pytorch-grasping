import subprocess
import time

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import bytescale
from torch.optim import lr_scheduler


def lr_sched(scheduler_type, optimizer, **kwargs):
  if scheduler_type == 'step':
    return lr_scheduler.StepLR(optimizer, **kwargs)
  if scheduler_type == 'lambda':
    return lr_scheduler.LambdaLR(optimizer, **kwargs) 
  if scheduler_type == 'multi' or scheduler_type == 'multi_step':
    return lr_scheduler.MultiStepLR(optimizer, **kwargs) 
  if scheduler_type == 'exponential':
    return lr_scheduler.ExponentialLR(optimizer, **kwargs) 
  if scheduler_type == 'cosine':
    return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs) 
  if scheduler_type == 'plateau':
    return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs) 

def set_optimizer(optimizer_type):
  if optimizer_type == 'sgd' or optimizer_type == 'SGD':
    return 


def get_job_id():
  try:
    job_id = os.environ['SLURM_JOB_ID']
  except:
    job_id = None
  return job_id


def time_remaining(job_id, time_limit):
  time_used = subprocess.run(['squeue -j {} -h -o%M'.format(job_id)], stdout=subprocess.PIPE, shell=True)
  time_used = time_used.stdout.decode("utf-8").strip('\n')
  try:
    time_used = time.strptime(time_used, "%H:%M:%S")
  except:
    time_used = time.strptime(time_used, "%M:%S")
  hours, mins, sec = time_used.tm_hour, time_used.tm_min, time_used.tm_sec
  total_sec = sec + 60 * (mins + 60 * hours)
  return time_limit - total_sec


def enough_time(job_id, epoch_time, time_limit=14400):
  if job_id is None:
    return True
  return time_remaining(job_id, time_limit) > 2 * epoch_time  

