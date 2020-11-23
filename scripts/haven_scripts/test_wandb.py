import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import wandb
from haven import haven_wizard as hw

if __name__ == "__main__":
  # first way
  score_dict = {'loss':loss}
  wandb.send(score_dict)
    
  # second way
  chk = load_checkpoint(savedir)
  hw.save_checkpoint(savedir, score_dict=score_dict, wandb_config={})
