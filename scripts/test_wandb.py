from haven import haven_wizard as hw
import wandb
import sys
import os
import pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)


if __name__ == "__main__":
    # first way
    score_dict = {"loss": loss}
    wandb.send(score_dict)

    # second way
    chk = load_checkpoint(savedir)
    hw.save_checkpoint(savedir, score_dict=score_dict, wandb_config={})
