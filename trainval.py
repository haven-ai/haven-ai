import tqdm
import argparse
import os

from haven import haven_examples as he
from haven import haven_wizard as hw
from haven import haven_results as hr


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # Create data loader and model
    train_loader = he.get_loader(
        name=exp_dict["dataset"], split="train", datadir=os.path.dirname(savedir), exp_dict=exp_dict
    )
    model = he.get_model(name=exp_dict["model"], exp_dict=exp_dict)

    # Resume or initialize checkpoint
    cm = hw.CheckpointManager(savedir)
    state_dict = cm.load_model()
    if state_dict is not None:
        model.set_state_dict(state_dict)

    # Train and Validate
    for epoch in tqdm.tqdm(range(cm.get_epoch(), 3), desc="Running Experiment"):
        # Train for one epoch
        train_dict = model.train_on_loader(train_loader, epoch=epoch)

        # Get Metrics
        score_dict = {"epoch": epoch, "acc": train_dict["train_acc"], "loss": train_dict["train_loss"]}

        # Save Metrics and Model
        cm.log_metrics(score_dict)
        cm.save_torch("model.pth", model.state_dict())

    print("Experiment done\n")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb", "--savedir_base", required=True, help="Define the base directory where the experiments will be saved."
    )
    parser.add_argument("-r", "--reset", default=0, type=int, help="Reset or resume the experiment.")
    parser.add_argument("-j", "--job_scheduler", default=None, help="Choose Job Scheduler.")

    args, others = parser.parse_known_args()

    # Define a list of experiments
    if args.exp_group == "syn":
        exp_list = []
        for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            exp_list += [{"lr": lr, "dataset": "syn", "model": "linear"}]

    # Choose Job Scheduler
    job_config = None

    if args.job_scheduler == "slurm":
        job_config = {
            "account_id": "def-dnowrouz-ab",
            "time": "1:00:00",
            "cpus-per-task": "2",
            "mem-per-cpu": "20G",
            "gres": "gpu:1",
        }

    elif args.job_scheduler == "toolkit":
        import job_configs
        job_config = job_configs.JOB_CONFIG

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_list,
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname="results.ipynb",
        args=args
    )
