import tqdm
import argparse
import os, torch, exp_configs

from src import datasets, models
from haven import haven_examples as he
from haven import haven_wizard as hw
from haven import haven_results as hr
from haven import haven_utils as hu


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # Create data loader and model
    train_set = datasets.get_dataset(
        name=exp_dict["dataset"],
        split="train",
        datadir=os.path.dirname(savedir),
        exp_dict=exp_dict,
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, num_workers=8)
    model = models.get_model(name=exp_dict["model"], exp_dict=exp_dict).cuda()

    # Resume or initialize checkpoint
    cm = hw.CheckpointManager(savedir)
    state_dict = cm.load_model()
    if state_dict is not None:
        model.set_state_dict(state_dict)

    # Train and Validate
    for epoch in tqdm.tqdm(range(cm.get_epoch(), 100), desc="Running Experiment"):
        # Train for one epoch
        train_dict = model.train_on_loader(train_loader, epoch=epoch)

        # Get Metrics
        score_dict = {
            "epoch": epoch,
            "acc": train_dict["train_acc"],
            "loss": train_dict["train_loss"],
        }

        # Save Metrics in "savedir" as score_list.pkl
        cm.log_metrics(score_dict)
        torch.save(model.state_dict(), os.path.join(savedir, "model.pth"))

        # Save Example Image for qualitative results
        image = torch.zeros(50, 50, 3)
        hu.save_image(os.path.join(savedir, "images", "example.png"), image)

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
        "-sb",
        "--savedir_base",
        required=True,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument("-r", "--reset", default=0, type=int, help="Reset or resume the experiment.")
    parser.add_argument("-j", "--job_scheduler", default=None, help="Choose Job Scheduler.")
    parser.add_argument("--python_binary", default="python", help="path to your python executable")

    args, others = parser.parse_known_args()

    # Choose Job Scheduler
    job_config = None

    if args.job_scheduler == "toolkit":
        import job_configs

        job_config = job_configs.JOB_CONFIG

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_configs.EXP_GROUPS[args.exp_group],
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        results_fname="results_haven.ipynb",
        python_binary_path=args.python_binary,
        args=args,
    )
