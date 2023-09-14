import argparse
import pickle
import os, torch, torchvision
import pandas as pd
import numpy as np

from haven import haven_wizard as hw


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # Get MNIST dataset
    dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset, collate_fn=lambda x: x, batch_size=16, shuffle=True)

    # Create Linear Model and Optimizer
    model = torch.nn.Linear(784, 10)
    opt = torch.optim.Adam(model.parameters(), lr=exp_dict["lr"])

    # Resume or initialize checkpoint
    savedir_model = os.path.join(savedir, "model.pth")
    savedir_score_list = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(savedir_model):
        state_dict = torch.load(savedir_model)
        model.load_state_dict(state_dict)
        with open(savedir_score_list, "rb") as f:
            score_list = pickle.load(f)
    else:
        score_list = []

    # Train and Validate
    for epoch in range(10):
        # Train for one epoch
        train_dict = []
        for batch in train_loader:
            # Get Images
            X = [torch.FloatTensor(np.array(x[0])) for x in batch]
            X = torch.stack(X).view(-1, 784)

            # Get labels
            y = [x[1] for x in batch]

            # Forward pass
            out = model.forward(X)
            loss = torch.nn.functional.cross_entropy(out, torch.LongTensor(y))

            # Backpropagate
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Get scores for one iteration
            acc = (out.argmax(dim=1) == torch.LongTensor(y)).float().mean()
            train_dict += [{"loss": float(loss), "acc": float(acc)}]

        # Get avg scores for one epoch
        train_dict_avg = pd.DataFrame(train_dict).mean().to_dict()

        # Get Metrics from last iteration of the epoch
        score_dict = {"epoch": epoch, "loss": train_dict_avg["loss"], "acc": train_dict_avg["acc"]}

        # Save Metrics in "savedir" as score_list.pkl
        score_list += [score_dict]

        with open(savedir_score_list, "wb") as f:
            pickle.dump(score_list, f)
        torch.save(model.state_dict(), os.path.join(savedir, "model.pth"))

        # Print Metrics
        print(pd.DataFrame(score_list).tail())

    print("Experiment done\n")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        default="syn",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="results/",
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument("-r", "--reset", default=0, type=int, help="Reset or resume the experiment.")
    parser.add_argument("-j", "--job_scheduler", default=None, help="Choose Job Scheduler.")
    parser.add_argument("--python_binary", default="python", help="path to your python executable")

    args, others = parser.parse_known_args()

    # Define a list of experiments
    if args.exp_group == "syn":
        exp_list = []
        for lr in [1e-3, 1e-4, 1e-5]:
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
        python_binary_path=args.python_binary,
        args=args,
    )
