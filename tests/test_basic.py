import sys, os

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import numpy as np
import os
import torch
import shutil

try:
    import job_configs
except:
    print("Warning: job_configs not found. Some tests will be skipped")

from haven import haven_wizard as hw
from haven import haven_img as hi
from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jobs as hjb

DST_PATH = ".tmp/tests"


def test_cartesian_product():
    # test whether the cartesian product covers all needed variations
    exp_dict_1 = {"dataset": "mnist", "model": "mlp", "batch_size": 1}
    exp_dict_2 = {"dataset": "mnist", "model": "mlp", "batch_size": 5}
    exp_dict_3 = {"dataset": "cifar10", "model": "mlp", "batch_size": 1}
    exp_dict_4 = {"dataset": "cifar10", "model": "mlp", "batch_size": 5}

    exp_list = [exp_dict_1, exp_dict_2, exp_dict_3, exp_dict_4]
    exp_list_cartesian = hu.cartesian_exp_group({"dataset": ["mnist", "cifar10"], "model": "mlp", "batch_size": [1, 5]})

    exp_list_hash = [hu.hash_dict(e) for e in exp_list]
    exp_list_cartesian_hash = [hu.hash_dict(e) for e in exp_list_cartesian]

    # check if the # experiments is correct
    assert len(exp_list_cartesian_hash) == len(exp_list_hash)

    # check that the hashes in the cartesian are all there
    for h in exp_list_hash:
        assert h in exp_list_cartesian_hash

    # check that every hash is unique
    assert len(exp_list_cartesian_hash) == len(np.unique(exp_list_cartesian_hash))


def test_hash():
    # test whether hashing works for nested dicts
    exp_dict_1 = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
    exp_dict_2 = {"dataset": "mnist", "batch_size": 1, "model": {"name": "mlp", "n_layers": 30}}
    exp_dict_3 = {"dataset": "mnist", "batch_size": 1, "model": {"name": "mlp"}}

    assert hu.hash_dict(exp_dict_1) == hu.hash_dict(exp_dict_2)
    assert hu.hash_dict(exp_dict_1) != hu.hash_dict(exp_dict_3)


def test_checkpoint():
    savedir_base = ".results"
    # create exp folder
    exp_dict = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
    savedir = os.path.join(savedir_base, hu.hash_dict(exp_dict))

    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    # hw.report(savedir, [])
    hu.torch_save(os.path.join(savedir, "model.pth"), torch.zeros(10))
    hu.torch_load(os.path.join(savedir, "model.pth"))
    hc.load_checkpoint(exp_dict, savedir_base, fname="model.pth")
    assert os.path.exists(savedir)

    # delete exp folder
    hc.delete_experiment(savedir)
    assert not os.path.exists(savedir)

    # check backup folder
    os.rmdir(savedir_base)


def test_get_score_lists():
    # save a score_list
    savedir_base = "../haven_results"
    exp_dict = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
    score_list = [{"epoch": 0, "acc": 0.5}, {"epoch": 0, "acc": 0.9}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)
    # check if score_list can be loaded and viewed in pandas
    exp_list = hu.get_exp_list(savedir_base=savedir_base)

    score_lists = hr.get_score_lists(exp_list, savedir_base=savedir_base)
    assert score_lists[0][0]["acc"] == 0.5
    assert score_lists[0][1]["acc"] == 0.9

    shutil.rmtree(savedir_base)


def test_get_score_df():
    # save a score_list
    savedir_base = "../haven_results"
    exp_dict = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
    exp_dict2 = {"model": {"name": "mlp2", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}

    score_list = [{"epoch": 0, "acc": 0.5}, {"epoch": 0, "acc": 0.9}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)

    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)

    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict2), "exp_dict.json"), exp_dict)
    # check if score_list can be loaded and viewed in pandas
    exp_list = hu.get_exp_list(savedir_base=savedir_base)
    score_df = hr.get_score_df(exp_list, savedir_base=savedir_base)

    assert np.array(score_df["dataset"])[0].strip("'") == "mnist"

    shutil.rmtree(savedir_base)


def test_get_images():
    # save a score_list
    hi.points_on_image([3, 2], [3, 2], np.ones((100, 100, 3)), radius=3, c_list=[0, 1])


def test_get_plot():
    # save a score_list
    savedir_base = "../haven_results"
    exp_dict = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
    score_list = [{"epoch": 0, "acc": 0.5}, {"epoch": 1, "acc": 0.9}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)
    # check if score_list can be loaded and viewed in pandas
    exp_list = hu.get_exp_list(savedir_base=savedir_base)

    fig, axis = hr.get_plot(
        exp_list,
        savedir_base=savedir_base,
        filterby_list=[({"model": {"name": "mlp"}}, {"style": {"color": "red"}})],
        x_metric="epoch",
        y_metric="acc",
    )
    # fig, axis = hr.get_plot(exp_list,
    #      savedir_base=savedir_base,
    #      x_metric='epoch',
    #      y_metric='acc',
    #      mode='pretty_plot')
    fig, axis = hr.get_plot(exp_list, savedir_base=savedir_base, x_metric="epoch", y_metric="acc", mode="bar")
    fig.savefig(os.path.join(savedir_base, "test.png"))

    shutil.rmtree(savedir_base)


def test_get_result_manager():
    # save a score_list
    savedir_base = "../haven_results"
    if os.path.exists(savedir_base):
        shutil.rmtree(savedir_base)
    exp_dict = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
    score_list = [{"epoch": 0, "acc": 0.5}, {"epoch": 1, "acc": 0.9}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)

    exp_dict = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "cifar10", "batch_size": 1}
    score_list = [{"epoch": 0, "acc": 0.25}, {"epoch": 1, "acc": 1.24}, {"epoch": 2, "acc": 1.5}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)

    exp_dict = {"model": {"name": "lenet", "n_layers": 30}, "dataset": "cifar10", "batch_size": 1}
    score_list = [{"epoch": 0, "acc": 0.35}, {"epoch": 1, "acc": 1.2}, {"epoch": 2, "acc": 1.3}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)

    exp_dict = {"model": {"name": "lenet", "n_layers": 30}, "dataset": "cifar10", "batch_size": 5}
    score_list = [{"epoch": 0, "acc": 0.15}, {"epoch": 1, "acc": 1.21}, {"epoch": 2, "acc": 1.7}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)
    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list_test.pkl"), score_list)

    exp_dict = {
        "model": [{"name": "lenet", "n_layers": 30}, {"name": "lenet", "n_layers": 15}],
        "dataset": "cifar10",
        "batch_size": 5,
    }
    score_list = [{"epoch": 0, "acc": 0.15}, {"epoch": 1, "acc": 1.21}, {"epoch": 2, "acc": 1.7}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)

    rm2 = hr.ResultManager(savedir_base=savedir_base, score_list_name="score_list_test.pkl")
    assert len(rm2.get_score_lists()) == 1
    rm = hr.ResultManager(savedir_base=savedir_base)
    rm.get_latex_table(legend=["dataset"], metrics=["acc"])
    # assert(len(rm.exp_groups) == 2)
    # for exp_list in rm.exp_groups:
    #     assert(exp_list[0]['dataset'] in ['mnist', 'cifar10'])
    rm.get_score_df()
    rm.get_exp_list_df()
    rm.get_score_df(avg_across="dataset")
    rm.get_score_df(avg_across="dataset")

    rm.get_score_lists()
    rm.get_images()
    table = rm.get_score_table()
    table = rm.get_exp_table()

    fig_list = rm.get_plot(x_metric="epoch", y_metric="acc", title_list=["dataset"], legend_list=["model.n_layers"])
    for i, fig in enumerate(fig_list):
        fig.savefig(os.path.join(savedir_base, "%d.png" % i))

    order = "groups_by_metrics"
    fig_list = rm.get_plot_all(
        order=order,
        x_metric="epoch",
        y_metric_list=["acc", "epoch"],
        title_list=["dataset"],
        legend_list=["model"],
        groupby_list=["dataset"],
        log_metric_list=["acc"],
        map_title_list=[{"mnist": "MNIST"}, {"cifar10": "CIFAR-10"}],
        map_xlabel_list=[{"epoch": "EPOCHS"}],
        map_ylabel_list=[{"acc": "Score"}],
        ylim_list=[[(0.5, 0.8), (0.5, 0.8)], [(0.5, 0.8), (0.5, 0.8)]],
    )

    for i, fig in enumerate(fig_list):
        fig.savefig(os.path.join(savedir_base, "%s_%d.png" % (order, i)))

    order = "metrics_by_groups"
    fig_list = rm.get_plot_all(
        order=order,
        x_metric="epoch",
        y_metric_list=["acc", "epoch"],
        title_list=["dataset"],
        legend_list=["model"],
        avg_across="batch_size",
    )
    for i, fig in enumerate(fig_list):
        fig.savefig(os.path.join(savedir_base, "%s_%d.png" % (order, i)))

    # shutil.rmtree(savedir_base)


def test_filter_exp_list():
    exp_list = hu.cartesian_exp_group(
        {"dataset": ["imagenet", "mnist", "cifar10"], "model": "mlp", "batch_size": [1, 5]}
    )

    exp_list1 = hu.filter_exp_list(exp_list, filterby_list=[{"dataset": "mnist"}])

    exp_list2 = hu.filter_exp_list(exp_list, filterby_list=[[{"dataset": "mnist"}]])

    exp_list = hu.filter_exp_list(exp_list, filterby_list=[{"dataset": "mnist"}, {"dataset": "cifar10"}])
    visited = []
    for exp_dict in exp_list:
        assert exp_dict["dataset"] in ["mnist", "cifar10"]
        visited += [exp_dict["dataset"]]

    assert "mnist" in visited
    assert "cifar10" in visited


def test_group_exp_list():
    exp_list = hu.cartesian_exp_group(
        {"dataset": ["imagenet", "mnist", "cifar10"], "model": "mlp", "batch_size": [1, 5], "mode": {"fset": 1}}
    )

    list_of_exp_list = hu.group_exp_list(exp_list, groupby_list=["dataset", "mode.fset"])

    list_of_exp_list = hu.group_exp_list(exp_list, groupby_list="dataset")
    for exp_list in list_of_exp_list:
        assert len(set([exp_dict["dataset"] for exp_dict in exp_list])) == 1


def test_hash():
    exp_list = [
        {
            "batch_size": 64,
            "dataset_directory": "/mnt/public/datasets2/softras/mesh_reconstruction",
            "image_size": 64,
            "lambda_flatten": 0.0005,
            "lambda_laplacian": 0.005,
            "learning_rate": 0.0001,
            "lr_type": "step",
            "model_directory": "/mnt/public/datasets2/softras/results/models",
            "reset": 1,
            "seed": 0,
            "sigma_val": 0.0001,
            "loss": [{"name": "base_ubl", "cheat": True}],
            "n_train_ratio": 0.001,
            "n_val_ratio": 0.1,
            "epochs": 6000,
            "version": 2,
            "classes": ["Airplane"],
        }
    ]
    assert hu.hash_dict(exp_list[0]) == "5a4dd8e6a694c732615b9829e9d2eddb"


def test_get_best_exp_dict():
    savedir_base = "../haven_results"
    exp_dict_1 = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
    score_list = [{"epoch": 0, "acc": 0.5}, {"epoch": 1, "acc": 0.9}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict_1), "score_list.pkl"), score_list)

    exp_dict_2 = {"model": {"name": "mlp", "n_layers": 35}, "dataset": "mnist", "batch_size": 1}
    score_list = [{"epoch": 0, "acc": 0.6}, {"epoch": 1, "acc": 1.9}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict_2), "score_list.pkl"), score_list)

    best_exp_list = hu.filter_exp_list(
        [exp_dict_1, exp_dict_2],
        savedir_base=savedir_base,
        filterby_list=[({"model.name": "mlp"}, {"best": {"avg_across": "run", "metric": "acc", "metric_agg": "max"}})],
    )
    assert len(best_exp_list) == 1
    assert best_exp_list[0]["model"]["n_layers"] == 35

    best_exp_list = hu.filter_exp_list(
        [exp_dict_1, exp_dict_2],
        savedir_base=savedir_base,
        filterby_list=[({"model.name": "mlp"}, {"best": {"avg_across": "run", "metric": "acc", "metric_agg": "min"}})],
    )
    assert best_exp_list[0]["model"]["n_layers"] == 30
    # exp 2
    exp_dict_2 = {"model": {"name": "mlp2", "n_layers": 30}, "dataset": "mnist", "batch_size": 1, "run": 0}
    score_list = [{"epoch": 0, "acc": 1.5}, {"epoch": 1, "acc": 1.8}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict_2), "score_list.pkl"), score_list)
    # exp 3
    exp_dict_3 = {"model": {"name": "mlp2", "n_layers": 30}, "dataset": "mnist", "batch_size": 1, "run": 1}
    score_list = [{"epoch": 0, "acc": 1.5}, {"epoch": 1, "acc": 1.3}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict_3), "score_list.pkl"), score_list)

    exp_list = [exp_dict_1, exp_dict_2, exp_dict_3]
    best_exp_dict = hu.get_best_exp_dict(
        exp_list,
        savedir_base=savedir_base,
        metric="acc",
        avg_across="run",
        metric_agg="max",
    )

    assert best_exp_dict["model"]["name"] == "mlp2"


def test_toolkit():
    # toolkit tests
    import job_configs

    exp_list = [{"model": {"name": "mlp", "n_layers": 20}, "dataset": "mnist", "batch_size": 1}]
    savedir_base = "../haven_results"
    os.makedirs(savedir_base, exist_ok=True)
    jm = hjb.JobManager(
        exp_list=exp_list,
        savedir_base=savedir_base,
        workdir=os.path.dirname(os.path.realpath(__file__)),
        job_config=job_configs.JOB_CONFIG,
        job_scheduler="toolkit",
    )
    # get jobs
    job_list_old = jm.get_jobs()

    # run single command
    savedir_logs = os.path.join(savedir_base, str(np.random.randint(1000)))
    os.makedirs(savedir_logs, exist_ok=True)
    command = "echo 2"
    job_id = jm.submit_job(command, workdir=jm.workdir, savedir_logs=savedir_logs)

    # get jobs
    job_list = jm.get_jobs()
    job = jm.get_job(job_id)
    assert job_list[0]["id"] == job_id

    # jm.kill_job(job_list[0].id)
    # run
    print("jobs:", len(job_list_old), len(job_list))
    assert (len(job_list_old) + 1) == len(job_list)

    # command_list = []
    # for exp_dict in exp_list:
    #     command_list += []

    # hjb.run_command_list(command_list)
    # jm.launch_menu(command=command)
    jm.launch_exp_list(command="echo 2 -e <exp_id>", reset=1, in_parallel=False)

    assert os.path.exists(os.path.join(savedir_base, hu.hash_dict(exp_list[0]), "job_dict.json"))
    summary_list = jm.get_summary_list()
    print(hu.filter_list(summary_list, {"job_state": "SUCCEEDED"}))
    print(hu.group_list(summary_list, key="job_state", return_count=True))

    rm = hr.ResultManager(exp_list=exp_list, savedir_base=savedir_base)
    rm_summary_list = rm.get_job_summary()
    df = rm.get_resources_stats(job_dict={"cpu": 4, "mem": 20, "gpu": 1, "gpu_mem": 16})
    assert df is not None
    # print(df)


def test_avg_runs():
    # save a score_list
    savedir_base = "../haven_results"
    exp_dict = {"run": 0, "model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
    exp_dict2 = {"run": 1, "model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}

    score_list = [
        {
            "epoch": 0,
            "acc": 0.2,
        },
        {"epoch": 1, "acc": 1.0},
        {"epoch": 2, "acc": 0.9},
    ]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)

    score_list = [
        {
            "epoch": 0,
            "acc": 0.4,
        },
        {"epoch": 1, "acc": 1.5},
        {"epoch": 2, "acc": 0.5},
    ]
    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict2), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict2), "exp_dict.json"), exp_dict2)
    # check if score_list can be loaded and viewed in pandas
    exp_list = hu.get_exp_list(savedir_base=savedir_base)
    score_df = hr.get_score_df(exp_list, savedir_base=savedir_base, avg_across="run")

    assert score_df[("acc", "mean")].max() == 0.7
    assert abs(score_df[("acc (min)", "mean")].max() - 0.3) < 1e-4
    assert score_df[("acc (max)", "mean")].max() == 1.25

    result_dict = hr.get_result_dict(
        exp_dict, savedir_base, exp_list=exp_list, x_metric="epoch", y_metric="acc", avg_across="run"
    )

    np.testing.assert_array_almost_equal(result_dict["y_list"], np.array([0.3, 1.25, 0.7]))

    score_list = [{"epoch": 1, "acc": 1.5}, {"epoch": 2, "acc": 0.5}]
    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict2), "score_list.pkl"), score_list)
    result_dict = hr.get_result_dict(
        exp_dict, savedir_base, exp_list=exp_list, x_metric="epoch", y_metric="acc", avg_across="run"
    )
    np.testing.assert_array_almost_equal(result_dict["x_list"], np.array([1, 2]))
    np.testing.assert_array_almost_equal(result_dict["y_list"], np.array([1.25, 0.7]))
    shutil.rmtree(savedir_base)


def test_wizard():
    import job_configs

    savedir_base = "../haven_results"

    # if exists
    if os.path.exists(savedir_base):
        shutil.rmtree(savedir_base)

    # not in cluster
    hw.run_wizard(
        func=test_trainval,
        exp_list=[{"lr": 1e-3}],
        savedir_base=savedir_base,
        reset=0,
        results_fname=f"{savedir_base}/results.ipynb",
    )

    print("results_fname:", f"{savedir_base}/results.ipynb\n")

    # test for result manager
    hr.ResultManager(savedir_base=savedir_base)

    # in cluster
    hw.run_wizard(
        func=test_trainval,
        exp_list=[{"lr": 1e-3}],
        savedir_base=savedir_base,
        reset=0,
        results_fname=f"{savedir_base}/results.ipynb",
        job_config=job_configs.JOB_CONFIG,
        job_scheduler="toolkit",
        job_option="run",
    )

    shutil.rmtree(savedir_base)


def test_code_save():
    src_path = "tests"

    def test_ignore(path):
        # assert the files in .havenignore are not copied
        assert not os.path.exists(os.path.join(path, "test_haven.ipynb"))
        assert not os.path.exists(os.path.join(path, "test_basic.py"))

        # assert the rest of the files are copied
        assert os.path.exists(os.path.join(path, "test_optim.py"))
        assert os.path.exists(os.path.join(path, "test_other.py"))

    file_list_to_ignore = ["test_haven.ipynb", "test_basic.py"]

    # 1. Haven ignore way
    # -------------------
    havenignore_fname = os.path.join(src_path, ".havenignore")

    # create a .havenignore
    with open(havenignore_fname, "w") as f:
        f.write("\n".join(file_list_to_ignore))

    # copy the code
    hu.copy_code(src_path=src_path, dst_path=DST_PATH, use_rsync=False)

    # delete .havenignore
    os.remove(havenignore_fname)

    # test ignore
    test_ignore(path=DST_PATH)

    # delete dst
    shutil.rmtree(DST_PATH)

    # 2. Direct ignore way
    # -------------------
    hu.copy_code(src_path=src_path, dst_path=DST_PATH, use_rsync=False, ignore_patterns=file_list_to_ignore)
    test_ignore(path=DST_PATH)

    # delete dst
    shutil.rmtree(DST_PATH)

    # 3. Job Way
    # -------------------
    exp_dict = {"example": 1}
    jm = hjb.JobManager(
        exp_list=[exp_dict],
        savedir_base=DST_PATH,
        workdir=os.path.dirname(os.path.realpath(__file__)),
        job_config=job_configs.JOB_CONFIG,
        job_scheduler="toolkit",
        job_copy_ignore_patterns=file_list_to_ignore,
    )
    command = "echo 2"
    job_dict = jm.launch_exp_dict(exp_dict=exp_dict, savedir=DST_PATH, command=command, job=None)
    test_ignore(path=os.path.join(DST_PATH, "code"))

    # 4. wizard way
    # -------------------

    # delete dst
    shutil.rmtree(DST_PATH)

    exp_dict = {"lr": 1e-3}
    hw.run_wizard(
        func=test_trainval,
        exp_list=[exp_dict],
        savedir_base=DST_PATH,
        reset=0,
        workdir=os.path.dirname(os.path.realpath(__file__)),
        job_config=job_configs.JOB_CONFIG,
        job_scheduler="toolkit",
        job_option="run",
        job_copy_ignore_patterns=file_list_to_ignore,
    )
    test_ignore(path=os.path.join(DST_PATH, hu.hash_dict(exp_dict), "code"))
    shutil.rmtree(DST_PATH)


def test_ignore_status():
    if os.path.exists(DST_PATH):
        shutil.rmtree(DST_PATH)
    exp_dict = {"example": 1}
    hw.run_wizard(
        func=test_trainval,
        exp_list=[exp_dict],
        savedir_base=DST_PATH,
        reset=0,
        workdir=os.path.dirname(os.path.realpath(__file__)),
        job_config=job_configs.JOB_CONFIG,
        job_scheduler="toolkit",
        job_option="run",
        job_ignore_status=["NEVER LAUNCHED", "ABC", "SUCCEEDED"],
    )
    # get from stdout
    assert os.path.exists(os.path.join(DST_PATH, hu.hash_dict(exp_dict), "code")) == False

    hw.run_wizard(
        func=test_trainval,
        exp_list=[exp_dict],
        savedir_base=DST_PATH,
        reset=0,
        workdir=os.path.dirname(os.path.realpath(__file__)),
        job_config=job_configs.JOB_CONFIG,
        job_scheduler="toolkit",
        job_option="run",
        job_ignore_status=["abc"],
    )
    # get from stdout
    assert os.path.exists(os.path.join(DST_PATH, hu.hash_dict(exp_dict), "code")) == True

    shutil.rmtree(DST_PATH)

    print()


def test_trainval(exp_dict, savedir, args):
    assert isinstance(exp_dict, dict)
    assert isinstance(savedir, str)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", nargs="+", default=["basic"])
    args, others = parser.parse_known_args()

    if "basic" in args.mode:
        # baasic tests
        test_ignore_status()
        test_code_save()
        test_wizard()
        test_get_result_manager()
        test_hash()
        test_cartesian_product()
        test_hash()
        test_checkpoint()
        test_get_best_exp_dict()
        test_filter_exp_list()
        test_group_exp_list()

        test_get_plot()
        test_get_score_df()
        test_avg_runs()
        test_get_score_lists()

    if "toolkit" in args.mode:
        test_toolkit()

    if "wizard" in args.mode:
        test_wizard()

    if "trainval" in args.mode:
        test_trainval()
