import unittest
from haven import haven_results as hr
import shutil
from haven import haven_utils as hu
import sys
import os

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)


class Test(unittest.TestCase):
    def test_zipdir(self):
        # save a score_list
        savedir_base = ".tmp"
        exp_dict = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}
        score_list = [{"epoch": 0, "acc": 0.5}, {"epoch": 0, "acc": 0.9}]

        hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
        hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)
        # check if score_list can be loaded and viewed in pandas
        exp_list = hr.get_exp_list(savedir_base=savedir_base)

        score_lists = hr.get_score_lists(exp_list, savedir_base=savedir_base)
        assert score_lists[0][0]["acc"] == 0.5
        assert score_lists[0][1]["acc"] == 0.9
        from haven import haven_dropbox as hd

        hd.zipdir([hu.hash_dict(exp_dict) for exp_dict in exp_list], savedir_base, src_fname=".tmp/results.zip")
        shutil.rmtree(savedir_base)


if __name__ == "__main__":
    unittest.main()
