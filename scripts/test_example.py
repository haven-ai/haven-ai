from haven import haven_examples as he
import sys
import os
import pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)


if __name__ == "__main__":
    he.save_example_results(savedir_base=".tmp")
