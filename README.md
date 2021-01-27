


<p align="center" width="100%">
    <img width="75%" src="docs/images/haven_logo.png"> 
</p>





    
## **Getting Started**

- Run a minimal example with <a href="https://colab.research.google.com/drive/1iqmJWTfsC3Erfay_SwEoUhq_wS4l18Fd?usp=sharing">Google Colab</a>.

- Run experiments in sequence: 

```
python trainval.py --savedir_base /mnt/home/results -r 1  -v results.ipynb
```

- Run experiments in parallel using **slurm** tested under **Compute Canada** servers:

```
python trainval.py --savedir_base /mnt/home/results -r 1 -j 1 -v results.ipynb
```

- Run experiments in parallel using **toolkit**:

```
python trainval.py --savedir_base /mnt/home/results -r 1 -j 2 -v results.ipynb
```
<p align="center" width="100%">
<img width="65%" src="docs/images/ork2.gif">
</p>

- Visualize experiments by opening `results.ipynb` and get the following output.
<p align="center" width="100%">
<img width="65%" src="docs/images/vis.gif">
</p>

### Install
```
pip install --upgrade git+https://github.com/haven-ai/haven-ai
```

## Structure

<table>
      <tr>
    <td>Codebase Structure</td>
     <td>Result Folder Structure</td>
  </tr>
      <tr>
    <td valign="top">
          <pre>
project/
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   └── models.py
├── scripts/
│   └── train_on_single_image.py
├── exp_configs.py
├── README.md
└── trainval.py          # a copy of the code
          </pre>
          </td>
    <td valign="top">
           <pre>
results/
├── experiment_1/
│   ├── code/            # a copy of the code
│   ├── images/          # qualitative results
│   ├── exp_dict.json    # defines the hyperparameters
│   ├── score_list.pkl   # list of scores saved each epoch
│   ├── model.pth        # saved model state
│   └── job_dict.json    # contains the job info
          </pre>
       </td>
  </tr>
 </table>
 
## Papers based on Haven-AI

- COVID19: https://www.elementai.com/news/2020/accelerating-the-creation-of-ai-models-to-combat-covid-19-with-element-ai-orkestrator
- LCFCN: https://github.com/ElementAI/LCFCN
- Embedding Propagation: https://github.com/ElementAI/embedding-propagation
- Bilevel Augmentation: https://github.com/ElementAI/bilevel_augment
- SSN optimizer: https://github.com/IssamLaradji/ssn
- SLS optimizer: https://github.com/IssamLaradji/sls
- Ada SLS optimizer: https://github.com/IssamLaradji/ada_sls
- SPS optimizer: https://github.com/IssamLaradji/sps
- Fish Dataset: https://github.com/alzayats/DeepFish
- Covid Weak Supervision: https://github.com/IssamLaradji/covid19_weak_supervision

## Motivation

- Haven is a library for building, managing and visualizing large-scale reproducible experiments. It helps developers establish a workflow that allows them to quickly prototype a reliable codebase. It also helps them easily  scale that codebase to one that can run, manage, and visualize thousands of experiments seamlessly. 

- The library provides a wide range of functionality including best practices for defining experiments, checkpointing, visualizing and debugging experiments, and ensuring reproducible benchmarks. 

- This library could strongly boost productivity for building great products, winning machine learning competitions, and getting research papers published. 

- The only structure required is that each experiment has the following.
    * `<savedir_base>/<exp_id>/exp_dict.json` that defines a single set of hyperparamters as a python `dict`.
    * `exp_id` is the hash of that hyperparameter python `dict`.
    * `<savedir_base>/<exp_id>/score_list.pkl` that has a list of dicts where each dict contains metrics as keys with scores as their values.


## Expected structure of a Havenized project
```
project/
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   └── models.py
├── scripts/
│   ├── visualize_mnist.py
│   └── train_on_single_image.py
├── exp_configs.py
├── README.md
└── trainval.py
```

## Contributing

We love contributions!
