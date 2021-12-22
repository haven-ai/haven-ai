


<p align="center" width="100%">
    <img width="75%" src="docs/images/haven_logo.png"> 
</p>


### Motivation (<a href="https://colab.research.google.com/drive/1iqmJWTfsC3Erfay_SwEoUhq_wS4l18Fd?usp=sharing">Try the Google Colab</a>)

Haven-AI is a library that helps you easily turn your code base into an effective, large-scale machine learning toolkit. You will be able to launch thousands of experiments in parallel, visualize their results and status, and ensure that they are reliable, reproducible, and that the code base is modular to facilitate collaboration and easy integration of new models and datasets.

The goal of this library is help you quickly and efficiently find solutions to machine learning problems, get papers accepted, and win at competitions.

## **Getting Started** 

See this 

- <a href="https://github.com/IssamLaradji/image_classification_template">mnist template code</a>
- <a href="https://github.com/IssamLaradji/semantic_segmentation_template">semantic segmentation template code</a>

### 1. Install requirements

`pip install --upgrade git+https://github.com/haven-ai/haven-ai` 

### 2. Train and Validate

```python
python trainval.py -e syn -sb results -r 1 -j None
```

Argument Descriptions:
```
-e  [Experiment group to run like 'syn'] 
-sb [Directory where the experiments are saved]
-r  [Flag for whether to reset the experiments]
-j  [Scheduler for launching the experiment like 'slurm, toolkit, gcp'. 
     None for running them on local machine]
```

### 3. Visualize the Results

Open `results.ipynb` for visualization.

<p align="center" width="100%">
<img width="65%" src="docs/vis.gif">
</p>


### 4. Run experiments on the cluster

- By defining the job scheduler to be slurm or toolkit for example, you get the following functionality.

<p align="center" width="100%">
<img width="65%" src="docs/ork2.gif">
</p>


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
│   ├── datasets.py
│   └── models.py
├── exp_configs.py
└── trainval.py          
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
