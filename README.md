


<p align="center" width="100%">
    <img width="50%" src="docs/images/haven_logo.png"> 
</p>



### **Create end-to-end ML projects with the following 6 steps**

### 0. Install
```
pip install -r requirements.txt
```

### 1. Create trainval.py with the following code

```python
import tqdm, os

from haven import haven_examples as he
from haven import haven_wizard as hw

# 3. define trainval function
def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # 4. Create data loader and model 
    train_loader = he.get_loader(name=exp_dict['dataset'], split='train', 
                                 datadir=os.path.dirname(savedir),
                                 exp_dict=exp_dict)
    model = he.get_model(name=exp_dict['model'], exp_dict=exp_dict)

    # 5. load checkpoint
    chk_dict = hw.get_checkpoint(savedir)

    # 6. Add main loop
    for epoch in range(chk_dict['epoch'], 3):
        # 7. train for one epoch
        for batch in tqdm.tqdm(train_loader):
            train_dict = model.train_on_batch(batch)

        # 8. get and save metrics
        score_dict = {'epoch':epoch, 'loss':train_dict['loss']}
        chk_dict['score_list'] += [score_dict]
        hw.save_checkpoint(savedir, score_list=chk_dict['score_list'])

    print('Experiment done')

# 0. create main
if __name__ == '__main__':
    # 1. define a list of experiments
    exp_list = [{'dataset':'mnist', 'model':'linear', 'lr':lr} 
                for lr in [1e-3, 1e-4]]

    # 2. Launch experiments using magic command
    hw.run_wizard(func=trainval, exp_list=exp_list)
```


### 4. Run the experiments

```
python trainval.py --reset 1
```

### 5. Visualize 

Step 4 creates `trainval_results.ipynb`, open the file on Jupyter to get tables and plots

![](docs/images/table1.png)

You can launch Jupyter with,

```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter notebook
```

### 6. Run the experiments in cluster

Using the `api`,

```
python trainval.py --run_jobs 1 --reset 1
```

Using the `cli`,

```
haven --fname trainval.py --function trainval --dataset mnist --model lenet --lr 0.001,0.1
```


## Projects based on Haven-AI

- COVID19: https://www.elementai.com/news/2020/accelerating-the-creation-of-ai-models-to-combat-covid-19-with-element-ai-orkestrator?fbclid=IwAR33nO8gqq7Bf8F4NyG3WYlH3c7t7QkgPAZEru4UpP2FGzSYfHZ5nnPLKZI
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
