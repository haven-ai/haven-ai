# Classification - MNIST Example

## 1. Install Haven
`pip install --upgrade git+https://github.com/haven-ai/haven-ai`

This command installs the [Haven library](https://github.com/haven-ai/haven-ai) which helps in managing the experiments.

## 2. Experiement
Run the experiment with the following command on terminal

`python trainval.py -e <exp_group_name> -sb <directory_to_save_results> -d <directory_to_save_datasets> -r 1`

-e: Indicate which experiemnt you want to run

option1: mnist_batch_size. This experiemnt group investigates the effect of training batch sizes on the model. 
    
option2: mnist_learning_rate. This experiment group investigates the effect of learning rates on the model.

-sb: Specify where the experiment results will be stored

-d: Specify where the datasets will be stored

-r: Indicate whether to reset the model after each experiment


## 3. Visualize Results
### 3.1 Launch Jupyter by running the following  on terminal
```
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter notebook
```

### 3.2 On a Jupyter cell, run the following script
```python
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu
savedir_base = '<the directory of the result>'
fname = '<the directory of the experiemnt configuration file>'

exp_list = []
# indicate the experiment group you have run here
for exp_group in [
    'mnist_learning_rate', 
    'mnist_batch_size'
                 ]:
    exp_list += hu.load_py(fname).EXP_GROUPS[exp_group]

# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      verbose=0
                     )
hj.get_dashboard(rm, vars(), wide_display=True)
```

## 4. Results

### 4.1 Table

<img width="1065" alt="Screen Shot 2020-10-18 at 6 12 01 PM" src="https://user-images.githubusercontent.com/46538726/96387114-dbb34b00-116d-11eb-824d-c532436a57d1.png">

### 4.2 Line Plots

<img width="1020" alt="trainloss_runs_1" src="https://user-images.githubusercontent.com/46538726/98453073-331c5980-2123-11eb-8cd6-88488649a45a.png">

<img width="1022" alt="trainloss_runs_2" src="https://user-images.githubusercontent.com/46538726/98453075-33b4f000-2123-11eb-9cde-5525c00b4fc3.png">

<img width="1020" alt="valacc_runs_1" src="https://user-images.githubusercontent.com/46538726/98453076-344d8680-2123-11eb-93f0-309d8ab73ed8.png">

<img width="1020" alt="valacc_runs_2" src="https://user-images.githubusercontent.com/46538726/98453078-36afe080-2123-11eb-8539-15b8947cc741.png">


### 4.3 Bar plots

<img width="1021" alt="epochtime_runs_1" src="https://user-images.githubusercontent.com/46538726/98453163-09affd80-2124-11eb-978c-d6af9ac765de.png">

<img width="1021" alt="epochtime_runs_2" src="https://user-images.githubusercontent.com/46538726/98453164-09affd80-2124-11eb-9512-7d25404bb83a.png">
