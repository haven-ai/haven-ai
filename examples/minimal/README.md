## MNIST EXAMPLE


### 1. Train mlp on mnist

On the workstation,
```
python trainval.py -e mnist -r 1 -sb <savedir_base>
```

On the orkestrator

```
python trainval.py -e mnist -r 1 -sb <savedir_base> -j 1
```

`<savedir_base>` is where the results are saved.

### 2.  Visualize

Create the jupyter file as follows
```
python create_jupyter.py
```
