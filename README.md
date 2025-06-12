# A Deep Dive into MAMBA and State Space Models for Long-Sequence Modeling

This repository accompanies the research article we wrote about State Space Models and MAMBA.

# Setup

### Clone the repo
```sh
git clone https://github.com/mirkosprojects/mamba-deep-dive.git
```

### Filter notebook output
Set up a filter to clear outputs before committing changes (See [here](https://gist.github.com/33eyes/431e3d432f73371509d176d0dfb95b6e))
```sh
git config filter.strip-notebook-output.clean 'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR'
```

### Remove filter temporarily
If you want to commit output for a jupyter notebook, stage the file with the following command, then commit as usual.
```sh
git -c filter.strip-notebook-output.clean= add path/to/a/notebook
```