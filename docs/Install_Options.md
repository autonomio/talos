# Install Options

Before installing Talos, it is recommended to first setup and start either a python or conda environment.

#### Creating a python virtual environment
```python
virtualenv -p python3 talos_env
source talos_env/bin/activate
```

#### Creating a conda virtual environment
```python
conda create --name talos_env
conda activate talos_env
```

#### Install latest from PyPi
```python
pip install talos
```

#### Install a specific version from PyPi
```python
pip install talos==0.6.1
```

#### Upgrade installation from PyPi
```python
pip install -U --no-deps talos
```

#### Install from monthly
```python
pip install --upgrade --no-deps --force-reinstall git+https://github.com/autonomio/talos
```

#### Install from weekly
```python
pip install --upgrade --no-deps --force-reinstall git+https://github.com/autonomio/talos@dev
```

#### Install from daily
```python
pip install --upgrade --no-deps --force-reinstall git+https://github.com/autonomio/talos@daily-dev
```
