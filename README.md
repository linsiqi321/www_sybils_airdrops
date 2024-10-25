# www_sybils_airdrops

**This repository is an official implementation of** *Catching Sybils in Web3 Airdrops: Unveiling and Detecting Sybil Addresses at Scale*

# Dataset

The dataset is saved in the **data** directory, including **invited data**, **transactions data**, **social activity data** and **labeled deeplearning data**.

# Usage

**Clone the repo, change directory to the root of the repo directory and run:**

- Extract features with invited data

  ```{python}
  python invited_data_analysis.py
  python invited_data_graph.py
  ```

- Extract features with transactions data

  ```{python}
  python transactions_data_analysis
  python transactions_data_graph.py
  ```

- Merge all the features extracted before

  ```python
  python feature_merge.py
  ```

- Use Random Forests for Deep Learning

  ```python
  python deeplearning.py
  ```
