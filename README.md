# www_sybils_airdrops

**Official code for** "Airdrop Giving Is Never Easy: Unveiling and Detecting Sybil Behavior in Web3 Airdrops".

For ease of use, we have made some changes to the original implementation in the paper.

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
