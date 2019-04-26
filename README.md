# The ABCD Neurocognitive Prediction Challenge 2019

This repository implements the work of Kao, Po-Yu, et al. ["Predicting Fluid Intelligence of Children using T1-weighted MR Images and a StackNet"](https://arxiv.org/abs/1904.07387) submitted to ABCD-NP-Challenge 2019. 

## Dependencies

Python 3.6

## Required Python Libraries

```numpy, pandas, matplotlib, sklearn, pystacknet, xgboost```

[Python implementation of StackNet.](https://github.com/h2oai/pystacknet])

[XGBoost](https://github.com/dmlc/xgboost)

## Run the code

1. Please change the paths accordingly from line #52 to #65.

2. ```python predict_gf.py```

You will find the predicted fluid intelligence (pred_test.csv) of testing dataset in the current directory. 