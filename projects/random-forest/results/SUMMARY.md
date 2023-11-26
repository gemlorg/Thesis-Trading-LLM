# Results

### Visualisation 
[random-forest/src/visual.ipynb](../src/visual.ipynb)

### Report
So far random forests were tested only on the simple sales data.
The following options were tried

- `num_lags: [1, 5, 10, 13, 25, 40, 50]`
- `n_estimators: [20, 50, 100]`
- `max_features: [2, 4, 8]`
- `criterion: ["gini", "entropy", "log_loss"]`

As can be seen in the visualisation, accuracy of the models ranged from 0.749 to 0.827. The main influence seems to be the lag number - the optimal being around 10. Then slightly better are those with  max_features = 4, and number of trees >= 50. The criterion doesn't seem to play a significant role.
