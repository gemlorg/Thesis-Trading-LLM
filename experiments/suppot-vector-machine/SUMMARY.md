# Results

### Results Picture 
[support-vector-machine/results/svm_models_results.png](./svm_models_results.png)

### Report
There were overall nine models tried:
- three kernels: `rbf`, `poly`, `sigmoid`
- one gamma: `scale`
- three C values: `0.1`, `1.0`, `10.0`

As can be seen in the above-linked picture, the `poly` kernel works much better than both `rbf` and `sigmoid`, which both work equally badly. Overall, though, the statistics for every model are terrible. The value of `C` has very little impact and only on `rbf` kernel.
    
