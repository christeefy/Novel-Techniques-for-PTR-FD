# Novel Techniques for Process Topology Reconstruction and Fault Diagnosis (PTR-FD)

This repository contains the codebases for the three PTR-FD algorithms investigated in my MASc thesis. The three algorithms include:

1. Granger Causality [[1](https://www.jstor.org/stable/1912791)]
2. Granger Net [[2](https://arxiv.org/abs/1711.08160)]
3. Extended Convergent Cross-Mapping [[3](http://science.sciencemag.org/content/338/6106/496), [4](https://www.nature.com/articles/srep14750)]

## Setup
To use this repo, first clone it.

Next, install external module dependencies:
```
pip install -r requirements.txt
```


## Quickstart
A guide on how to use these functions provided in `packages` can be found using the Jupyter Notebook `Quickstart.ipynb`.

The functions are also callable through the command line (examples provided below). These functions require a csv containing only the time series variables of interest for the analysis, and the functions visualize the resulting output as a causal heatmap.

#### Granger Causality
```
python3 -m packages.granger_causality.granger_causality <path_to_csv> <max_lag> --autocausation=True --pval=0.05
```

#### Granger Net
```
python3 -m packages.granger_net.core.analysis <path_to_csv> <max_lag> --autocausation=True --epochs=3000 --initial_batch_size=32 --threshold=0.1
```

#### Extended Convergent Cross-Mapping
```
python3 -m packages.eccm.models.eccm.eccm <path_to_csv> --cross_map_lags=5 --use_all_points=True --criterion=Peak --p_val=0.05 --verbose=True
```

## References
1. Clive WJ Granger. Investigating causal relations by econometric models and cross-spectral methods. *Econometrica: Journal of the Econometric Society*, pages 424–438, 1969.
2. Alex Tank, Ian Cover, Nicholas J Foti, Ali Shojaie, and Emily B Fox. An interpretable and sparse neural network model for nonlinear granger causality discovery. *arXiv preprint arXiv:1711.08160*, 2017.
3. George Sugihara, Robert May, Hao Ye, Chih-hao Hsieh, Ethan Deyle, Michael Fogarty, and Stephan Munch. Detecting causality in complex ecosystems. *Science*, 338(6106):496–500, 2012.
4. Hao Ye, Ethan R Deyle, Luis J Gilarranz, and George Sugihara. Distinguishing time-delayed causal interactions using convergent cross mapping. *Scientific reports*, 5:14750, 2015.
