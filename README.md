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

A guide on how to use these functions provided in `packages` can be found using the Jupyter Notebook `Quickstart.ipynb`.


## References
1. Clive WJ Granger. Investigating causal relations by econometric models and cross-spectral methods. *Econometrica: Journal of the Econometric Society*, pages 424–438, 1969.
2. Alex Tank, Ian Cover, Nicholas J Foti, Ali Shojaie, and Emily B Fox. An interpretable and sparse neural network model for nonlinear granger causality discovery. *arXiv preprint arXiv:1711.08160*, 2017.
3. George Sugihara, Robert May, Hao Ye, Chih-hao Hsieh, Ethan Deyle, Michael Fogarty, and Stephan Munch. Detecting causality in complex ecosystems. *Science*, 338(6106):496–500, 2012.
4. Hao Ye, Ethan R Deyle, Luis J Gilarranz, and George Sugihara. Distinguishing time-delayed causal interactions using convergent cross mapping. *Scientific reports*, 5:14750, 2015.
