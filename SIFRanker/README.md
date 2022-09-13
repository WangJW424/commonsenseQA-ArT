# SIFRank
We implement the keyphrases extraction module of ArT based on [SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model](https://ieeexplore.ieee.org/document/8954611). The original source code is at https://github.com/sunyilgdx/SIFRank.

## Environment
```
Python 3.8
nltk 3.5
StanfordCoreNLP 3.9.1.1
torch 1.7.1+cu110
allennlp 2.6.0
```

## Download
* ELMo ``elmo_2x4096_512_2048cnn_2xhighway_options.json`` and ``elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5`` from [here](https://allennlp.org/elmo) , and save it to the ``elmo/`` directory
* StanfordCoreNLP ``stanford-corenlp-4.2.2`` from [here](https://huggingface.co/stanfordnlp/CoreNLP/resolve/main/stanford-corenlp-latest.zip), and save it to to the ``corenlp/`` directory

## Usage
We provide three shell scripts: *ke_copa.sh*, *ke_socialiqa.sh*, and *ke_rocstory.sh* to conduct keyphrases extraction with our parameters for datasets: COPA, SocialIQA and ROCStory (also named as SCT in our paper).

You can directly run these scripts, e.g. 
```
sh run_copa.sh
```