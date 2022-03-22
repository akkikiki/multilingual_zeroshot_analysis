# multilingual_zeroshot_analysis
This repository contains the code used for the paper [Match the Script, Adapt if Multilingual: Analyzing the Effect of Multilingual Pretraining on Cross-lingual Transferability](http://arxiv.org/abs/2203.10753).

## Prerequisites
* pytorch
* transformers
* jsonlines
* conllu (for preprocessing)

If running regression analysis:
* statsmodels
* lang2vec

## Obtaining and Preprocessing the Pretraining Data
1. Obtain CoNLL 2017 Wikipedia dump from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1989. 
   1. or `wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/English-annotated-conll17.tar` and change "English" for other languages.
2. Preprocess by obtaining the raw text by running e.g., `scripts/preprocess_en/en00.sh`
3. Downsample by running `src/scripts/downsample_train_dev.sh`

## Pretraining
1. Run `scripts/train/pretrain_en.sh` (for pretraining only on English, remember to set all necessary constants like output directory)

## Adapting or Continued Pretraining
Please refer to https://github.com/facebookresearch/XLM for the details on XLM-R, XLM-17, and XLM-100.

For XLM-R models, use the `--continued_pretraining` flag and specify the models to adapt.

For XLM-{17,100} models,
1. Run `scripts/train/adapt_xlm{17,100}.sh` 

## Fine-Tuning
Check out https://github.com/google-research/xtreme repository.
* `xtreme/scripts/train_udpos.sh` is used for POS tagging.
* `xtreme/scripts/train_panx.sh` is used for NER.
* `xtreme/scripts/train_xnli.sh` is used for NLI.

## Regression Analysis
* `src/regression/regression.py` provides an example output.

Notes:
* `src/regression/lang_data.txt` is referred from Table 5 in the appendix of the [XTREME paper](https://arxiv.org/pdf/2003.11080.pdf).

## Reference 

```
@inproceedings{fujinuma2022match,
    title = "Match the Script, Adapt if Multilingual: Analyzing the Effect of Multilingual Pretraining on Cross-lingual Transferability",
    author = "Yoshinari Fujinuma and Jordan Boyd-Graber and Katharina Kann",
    booktitle = "Proceedings of the Association for Computational Linguistics (to appear)",
    year = "2022",
    url = "http://arxiv.org/abs/2203.10753",
}
```
