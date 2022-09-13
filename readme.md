# Feedback Prize - Predicting Effective Arguments

My pipeline for the [Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness/) competition on Kaggle.

CV and LB scores for my single model experiments can be found [here](https://docs.google.com/spreadsheets/d/1gKrqIm8tjEO4KzofWHCTNN-p-UZ-h2NIMsmfKV5eqBY/edit?usp=sharing).

This pipeline gives good single model scores, but ensembling did not improve scores at all. Reasons for this, and other learnings from the competition are compiled towards the end of the readme.

## Leaderboard Scores:
| Model                                                        | Weights                   | Public  | Private |
| ------------------------------------------------------------ |:-------------------------:|:-------:| -------:|
| DeBERTaV3Large + RoBERTaLarge + DeBERTaV3Base + RoBERTaBase  | 0.6, 0.196, 0.075, 0.128  | 0.61263 | 0.61258 |
| DeBERTaV3Large + RoBERTaLarge + DeBERTaV3Base + RoBERTaBase  | 0.6, 0.32, 0.16, 0.19     | 0.61493 | 0.61507 |
| DeBERTaV3Large + RoBERTaLarge + DeBERTaV3Base + RoBERTaBase  | 0.454, 0.248, 0.144, 0.15 | 0.61479 | 0.61474 |
| DeBERTaV3Large + RoBERTaLarge + DeBERTaV3Base + RoBERTaBase  | 0.7, 0.0, 0.0, 0.3        | 0.61383 | 0.61393 |
| DeBERTaV3Large + RoBERTaLarge + DeBERTaV3Base + RoBERTaBase  | 0.6, 0.2, 0.1, 0.1        | 0.61268 | 0.61254 |
| DeBERTaV3Large + RoBERTaLarge + DeBERTaV3Base + RoBERTaBase  | 0.25, 0.25, 0.25, 0.25    | 0.62021 | 0.62012 |

## Ensembling
Kaggle Notebook where I have tried different experiments for ensembling can be found [here](https://www.kaggle.com/code/kevinmathewt/feedbackpea-find-weights).

I largely used SciPy's [optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html), which gives you functions for minimizing objective functions. Now you can use this to find optimal weights for out-of-fold predictions from each model.

## Training
Training configurations can be set in `config.yaml` (Pre-training using MLM configurations present towards the end)

1. Before running training or pre-training, we need some basic setup - for this refer to `setup/linux_setup.sh`.

2. Once setup is complete training can be run using `setup/linux_run.sh`.
   * Model architecture can be set using the `model_name` field in `config.yaml`. This field accepts models available on HuggingFace.
     * Other configurations such as which pooler to use (this pipeline has several implementations; check `src/model.py`), multi-dropout etc. can also be configured in `config.yaml`.
   * This will automatically create the folds, create a Kaggle Dataset, and save the trained weights to the Dataset (refer to `setup/save_weights.py` to see how this exactly works). 
   * Do add your Kaggle Username and Key to the script before runnning. 
   * Also for training from a pretrained model, add the Kaggle Dataset containing the pretrained model weights in the `setup/linux_run.sh` script, and also update the `use_pretrained` field in `config.yaml` to `true`.

3. For pre-training using Masked Language Modelling (MLM), run `setup/linux_mlm_run.sh`.
   * Pipeline for MLM was largely adapted from the the [MLM no-trainer scipt](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py) from the HuggingFace library, with certain modifications (check `src/mlm/mlm_no_trainer.py`)


## Learnings
Considering I got the silver medal in my last NLP Kaggle competition, I did not do well in this competition (Top 39% finish). Here are my learnings:
1. Focusing only on single models.
   * Hearing some of the top Kaggle performers talking about focusing on single models, and leaving ensembling to the last 1-2 weeks, I decided to attempt adopting the same philosophy.
   * Now I feel having some early experiments at ensembling gives you a better understanding of how much ensembling can contribute.
     * And this contribution will be different for different ensembles (for e.g. in [this notebook](https://www.kaggle.com/code/mountpotatoq/autogluon-finetune-solutions), ensembling improved single models by almost 0.04!).
2. Lack of variety of models in training (which is why single models were good, but ensembling did not improve much).
   * In my last competition, my objective was to improve publically available notebooks, and ensemble them.
     * This provides good variation, as different implementations might do small things differently, and this variety leads to good ensembling.
   * In this competition, all models were trained using this repo, and there was very less variety (even the GPU was the same, I used A100 from [jarvislabs.com](jarvislabs.com)).
