# 🥈 Code for "Topic ↔ Content Matching" [Kaggle competition](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394955) (28th place solution)
For a detailed explanation of the approach see WRITEUP.md.
Two stages: **Bi-Encoder candidate generation** and topic resolution via **Cross-Encoder**.

The task for this competition is to match educational content to specific topics. Topics are organized in a hierarchical tree, and at test time the model will have to match to an additional 10,000 new unseen topics, and of course the contents are new as well.

Order of magnitude for topics is 100k distinct topics, and the amount of predictions to be made at test time under the constraints of the Kaggle environment is approximately 50k.

## Download data
## Setup environment
Re-create conda environment
```
conda env create -f environment.yml
```

The models were trained on cloud GPU VM environments, and the environment file may have to be adapted.

## Experiment tracking via neptune.ai
If you have have a [neptune.ai](https://neptune.ai) account, provide your API token in `NEPTUNE_API_TOKEN`.

## Train
Adapt `config.py` to point to your data environment, choose hyperparameters or leave default.

- Train biencoder with `train_bienc.py --experiment_name my-bienc`.
- Generate training data for crossencoder via
```
python gen_cross_data.py            \
    --bienc_path out/silver/bienc   \
    --tokenizer_path out/silver/tokenizer
```
- Train final model  via
```
python train_cross.py               \
    --df cross/silver.csv           \
    --cross_num_cands 50            \
    --experiment_name cross
```

## Inference
Run `main.py` for local inference.
To run in Kaggle environment see the submission [notebook](https://www.kaggle.com/code/vmorelli/submit-cross-post?scriptVersionId=122154772).
