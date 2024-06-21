# Pretrained Speech Encoders and Efficient Fine-tuning Methods for Speech Translation: UPC at IWSLT 2022

The paper is available [here](https://aclanthology.org/2022.iwslt-1.23/).


## Contents

- [Environment Setup](#environment-setup)
- [Pretrained Models](#pretrained-models)
- [Data](#data)
- [Knowledge Distillation](#knowledge-distillation)
- [Training](#training)
- [MuST-C Evaluation](#evaluation-on-must-c-known-segmentation)

## Environment Setup
- OS : Linux

Set the environment variables:

```bash
export IWSLT_ROOT=...                          # where to clone this repo
```

Clone this repository to `$IWSLT_ROOT`:

```bash
git clone --recursive https://github.com/lltlien/iwslt2022-speech-translation-system.git ${IWSLT_ROOT}
```

Install Miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
source ~/miniconda3/bin/activate
conda init
cd ${IWSLT_ROOT}
```
To activate conda's base environment:
```bash
eval "$(/home/your_path/miniconda3/bin/conda shell.bash hook)"
```
Create a conda environment using the `environment.yml` file, activate it and install Fairseq:
```bash
conda env create -f ${IWSLT_ROOT}/environment.yml && \
conda activate iwslt22 && \
pip install --editable ${IWSLT_ROOT}/fairseq/
pip install jiwer bitarray
pip install omegaconf hydra-core
pip install torch==1.11.0+cu115 torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
```

Install NVIDIA's [apex](https://github.com/NVIDIA/apex) library for faster training with fp16 precision:

```bash
git clone https://github.com/NVIDIA/apex && cd apex
```

Run:
```bash
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
--global-option="--deprecated_fused_adam" --global-option="--xentropy" \
--global-option="--fast_multihead_attn" ./
```
or
```bash
pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
--global-option="--deprecated_fused_adam" --global-option="--xentropy" \
--global-option="--fast_multihead_attn" ./
```

## Pretrained models

In this project we use pre-trained speech encoders and text decoders.\
Download HuBERT, wav2vec2.0 and mBART models to `$MODELS_ROOT`:

```bash
export MODELS_ROOT=...

mkdir -p ${MODELS_ROOT}/{wav2vec,hubert,mbart}
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt -P ${MODELS_ROOT}/wav2vec
wget https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt -P ${MODELS_ROOT}/hubert
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.1n.tar.gz -O - | \
tar -xz --strip-components 1 -C ${MODELS_ROOT}/mbart
```

## Data

### Download

Set the data environment variables:

```bash
export MUSTC_ROOT=...           # where to download MuST-C v2                  
```
Download MuST-C v2 en-pt to `$MUSTC_ROOT`:\
The dataset is available [here](https://mt.fbk.eu/must-c-release-v1-0/). 

### Data Preparation


Prepare the tsvs for the MuST-C data: \
We do this process for both the ASR and ST tasks and for all language pairs. \
We only prepare the tsvs and do not learn a vocabulary since we will reuse the one from mBART50.

```bash
# PYTHONPATH
export PYTHONPATH=/home/your_path/fairseq:/home/your_path:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```
```bash
# MuST-C (en-pt)
for task in {asr,st}; do
    python ${IWSLT_ROOT}scripts/data_prep/prep_mustc_data.py \
    --data-root ${MUSTC_ROOT} --task $task --use-audio-input --only-manifest --append-lang-id
done
```

### Data Filtering

Do ASR inference on the "train" sets using a pre-trained wav2vec 2.0 model and save the results at `$FILTER_ROOT`:

```bash
export FILTER_ROOT=...

# MuST-C
# add batch_size if you dont have gpu
python ${IWSLT_ROOT}/scripts/filtering/asr_inference.py \
    --tsv_path ${MUSTC_ROOT}/en-pt/train_asr.tsv \
    -o ${FILTERING_ROOT}/MUSTC_v2.0/en-pt/ \
    --batch_size 8
# example result : *Macro-averaged WER = 0.1962*
```

Apply ASR-based and text-based filtering to create clean versions of the train sets:

```bash
# MuST-C
python ${IWSLT_ROOT}/scripts/filtering/filter_tsv.py \
    -tsv ${MUSTC_ROOT}/en-pt/train_st.tsv \
    -p ${FILTERING_ROOT}/MUSTC_v2.0/en-pt/train_asr_wer_results.json \
    -o ${MUSTC_ROOT}/en-pt \
    -par -wer 0.75
```

### Combine the different datasets into en-pt directories

Set up the path:

```bash
export DATA_ROOT=...
mkdir -p ${DATA_ROOT}/en-pt
```

Make symbolink links:

```bash
# from MuST-C
for task in {asr,st}; do
    ln -s ${MUSTC_ROOT}/en-pt/train_${task}_filtered.tsv ${DATA_ROOT}/en-pt/train_${task}_mustc.tsv
    ln -s ${MUSTC_ROOT}/en-pt/dev_${task}.tsv ${DATA_ROOT}/en-pt/dev_${task}_mustc.tsv
    ln -s ${MUSTC_ROOT}/en-pt/tst-COMMON_${task}.tsv ${DATA_ROOT}/en-pt/tst-COMMON_${task}_mustc.tsv
done
```

## Knowledge Distillation

We are using knowledge distillation for en-de with mBART50 as the teacher. \
Extract the top-k probabilities offline before training and save them at `$KD_ROOT`:

```bash
export KD_ROOT=...
mkdir -p ${KD_ROOT}/en-pt

# add batch-size if you dont have gpu
for asr_tsv_file in ${DATA_ROOT}/en-pt/train*asr*.tsv; do
    st_tsv_file=$(echo $asr_tsv_file | sed "s/_asr_/_st_/g")
    kd_subdir=$(basename "$st_tsv_file" .tsv)
    python ${IWSLT_ROOT}/scripts/knowledge_distillation/extract_topk_logits.py \
    --path-to-asr-tsv $asr_tsv_file --path-to-st-tsv $st_tsv_file --path-to-output ${KD_ROOT}/en-pt/${kd_subdir} --batch-size 8
done
```
## Training

Set up the path to save the training outputs:

```bash
export SAVE_DIR=...
```

All our experiments can be found at `${IWSLT_ROOT}/config`.\
To train an experiment called `EXP_NAME`, run the following command:

```bash
EXP_NAME=...     # one of the available experiments

# to adjust the update_freq according to the number of available GPUs
base_update_freq=24
n_gpus=$(nvidia-smi --list-gpus | wc -l)

fairseq-hydra-train \
    --config-dir ${IWSLT_ROOT}/config/ \
    --config-name ${EXP_NAME}.yaml \
    dataset.num_workers=$(($(eval nproc) / 2)) \
    optimization.update_freq=[$(( $base_update_freq / $n_gpus ))]
```

## Evaluation on MuST-C (known segmentation)

To generate the translations for the MuST-C dev or tst-COMMON sets run the following command:

```bash
EXP_NAME=...    # one of the trained experiments
CKPT_NAME=...   # the name of a .pt file
SUBSET=...      # dev_mustc or tst-COMMON_mustc
TGT_LANG=...    # de, zh or ja

${IWSLT_ROOT}/scripts/generate.sh $EXP_NAME $CKPT_NAME $SUBSET $TGT_LANG 
```
