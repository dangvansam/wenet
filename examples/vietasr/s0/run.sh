#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.

. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

stage=5
stop_stage=5

HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2023

# Optional train_config
# 1. conf/train_transformer_large.yaml: Standard transformer
train_config=conf/train_u2++_efficonformer_v2_bpe2000.yaml
checkpoint=/home/andrew/wenet/examples/vietasr/s0/exp_data_asr_tts_new_lower/train_u2++_efficonformer_v2_bpe2000/epoch_18.pt
num_workers=8
do_delta=false

# bpemode (unigram or bpe)
nbpe=2000
bpemode=bpe

# data
data_url=
# use your own data path
datadir=
# wav data dir
wave_data=data_asr_tts_new_lower
data_type=raw

dir=exp_${wave_data}/train_u2++_efficonformer_v2_bpe2000
tensorboard_dir=tensorboard

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
# maybe you can try to adjust it if you can not get close results as README.md
average_num=5
decode_modes="attention_rescoring ctc_greedy_search ctc_prefix_beam_search attention"

set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
recog_set="common_voice_17_0_test vivos_test"

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#   echo "stage -1: Data Download"
#   for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
#     local/download_and_untar.sh ${datadir} ${data_url} ${part}
#   done
# fi

# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#   ### Task dependent. You have to make data the following preparation part by yourself.
#   ### But you can utilize Kaldi recipes in most cases
#   echo "stage 0: Data preparation"
#   for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
#     # use underscore-separated names in data directories.
#     local/data_prep_torchaudio.sh ${datadir}/LibriSpeech/${part} $wave_data/${part//-/_}
#   done
# fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 1: Feature Generation"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp $wave_data/$train_set/wav.scp \
    --out_cmvn $wave_data/$train_set/global_cmvn

fi

dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p $wave_data/lang_char/

  if [ -f "${bpemodel}.model" ]; then
    echo "BPE model already exists at ${bpemodel}.model. Skipping training."
  else
    echo "Training BPE model..."
    # Combine train and dev text files for training the BPE model
    cut -f 2- -d" " $wave_data/${train_set}/text $wave_data/${dev_set}/text > $wave_data/lang_char/train_text.txt
    cat $wave_data/lang_char/train_text.txt > $wave_data/lang_char/input.txt
    head -n 3000000 /home/andrew/data/lm_text_052023.txt >> $wave_data/lang_char/input.txt
    tools/spm_train --input=$wave_data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  fi
    # Create dictionary by reading vocab file line-by-line and assigning consecutive IDs
    # Start assigning IDs from 3 (since 0, 1, 2 are already used for special tokens)
    echo "<blank> 0" > ${dict}  # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict}   # <unk> must be 1
    echo "<sos/eos> 2" >> ${dict} # <sos/eos>
    awk '{print $1 " " NR-1}' $bpemodel.vocab | tail -n +4 >> ${dict}  # Skip the first 3 lines for <unk>, <sos/eos>
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Prepare wenet required data
  echo "Prepare data, prepare required format"
  for x in $dev_set ${recog_set} $train_set ; do
    tools/make_raw_list.py $wave_data/$x/wav.scp $wave_data/$x/text $wave_data/$x/data.list
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=$job_id --rdzv_backend="c10d" \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type ${data_type} \
      --train_data $wave_data/$train_set/data.list \
      --cv_data $wave_data/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  # TODO, Add model average here
  mkdir -p $dir/test
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    if [ -f $decode_checkpoint ]; then
      echo "decode_checkpoint $decode_checkpoint already exist, removing..."
      rm -vf $decode_checkpoint
    fi
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  for test in $recog_set; do
    result_dir=$dir/${test}
    python wenet/bin/recognize.py --gpu 0 \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type raw \
      --test_data $wave_data/$test/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 20 \
      --batch_size 16 \
      --blank_penalty 0.0 \
      --result_dir $result_dir \
      --ctc_weight $ctc_weight \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

    for mode in $decode_modes; do
      test_dir=$result_dir/$mode
      decode_checkpoint_name=$(basename $decode_checkpoint)
      current_datetime=$(date +"%y%m%d%H%M")
      python tools/compute-wer.py --char=0 --v=0 \
        $wave_data/$test/text $test_dir/text > $test_dir/wer.avg_${average_num}.${current_datetime}
    done
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi

# Optionally, you can add LM and test it with runtime.
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  lm=data/local/lm
  lexicon=data/local/dict/lexicon.txt
  mkdir -p $lm
  mkdir -p data/local/dict

  # 7.1 Download & format LM
  which_lm=3-gram.pruned.1e-7.arpa.gz
  if [ ! -e ${lm}/${which_lm} ]; then
    wget http://www.openslr.org/resources/11/${which_lm} -P ${lm}
  fi
  echo "unzip lm($which_lm)..."
  gunzip -k ${lm}/${which_lm} -c > ${lm}/lm.arpa
  echo "Lm saved as ${lm}/lm.arpa"

  # 7.2 Prepare dict
  unit_file=$dict
  bpemodel=$bpemodel
  # use $dir/words.txt (unit_file) and $dir/train_960_unigram5000 (bpemodel)
  # if you download pretrained librispeech conformer model
  cp $unit_file data/local/dict/units.txt
  if [ ! -e ${lm}/librispeech-lexicon.txt ]; then
    wget http://www.openslr.org/resources/11/librispeech-lexicon.txt -P ${lm}
  fi
  echo "build lexicon..."
  tools/fst/prepare_dict.py $unit_file ${lm}/librispeech-lexicon.txt \
    $lexicon $bpemodel.model
  echo "lexicon saved as '$lexicon'"

  # 7.3 Build decoding TLG
  tools/fst/compile_lexicon_token_fst.sh \
     data/local/dict data/local/tmp data/local/lang
  tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;

  # 7.4 Decoding with runtime
  fst_dir=data/lang_test
  for test in ${recog_set}; do
    ./tools/decode.sh --nj 6 \
      --beam 10.0 --lattice_beam 5 --max_active 7000 --blank_skip_thresh 0.98 \
      --ctc_weight 0.5 --rescoring_weight 1.0 --acoustic_scale 1.2 \
      --fst_path $fst_dir/TLG.fst \
      --dict_path $fst_dir/words.txt \
      data/$test/wav.scp data/$test/text $dir/final.zip $fst_dir/units.txt \
      $dir/lm_with_runtime_${test}
    tail $dir/lm_with_runtime_${test}/wer
  done
fi
