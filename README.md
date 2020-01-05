# Quantized BERT
A PyTorch implementation of different Quantization Methods such as Incremental Network Quantization or Alternating Multibit Quantization to quantize the original BERT Model to 8 or less-than-8 bits.

Our implementation is built upon a fork from [INQ-pytorch](https://github.com/Mxbonn/INQ-pytorch.git)

----
#### Getting Glue Datasets

NOTE: since training happens on the LEONHARD Cluster we recommend downloading the GLUE datasets directly to the cluster for convenience.

[download_glue_data.py](download_glue_data.py) is a script to download all GLUE datasets from [here](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py), it is also included in our repo for convenience.

Either copy the file from that URL or use
[download_glue_data.py](download_glue_data.py).

Paste the python file to wherever you want the folder glue_data to be built and execute `python download_glue_data.py`.

----
#### Installation for INQ and MBQ

This assumes the GLUE datasets are already available. (see above)

To quantize on the LEONHARD Cluster from ETH after SSH-ing into the Cluster execute

`module load python_gpu/3.7.4` to load python 3.7.4

Then either copy this folder to LEONHARD or execute on LEONHARD 
`git clone https://github.com/FredericOdermatt/INQ-pytorch.git` for convenience.

Then execute
```
git clone https://github.com/huggingface/transformers.git
python -m pip install -U --user pip setuptools virtualenv
python -m venv .inq_env --system-site-packages
```

After entering the virtual environment run
   
```
pip install -e INQ-pytorch
pip install -r INQ-pytorch/requirements.txt
pip install -e transformers
```

#### Usage

To apply quantization run run_glue_inq.py like this or use the provided [script](inq.sh)

NOTE: the below command will not run without a GPU (even on the local node on LEONHARD) and you will get the error `AssertionError: Found no NVIDIA driver on your system.` Submitting with bsub removes this error.
 ```
python ./INQ-pytorch/run_glue_inq.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $taskname \
        --mode $mode \
        --nbits $nbits \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $datadir \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8 \
        --per_gpu_train_batch_size=8 \
        --learning_rate 2e-5 \
        --num_train_epochs 9.0 \
        --save_steps 100000 \
        --output_dir $SCRATCH/quant/${taskname}${nbits}${$mode} \
        --overwrite_output_dir
```
Options:
* `taskname` = [cola,mnli,mnli-mm,qnli,rte,sst-2,sts-b,wnli,mrpc,qqp]
* `mode` = [lin,p2,mb] (linear INQ, power of 2 INQ, alternating MultiBit quantization)
* `nbits`: any number between 1 and 8
* `datadir`: directory of GLUE datasets

To quantize one task with all three quantization methods to all levels of precision we run this command.
(Note: the following submits 24 jobs to the LSF System)

```
for quant in {lin,p2,mb} 
> do 
> for bits in {1..8} 
> do
> bsub -o $SCRATCH/outputfile${quant}${bits}.out -R "rusage[mem=8164,ngpus_excl_p=1]" -J mrpc${quant}${bits}bit -W 4:00 <<< "./inq.sh mrpc $quant $bits MRPC" 
> done 
> done

```

----
#### Reproducing results from NLP-Architect (Q8BERT [[Paper]](https://arxiv.org/abs/1910.06188))

This assumes the GLUE datasets are already available. (see above)

To train on the LEONHARD Cluster from ETH after SSH-ing into the Cluster execute

`module load python_gpu/3.6.1` to load python 3.6.1

`python3.6 -m pip3 install -U --user pip setuptools virtualenv`

`python3.6 -m venv .env --system-site-packages` to create a virtual environment, the option --system-site-packages is key

After entering the virtual environment `pip install nlp-architect==0.5.1` and `pip install torchvision==0.3.0`

NOTE: make sure to install version nlp-architect==0.5.1, not the newly released version 0.5.2

To start training a task use

```
nlp_architect train transformer_glue \
    --task_name $taskname \
    --model_name_or_path bert-base-uncased \
    --model_type quant_bert \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir tmp/cola-8bit \
    --evaluate_during_training \
    --data_dir $gluedata \
    --do_lower_case \
    --overwrite_output_dir
```
* `$taskname` can take values in [cola,mnli,mnli-mm,qnli,rte,sst-2,sts-b,wnli,mrpc,qqp]
* `$gluedata` specifies the directory where the GLUE datasets are saved.

This long command can also be found in the provided [script](nlp-architect.sh).

For Job Submission to the LFR system execute
```
bsub -o $SCRATCH/test/wnlitest.out -R "rusage[mem=8164,ngpus_excl_p=1]" -J wnlitest -W 4:00 < nlp-architect.sh
```
The shortest trainings we recorded were for WNLI and STS-B 1 epoch with under 10 minutes. The longest training we recorded was for QQP 3 epoc with around 870 hours. In general all tasks except MNLI, MNLIMM, QNLI and QQP will train reasonably fast (<4h) if trained with just 1 epoch on Leonhard.

With the `--evaluate_during_training` flag the result of the fine-tuned model will be reported in eval_results.txt when training has finished.

Any trained model can be evaluated using (in this example for CoLA)

```
nlp_architect run transformer_glue \
    --model_path tmp/cola-8bit \
    --task_name cola \
    --model_type quant_bert \
    --output_dir tmp_out/cola-8bit \
    --data_dir glue_data/CoLA \
    --do_lower_case \
    --overwrite_output_dir \
    --evaluate
```

 Why nlp-architect==0.5.2 is currently not used:
 
 because of 
 `AttributeError: 'QuantizedBertLayer' object has no attribute 'is_decoder'` 
 
 ----
 #### Recorded Data
 
 Recorded scores of our data can be found [here](measurements.xlsx)
