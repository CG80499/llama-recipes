export MODEL="meta-llama/Meta-Llama-3.1-70B"
conda create --name llama_ft python=3.10
conda activate llama_ft
conda install -c anaconda cudatoolkit
pip install -r requirements.txt
python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name "meta-llama/Meta-Llama-3.1-8B" --output_dir llama-8b-paper-summary-less-verbose --dataset paper_summary_dataset

# Multi-gpu
#torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name "meta-llama/Meta-Llama-3.1-70B" --output_dir llama-70b-paper-summary-less-verbose --dataset paper_summary_dataset