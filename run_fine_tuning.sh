export MODEL="lmsys/vicuna-13b-v1.5-16k"
conda create --name llama_ft python=3.10
conda activate llama_ft
conda install -c anaconda cudatoolkit
python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name "lmsys/vicuna-13b-v1.5-16k" --output_dir FT-vicuna-13b-v1.5-16k --dataset paper_summary_dataset 