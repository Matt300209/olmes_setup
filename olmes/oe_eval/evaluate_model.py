import argparse
import subprocess
import os
import torch
import json
from statistics import mean
from oe_eval.utils import save_jsonl

RAW_DUMP_PATH = "/home/mdieckmann/olmes_eval/throughput" # this should be the same as in vllm/metrics
OUTPUT_DIR = "/home/mdieckmann/olmes_eval"
BATCHSIZES = [256,64,16,4,1] # reversed
MAX_LENGTH = 4096 # gpu utilisation to 0.8, otherwise 2048 and truncate context
TASK_LIST_DEV = [
        "gsm8k::tulu",
        "drop::llama3",
        #"minerva_math::tulu",
        "codex_humaneval::tulu",
        "codex_humanevalplus::tulu",
        "ifeval::tulu",
        "popqa::tulu",
        #"mmlu:mc::tulu",
        #"alpaca_eval_v2::tulu",
        #"bbh:cot-v1::tulu",
        "truthfulqa::tulu",
        ]
TASK_LIST_UNSEEN = [
        "agi_eval_english:0shot_cot::tulu3",
        "gpqa:0shot_cot::tulu3",
        #"mmlu_pro:0shot_cot::tulu3",
        #"deepmind_math:0shot_cot::tulu3",
        "bigcodebench_hard::tulu",
        ]
_parser = argparse.ArgumentParser()
_parser.add_argument("--model-path", type=str, help="local modelpath")
_parser.add_argument("--name", type=str, help="given name")


def evaluate_model(args_dict, batchsizes):
    # create model output path
    subprocess.run(["bash", "-c", f"""mkdir -p {OUTPUT_DIR}/{args_dict["name"]}"""], check=True)
    for batchsize in batchsizes:
        task_list = TASK_LIST_DEV + TASK_LIST_UNSEEN
        # create output path for each batchsize
        subprocess.run(["bash", "-c", f"""mkdir -p {OUTPUT_DIR}/{args_dict["name"]}/{batchsize}"""], check=True)
        for task in task_list:
            task_name = task.split(':')[0]
            # create output path for each task
            subprocess.run(["bash", "-c", f"""mkdir -p {OUTPUT_DIR}/{args_dict["name"]}/{batchsize}"""], check=True)

            dir_path = f"{OUTPUT_DIR}/{args_dict["name"]}/{batchsize}/{task_name}"

            command = f"""
                CUDA_VISIBLE_DEVICES=0 python3 launch.py \
                --model-args '{{"model_path": "{args_dict["model_path"]}", "max_length" : {MAX_LENGTH}}}' \
                --task {task} \
                --output-dir {dir_path} \
                --limit {batchsizes[0]} \
                --batch-size {batchsize} \
                --model-type vllm
                """
            command_last = f"""
                CUDA_VISIBLE_DEVICES=0 python3 launch.py \
                --model-args '{{"model_path": "{args_dict["model_path"]}", "max_length" : {MAX_LENGTH}}}' \
                --task {task} \
                --output-dir {dir_path} \
                --batch-size {batchsize} \
                --model-type vllm
                """
            if batchsize == batchsizes[0]:
                    # get accurate results
                    result = subprocess.run(
                        command,
                        shell=True,
                        check=True,                    
                        text=True            
                    )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    check=True,                    
                    text=True            
                )
            # rename and move the raw throughputfile
            subprocess.run(["bash", "-c", f"""mv {RAW_DUMP_PATH}/unnamed_raw.jsonl {dir_path}/throughput.jsonl"""], check=True)

def get_throughput(args_dict, batchsizes):
    # create one file for each batchsize containing the averaged max throughputs, mean throughput, gpu mem util, and total e2e time
    for batchsize in batchsizes:
        task_list = TASK_LIST_DEV + TASK_LIST_UNSEEN
        throughput_dict_list = []
        for task in task_list:
            task_name = task.split(':')[0]
            full_data = []
            with open(f'{OUTPUT_DIR}/{args_dict["name"]}/{batchsize}/{task_name}/throughput.jsonl', 'r') as f:
                for line in f:
                    l = json.loads(line)
                    full_data.append(l)
            # there might not be batchsize samples in the data
            max_requests = max(item["running_requests"] for item in full_data)
            max_lines = [line for line in full_data if line["running_requests"] == max_requests]
            max_throughput = mean([line["gen_throughput"] for line in max_lines])

            # calculate mean only at generation time
            mean_throughput = mean([line["gen_throughput"] for line in full_data if line["prompt_throughput"] == 0.0])

            max_cache_util = mean([line["cache_usage"] for line in max_lines])

            e2e_list = [d["e2e"][0] if d["running_requests"] == 0 and d["e2e"] else None for d in full_data]
            e2e_list = [i for i in e2e_list if i]

            total_e2e = sum(e2e_list)

            # for avg_e2e_per_batch remove the last elem if possible since it contains less samples
            if len(e2e_list) > 1:
                e2e_list = e2e_list[:-1]
            avg_e2e_per_batch = mean(e2e_list)


            throughput_dict_list.append({"task" : task, "actual_batch_size": max_requests, "max_throughput": max_throughput, "mean_throughput": mean_throughput, "max_cache_util" : max_cache_util, "avg_e2e_per_batch": avg_e2e_per_batch, "total_e2e": total_e2e})

        save_jsonl(f'{OUTPUT_DIR}/{args_dict["name"]}/performance_bsz_{batchsize}.jsonl',throughput_dict_list)
        
                    
def get_performance(args_dict, batchsizes):
    task_list = TASK_LIST_DEV + TASK_LIST_UNSEEN
    all_metrics = []
    for task in task_list:
        task_name = task.split(':')[0]
        all = []
        with open(f'{OUTPUT_DIR}/{args_dict["name"]}/{batchsizes[0]}/{task_name}/metrics-all.jsonl', 'r') as f:
                for line in f:
                    l = json.loads(line)
                    all.append(l)
        all_metrics.append({"task" : task, "score" : all[0]["metrics"]["primary_score"]})
    save_jsonl(f'{OUTPUT_DIR}/{args_dict["name"]}/metrics.jsonl',all_metrics)

                
def main():
    # allow for more memory
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    batchsizes = BATCHSIZES
    args = _parser.parse_args()
    args_dict = vars(args)
    evaluate_model(args_dict, batchsizes)

    get_throughput(args_dict, batchsizes)
    get_performance(args_dict, batchsizes)
    

if __name__ == "__main__":
    main()


