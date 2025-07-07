import argparse
import subprocess
import os
import torch
import json
from statistics import mean
from oe_eval.utils import save_jsonl

_parser = argparse.ArgumentParser()
_parser.add_argument("--model-path", type=str, help="local modelpath")
_parser.add_argument("--name", type=str, help="given name")
# create new dir

def evaluate_model(args_dict, batchsizes):
    subprocess.run(["bash", "-c", f"""mkdir -p /nas/models/nlu/mdieckmann/olmes_eval/{args_dict["name"]}"""], check=True)
    for batchsize in batchsizes:
        dir_path = f"/nas/models/nlu/mdieckmann/olmes_eval/{args_dict["name"]}/{batchsize}"
        subprocess.run(
            ["bash", "-c", f"mkdir -p {dir_path} && rm -rf {dir_path}/*"],
            check=True
        )
        # dev
        task_list_dev =  [
        "gsm8k::tulu",
        "drop::llama3",
        "minerva_math::tulu",
        "codex_humaneval::tulu",
        "codex_humanevalplus::tulu",
        "ifeval::tulu",
        "popqa::tulu",
        "mmlu:mc::tulu",
        #"alpaca_eval_v2::tulu",
        "bbh:cot-v1::tulu",
        "truthfulqa::tulu",
        ]
        # unseen
        task_list_unseen = [
        "agi_eval_english:0shot_cot::tulu3",
        "gpqa:0shot_cot::tulu3",
        "mmlu_pro:0shot_cot::tulu3",
        "deepmind_math:0shot_cot::tulu3",
        "bigcodebench_hard::tulu",
        ]
        task_list = task_list_dev + task_list_unseen
        tasks_str = " ".join(task_list)
        command = f"""
            CUDA_VISIBLE_DEVICES=0 python3 launch.py \
            --model-args '{{"model_path": "{args_dict["model_path"]}"}}' \
            --task {tasks_str} \
            --output-dir {dir_path} \
            --limit {batchsize*20} \
            --dump-file {dir_path}/throughput.jsonl \
            --batch-size {batchsize} \
            --model-type vllm
            """
        command_last = f"""
            CUDA_VISIBLE_DEVICES=0 python3 launch.py \
            --model-args '{{"model_path": "{args_dict["model_path"]}"}}' \
            --task {tasks_str} \
            --output-dir {dir_path} \
            --dump-file {dir_path}/throughput.jsonl \
            --batch-size {batchsize} \
            --model-type vllm
            """
        try:
            if batchsize == batchsizes[-1]:
                # get accurate results
                result = subprocess.run(
                    command_last,
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
        except torch.cuda.OutOfMemoryError as e:
            # out of memory
            save_jsonl(f"{dir_path}/throughput.jsonl",[{"oom":True}])
            save_jsonl(f"{dir_path}/metrics-all.jsonl",[{"oom":True}])
def get_throughput(args_dict, batchsizes):
    #name the throughput chunk_seperators
    for batchsize in batchsizes:
        full_task_dict_list = []
        with open(f'/nas/models/nlu/mdieckmann/olmes_eval/{args_dict["name"]}/{batchsize}/metrics-all.jsonl', 'r') as f:
            for line in f:
                task = json.loads(line)
                #sort out the summary lines
                if task["task_config"]["primary_metric"] != "macro" and task["task_config"]["primary_metric"] != "micro":
                    full_task_dict_list.append({"task_name" : task["task_name"]})
        with open(f'/nas/models/nlu/mdieckmann/olmes_eval/{args_dict["name"]}/{batchsize}/throughput.jsonl', 'r') as f:
            counter = -1
            for line in f:
                dict = json.loads(line)
                if "__chunk_separator__" in dict:
                    counter += 1
                    full_task_dict_list[counter]["content"] = []
                else:
                    full_task_dict_list[counter]["content"].append(dict)
        print(f"first task: {full_task_dict_list[0]["task_name"]}")
        # calculate averages, captures contlen ,total tokens cont_tokens/s and total tokens /s
        averages = []
        for dict in full_task_dict_list:
            new_dict = {}
            new_dict["batch_size"] = batchsize
            new_dict["task_name"] = dict["task_name"]
            new_dict["avg_cont_len"] = mean([d["total_cont_len"] for d in dict["content"]])
            new_dict["avg_total_tokens"] = mean([d["total_tokens"] for d in dict["content"]])
            new_dict["avg_cont_tokens_per_s"] = mean([d["cont_tokens/s"] for d in dict["content"]])
            new_dict["avg_total_tokens_per_s"] = mean([d["token/s"] for d in dict["content"]])
            averages.append(new_dict)
        save_jsonl(f'/nas/models/nlu/mdieckmann/olmes_eval/{args_dict["name"]}/results/throughput_averages_{batchsize}.jsonl',averages)
        
                    
def get_performance(args_dict, batchsizes):
    perf_results = []
    with open(f'/nas/models/nlu/mdieckmann/olmes_eval/{args_dict["name"]}/{batchsizes[-1]}/metrics-all.jsonl', 'r') as f:
            for line in f:
                task = json.loads(line)
                new_dict = {}
                new_dict["task_name"] = task["task_name"]
                new_dict["score"] = task["metrics"]["primary_score"]
                perf_results.append(new_dict)
    save_jsonl(f'/nas/models/nlu/mdieckmann/olmes_eval/{args_dict["name"]}/results/performance.jsonl',perf_results)

                
    
def main():
    # allow for more memory
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    batchsizes = [1,2,4,8,16,32,64]
    args = _parser.parse_args()
    args_dict = vars(args)
    evaluate_model(args_dict, batchsizes)

    subprocess.run(["bash", "-c", f"""mkdir -p /nas/models/nlu/mdieckmann/olmes_eval/{args_dict["name"]}/results"""], check=True)
    get_throughput(args_dict, batchsizes)
    get_performance(args_dict, batchsizes)
    

if __name__ == "__main__":
    main()


