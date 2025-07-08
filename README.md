# olmes_setup
olmes evaluation setup modified for measuring performance metrics

Steps for setting up:
make sure that vllm v0 is active
(for singularity, add: `--env "VLLM_USE_V1=0"`)

```
git clone https://github.com/Matt300209/olmes_setup.git
cd olmes_setup
cd olmes
pip install -e .[gpu]
```

open `libs/vllm/engine/metrics.py` and change `RAW_DUMP_PATH` to any specified path (this will not be the output)

Find out where the site-packages of python are. For me `/home/NAME/.local/lib/python3.12/site-packages/`
Then, back in olmes_setup:

```
pip install vllm==0.8.5

pip uninstall lm_eval -y
cp -r ./libs/lm_eval /home/NAME/.local/lib/python3.12/site-packages/

rm /home/NAME/.local/lib/python3.12/site-packages/vllm/engine/metrics.py /home/NAME/.local/lib/python3.12/site-packages/vllm/engine/llm_engine.py
cp -r ./libs/vllm/engine/metrics.py ./libs/vllm/engine/llm_engine.py /home/NAME/.local/lib/python3.12/site-packages/vllm/engine/
```


```cd olmes/oe_eval/```

Open evaluate_model.py. This is the main script, so here you can specify `BATCHSIZES`, (`RAW_DUMP_PATH` as in metrics.py), `OUTPUT_DIR` and `TASK_LIST`'s

To evaluate a model, execute:

```python3 evaluate_model.py --model-path MODEL_PATH --name NAME```

