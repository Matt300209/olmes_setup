# olmes_setup
olmes evaluation setup modified for measuring performance metrics

Steps for setting up:
make sure that vllm v0 is active

git clone https://github.com/Matt300209/olmes_setup.git
cd olmes_setup
cd olmes
pip install -e .[gpu]

open libs/vllm/engine/metrics.py and change RAW_DUMP_PATH to any specified path (this will not be the output)

back in olmes_setup execute:

pip uninstall lm_eval -y
cp -r ./libs/lm_eval .../python3.12/site-packages/

pip uninstall vllm -y
cp -r ./libs/vllm .../python3.12/site-packages/

cd olmes/oe_eval/
Change the global variables in evaluate_model.py

Execute:

python3 evaluate_model.py --model-path ... --name ... 

