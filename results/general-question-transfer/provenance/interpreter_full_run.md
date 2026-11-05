# Full-run generation interpreter (DEVIATIONS 2026-11-03)

Built 2026-11-03 with `uv` 0.10.10, outside the repository, at `~/venvs/pva-torch260-tf4523`:

```
uv venv -p 3.10 ~/venvs/pva-torch260-tf4523
VIRTUAL_ENV=~/venvs/pva-torch260-tf4523 uv pip install --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple \
    "torch==2.6.0+cu124" "transformers==4.52.3" "accelerate==1.7.0" \
    pytest numpy safetensors huggingface_hub pandas scipy tqdm requests
VIRTUAL_ENV=~/venvs/pva-torch260-tf4523 uv pip install matplotlib   # only for one figure test
```

The full `pip freeze` is `interpreter_full_run_pip_freeze.txt` (next to this file).

## Versions and the CUDA check

```
python 3.10.12
torch 2.6.0+cu124 cuda 12.4 cudnn 90100
transformers 4.52.3
cuda:0 NVIDIA A40 bf16 matmul ok (8, 6)
cuda:1 NVIDIA A40 bf16 matmul ok (8, 6)
```

NVIDIA driver 580.173.02. Generation loads the model with `from_pretrained` and `.to("cuda:N")`,
as the pilot did (`accelerate` is installed but not used for placement).

## Tests under this interpreter (2026-11-03)

`python -m pytest -q` from the repository root: all suites pass (see the stage report for the
count on the final commit); `audit/tests/test_steer.py` and `lab/tests/test_general_questions.py`
(the steering hook and the generation library): 91 passed.
