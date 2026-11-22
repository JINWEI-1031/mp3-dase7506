# Reproduce the 197/240 candidate

The recorded 197 result was obtained with the frozen controller on Windows CPU.
The decision modules below are unchanged; `submission.py` is a new portable entry.
Windows cold reproduction of this entry exactly matched all 240 original SQL queries and tool traces. Linux cold reproduction completed at 197/240: all 240 final SQL strings match; one intermediate floating-point result differs by about 3e-14. See evidence/linux-reproduction/.
Do not describe the packaging static checks as a fresh 197 prediction.

Install Python 3.11 and requirements.txt. Use the unchanged course starter package externally.
Install the CPU wheel first: `python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu`, then install requirements.txt.

Provision Qwen/Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca before going offline.
Retain the revision as the real base snapshot directory name after Path.resolve(); if an existing directory symlink resolves to a folder named model, create a real revision-named directory with file symlinks to the unchanged original files. Unzip mp3-197-adapter.zip separately.

From the extracted source directory, with four CPUs and a 16 GiB process-tree memory limit enforced externally:

```bash
export PYTHONPATH="$PWD:/absolute/path/to/MP3_SQL_student_package"
export MP3_BASE_MODEL=/absolute/path/to/c1899de289a04d12100db370d81485cdf75e47ca
export MP3_ADAPTER_DIR=/absolute/path/to/adapter
export MP3_LOG_DIR=/absolute/path/to/fresh-model-logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m mp3_sql.predict --split test --method "$PWD/submission.py" --workers 1 --output fresh-test.json
python -m mp3_sql.score --split test --predictions fresh-test.json --workers 4 --output fresh-test-score.json
```

Use a 7200-second prediction timeout and a 180-second score timeout. For example, use a Linux cgroup/systemd scope with CPU affinity and MemoryMax=16G around each command. The agent uses no benchmark file access; tools/scorer access their own external bank.

`outputs/test.json` and `outputs/test-score.json` are the byte-preserved original outputs, for submission and comparison only. They are not read by the agent. Compare fresh SQL and tool traces with those outputs after inference ends.

Source files with historical step names are runtime dependencies of this one final controller. No other model checkpoints, competing runs, dataset, base weights, virtual environments or caches are included.
