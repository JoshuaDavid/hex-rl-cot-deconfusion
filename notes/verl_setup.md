# verl setup notes (single A100-40GB, GRPO + vLLM rollout)

Status: IN PROGRESS — this file is updated as setup proceeds.

## Environment

- Venv: `/venv/verl` (python 3.12.13, created with `uv venv /venv/verl --python 3.12`)
- Do NOT touch `/venv/main` (used by another process).
- verl source checkout: `/workspace/verl-src` (github.com/volcengine/verl @ ddfbf4ea, 2026-08-04, version 0.9.0.dev), installed editable.
- Smoke test files: `/workspace/verl-smoke/` (dataset script, reward fn, run script).

## Versions

- verl 0.9.0.dev (source, editable, `/workspace/verl-src` @ ddfbf4ea 2026-08-04)
- vllm 0.24.0 (verl's pin from `scripts/install_vllm_sglang_mcore.sh`)
- torch 2.11.0+cu130 (pinned by vllm 0.24.0), torchvision 0.26.0, torchaudio 2.11.0
- transformers 5.10.4 (verl pin: `>=5.5.3,!=5.6.0,<5.11`)
- ray 2.56.1, tensordict 0.10.0, numpy 2.x, pyarrow 25.0.0
- flash-attn 2.8.3 (prebuilt wheel `cu13torch2.10cxx11abiTRUE-cp312` from
  Dao-AILab GitHub releases — WORKS with torch 2.11.0+cu130 despite the
  torch2.10 tag; import + GPU forward verified. Source build vs torch 2.11
  actually FAILED, see pitfalls)
- liger-kernel 0.8.1
- TransferQueue 0.1.8 (required by the default v1 trainer; missing from setup.py
  install_requires — must be installed manually, it is only in requirements.txt)

## Dataset schema (parquet, one row per prompt)

Produced by `/workspace/verl-smoke/make_tiny_gsm8k.py` (mirrors
`examples/data_preprocess/gsm8k.py`). Columns:

- `data_source` (str): e.g. `"openai/gsm8k"`. Used as the key to select the
  reward function (`reward_fn_key`, default `"data_source"`).
- `prompt` (list[dict]): chat messages, e.g. `[{"role": "user", "content": "..."}]`.
  verl applies the model's chat template to this.
- `ability` (str): free-form tag (e.g. `"math"`), not load-bearing.
- `reward_model` (dict): `{"style": "rule", "ground_truth": "<answer str>"}`.
  `ground_truth` is what gets passed to the reward fn.
- `extra_info` (dict): arbitrary; passed to the reward fn as `extra_info`
  (verl also injects `num_turns` and `rollout_reward_scores` keys into it).

Optional columns: `agent_name` (selects which agent loop to run per-sample in
the v1/agent-loop rollout path).

## Custom reward function wiring

Config (this version, 0.9.0.dev — note it moved UNDER the `reward` group):

    reward.custom_reward_function.path=/abs/path/to/my_reward.py
    reward.custom_reward_function.name=compute_score      # default name
    # optional static kwargs merged into every call:
    reward.custom_reward_function.reward_kwargs.foo=bar

Legacy top-level `custom_reward_function.path=...` still works (copied into
`reward.custom_reward_function` by `verl/experimental/reward_loop/reward_loop.py`),
but prefer the `reward.`-prefixed form.

Exact call made by `verl/workers/reward_manager/naive.py` (keyword args):

    compute_score(data_source=<row's data_source>,
                  solution_str=<decoded response text>,
                  ground_truth=<row's reward_model["ground_truth"]>,
                  extra_info=<row's extra_info + num_turns + rollout_reward_scores>)

So the signature to implement is:

    def compute_score(data_source, solution_str, ground_truth, extra_info=None): ...

Return either a float, or a dict `{"score": float, ...}` (extra keys are logged
as `reward_extra_info` metrics). Async `async def` reward fns are also supported
(wrapped in `_call_with_kwargs_async`). A per-sample timeout can be set via the
reward manager's `compute_score_timeout`.

Demo file used in the smoke test: `/workspace/verl-smoke/my_reward.py`.

## Multi-turn rollout support (documented only — nothing built)

The installed verl (0.9.0.dev) has two relevant mechanisms:

1. **Agent Loop** (`verl/experimental/agent_loop/`, docs
   `docs/advance/agent_loop.rst`): the general multi-turn/agentic interface,
   works with BOTH vllm (async server mode) and sglang. In the v1 trainer
   (`trainer.use_v1=true`, the default) ALL rollouts already go through the
   agent loop (`AgentLoopManagerTQ`), with `rollout.agent.default_agent_loop=
   single_turn_agent`. Built-ins: `single_turn_agent`, `tool_agent`
   (ToolAgentLoop: model calls tools defined via `multi_turn.tool_config_path`
   or `@function_tool`-decorated fns via `multi_turn.function_tool_path`).
   Custom loops: subclass `AgentLoopBase`, implement `async run(sampling_params,
   **dataset_fields) -> AgentLoopOutput` (prompt_ids, response_ids,
   response_mask with tool/user tokens masked to 0), register via
   `rollout.agent.agent_loop_config_path` (list of `_target_` configs) and
   select per-sample with an `agent_name` dataset column. This is the natural
   path for hex-as-multi-turn (env interaction inside `run`, calling
   `LLMServerClient.generate` per turn).
2. **`rollout.multi_turn.*` config** (sglang-oriented tool rollout): yaml says
   "set rollout.name to sglang as well" for `multi_turn.enable=True`. The tool
   schema/config also feeds ToolAgentLoop in the agent-loop path.

Conclusion: multi-turn IS available with the vllm backend via Agent Loop
(status: alpha, API may change). Single-turn (one position -> one move) needs
nothing special.

## Pitfalls

- **Ray fails to start: "The current node timed out during startup" / raylet dies
  with dashboard agent "RuntimeError: can't start new thread".** Root cause: the
  container cgroup has `pids.max=2816` and ~2000 threads are already used by
  other processes (tensorboard, the other session, etc.). Fix (in run_smoke.sh):
  `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`,
  `export RAY_prestart_worker_first_driver=0`, and pass
  `+ray_kwargs.ray_init.include_dashboard=False ray_kwargs.ray_init.num_cpus=12`.
  If it recurs, `ray stop` to clear stale raylets and check
  `cat /sys/fs/cgroup/pids.current` vs `pids.max`. Even with the dashboard off,
  ray workers died with `pthread_create failed` until worker-actor counts were
  cut: `actor_rollout_ref.rollout.agent.num_workers=2 reward.num_workers=2
  transfer_queue.backend.SimpleStorage.num_data_storage_units=2` and env
  `RAY_num_server_call_thread=2`. Also `supervisorctl stop tensorboard` frees
  ~280 threads (restart it after runs if wanted). `pids.max` is host-set and
  read-only from inside the container.
- **GPU contention with the concurrent process**: another (out-of-namespace)
  process grabs ~20-32GB sporadically. A run that starts on a free GPU can
  still die mid-init with `ncclUnhandledCudaError: Failed to CUDA calloc`.
  `/workspace/verl-smoke/run_smoke_retry.sh` waits for two consecutive <10GB
  readings 45s apart, launches, and retries on failure.
- **Fastly (files.pythonhosted.org) throttled to ~12-100 KB/s from this host.**
  Default PyPI installs stall for hours. Fix: `--index-url
  https://repo.huaweicloud.com/repository/pypi/simple` (~10 MB/s aggregate).
  download.pytorch.org (CloudFront) is also fast (~2 MB/s/conn).
- vllm 0.24.0 (verl's pinned version in `scripts/install_vllm_sglang_mcore.sh`)
  pins `torch==2.11.0`. No prebuilt flash-attn wheel exists for torch 2.11
  (Dao-AILab releases stop at torch2.10/cu13). A source build
  (`FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=48`)
  ran ~45 min and FAILED with a compile error against torch 2.11/nvcc 13.0.
  Working fix: install the prebuilt `cu13torch2.10` wheel directly —
  `uv pip install "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"`
  — the torch 2.10 ABI is compatible with 2.11 (verified import + fwd pass).
- `TransferQueue==0.1.8` is required by the default v1 trainer
  (`trainer.use_v1=true`) but is only in requirements.txt, NOT in setup.py
  install_requires → `pip install -e .` alone leaves main_ppo broken
  (`ModuleNotFoundError: transfer_queue`). Install it explicitly.

## Live inference against the training policy (added 2026-08-05)

The rollout engine is an OpenAI-compatible HTTP server (vLLMHttpServer ray actor).
Find it: `ss -tlnp | grep vLLMHttpSe` (dynamic port on container IP, e.g. 172.17.0.10:45629).
- Serves the CURRENT policy (weights synced each step); model name still reads "Qwen/Qwen3-1.7B".
- Engine sleeps during the training phase of each step (free_cache_engine=True): requests
  queue until next wake. Use >=5 min client timeouts.
- Light probes only; they share rollout batch capacity.
- For policy-matched hex generation, replicate forced-close via /v1/completions two-phase.
