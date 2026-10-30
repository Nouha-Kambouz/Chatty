# Chatty — A Conversational Interface for a Mini-GPT Research Project

Chatty is a Streamlit application built alongside a from-scratch Mini-GPT research project. The project combines two related but distinct pieces of work:

1. **A small decoder-only Transformer trained from scratch**, used for language-modeling experiments, KV-cache validation, quantization experiments, and architecture ablation.
2. **A conversational web interface powered locally by Qwen3-0.6B**, which provides instruction-following chat responses through a local `llama.cpp` server.

The normal Chatty interface uses Qwen3 automatically. The Mini-GPT checkpoint and its inference code are retained as research artifacts; they are not presented as the default conversational assistant because the Mini-GPT was trained for next-token prediction, not instruction following.

> **Deployment status:** The current application has been tested locally on Windows. It is not ready to be deployed as a working public chatbot yet: `assistant_backend.py` currently connects to `http://127.0.0.1:8080`, which refers to the machine running the Streamlit process. On Streamlit Community Cloud, that would be the cloud container, not the developer's computer. A cloud-accessible Qwen inference backend must be configured before public deployment.

---

## Contents

- [Project overview](#project-overview)
- [System architecture](#system-architecture)
- [Chatty application features](#chatty-application-features)
- [Conversational model: Qwen3-0.6B](#conversational-model-qwen3-06b)
- [Mini-GPT research model](#mini-gpt-research-model)
- [Dataset and tokenizer](#dataset-and-tokenizer)
- [Training configuration](#training-configuration)
- [Research experiments and results](#research-experiments-and-results)
  - [Initial language-modeling baseline](#initial-language-modeling-baseline)
  - [KV-cache experiment](#kv-cache-experiment)
  - [Quantization experiments](#quantization-experiments)
  - [Controlled depth ablation](#controlled-depth-ablation)
- [Repository structure](#repository-structure)
- [Local installation and execution](#local-installation-and-execution)
- [Running the Qwen3 server](#running-the-qwen3-server)
- [Running Chatty](#running-chatty)
- [Testing](#testing)
- [GitHub setup](#github-setup)
- [Streamlit Community Cloud deployment](#streamlit-community-cloud-deployment)
- [Known limitations and next steps](#known-limitations-and-next-steps)
- [Technology stack and references](#technology-stack-and-references)

---

## Project overview

The research component investigates how a compact decoder-only Transformer behaves when trained on WikiText-103 and how changes to model depth, numerical precision, and inference implementation affect its results.

The application component provides a cleaner conversational experience using an instruction-tuned model. This separation is intentional:

- **Qwen3-0.6B** is used for ordinary Chatty conversations.
- **Mini-GPT** is retained for experiments and research analysis.

The project does not claim that the from-scratch Mini-GPT is equivalent to a modern instruction-tuned assistant.

## System architecture

### Local development architecture

```text
                         User
                           |
                           v
                  Streamlit application
                       app.py
                           |
                           v
                 assistant_backend.py
                           |
               OpenAI-compatible HTTP API
                  http://127.0.0.1:8080
                           |
                           v
                    llama.cpp server
                           |
                           v
                 Qwen3-0.6B Q4_K_M


       Separate Mini-GPT research artifacts
       -------------------------------------
       model.py       -> Transformer architecture
       tokenizer.py   -> Original BPE tokenizer
       generation.py -> Autoregressive generation
       best_model.pt  -> Trained Mini-GPT checkpoint
```

The two inference paths are separate. The current user-facing Chatty application calls Qwen3 through `assistant_backend.py`. The Mini-GPT files are preserved for research and are not required for the Qwen chat endpoint.

### Important difference between the two models

| Component | Mini-GPT | Qwen3-0.6B |
|---|---|---|
| Origin | Implemented and trained from scratch | Pretrained instruction-tuned model |
| Main objective | Next-token language modeling | Conversational instruction following |
| Size | 1,833,728 parameters | Qwen3 0.6B model family |
| Weights | Project checkpoint | QuantFactory GGUF Q4_K_M file |
| Runtime | PyTorch | `llama.cpp` |
| Chatty role | Research artifact | Default conversational assistant |

---

## Chatty application features

The Streamlit interface includes:

- Chat interface with user and assistant messages.
- Qwen3 selected automatically; no model selector is shown in the normal chat UI.
- Light and dark appearance toggle.
- Chatty wordmark and an inline SVG robot illustration.
- Conversation creation and switching within the active Streamlit session.
- Import of JSON, Markdown, and text files.
- Export of conversations as JSON or Markdown.
- Conversation title editing and deletion.
- Response-language preference: Automatic, English, French, or Arabic.
- Response-style preference: Balanced, Friendly, Professional, Concise, Detailed, or Creative.
- Custom conversation instructions.
- Sampling controls such as temperature, top-p, and maximum response tokens.
- Basic language detection for English, French, and Arabic.
- Right-to-left rendering for detected Arabic messages.

The Qwen backend receives recent conversation messages, not just the latest prompt. The backend currently keeps a bounded number of recent turns to fit the configured context window.

### Conversation storage status

Conversation state currently lives in Streamlit's session state. Import/export is available, but durable database-backed history, user accounts, and private per-user storage have not been implemented. Users should export conversations they want to keep.

### Grammar correction status

Automatic grammar correction for English, French, and Arabic is planned but is not currently implemented. Language detection and selecting a preferred response language are not the same as grammar correction.

---

## Conversational model: Qwen3-0.6B

The Chatty conversational assistant uses:

- Model repository: [`QuantFactory/Qwen3-0.6B-GGUF`](https://huggingface.co/QuantFactory/Qwen3-0.6B-GGUF)
- Quantization: `Q4_K_M`
- Runtime: `llama.cpp`
- API: OpenAI-compatible `/v1/chat/completions`
- Local address: `http://127.0.0.1:8080`
- Context size used in local testing: 1,024 tokens
- Local inference mode: CPU (`--n-gpu-layers 0`)

The local `/v1/models` response reported the GGUF model file size as 478,268,416 bytes. The exact memory needed at runtime is higher than the file size because inference also uses runtime buffers and the context cache.

The model was tested through the chat-completions endpoint with a system instruction identifying Chatty as a conversational assistant and with Qwen3 thinking disabled. A test greeting returned a natural conversational introduction.

### Local server command

Install `llama.cpp` on Windows:

```cmd
winget install llama.cpp
```

After installation, open a new terminal. If the command aliases are not on `PATH`, use the full executable path. On the tested Windows machine, the executable was located under:

```text
%LOCALAPPDATA%\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe
```

Start Qwen3:

```cmd
llama-server -hf QuantFactory/Qwen3-0.6B-GGUF:Q4_K_M --host 127.0.0.1 --port 8080 --ctx-size 1024 --threads 2 --batch-size 128 --ubatch-size 64 --n-gpu-layers 0 --jinja
```

If `llama-server` is not recognized, call the executable using its full path. Keep the server terminal open while using Chatty.

The backend checks the local model list at `/v1/models` and sends requests to `/v1/chat/completions`.

---

## Mini-GPT research model

The Mini-GPT is a compact decoder-only Transformer implemented using PyTorch. It was trained from scratch, without pretrained weights.

### Architecture configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 8,000 |
| Context length | 128 tokens |
| Embedding dimension | 128 |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Feed-forward hidden dimension | 512 |
| Dropout | 0.1 |
| Bias | Enabled |
| Parameters | 1,833,728 |

The model uses token embeddings, learned positional embeddings, causal self-attention, feed-forward sublayers, layer normalization, and a language-model output projection. The token embedding and output projection weights are tied in the reconstructed model implementation used for inference.

**Reconstruction note:** The original training checkpoint was available, but the original model source was not available in the later notebook session. The architecture source was reconstructed from checkpoint tensor shapes and the retained notebook implementation, then checked by loading the checkpoint and comparing outputs. The exported `model.py` produced a maximum absolute logit difference of `0.0` against the notebook model in the export verification test.

## Dataset and tokenizer

### Dataset

- Dataset: WikiText-103 raw (`wikitext-103-raw-v1`).
- Training split: 135,701,334 tokens.
- Validation split: 283,581 tokens.
- Test split: 324,085 tokens.

The processed token arrays were saved separately from the Chatty deployment directory. They are not required to run the conversational Qwen assistant.

### BPE tokenizer

- Tokenizer type: Byte Pair Encoding (BPE).
- Vocabulary size: 8,000.
- Special token IDs:
  - `[PAD]`: 0
  - `[UNK]`: 1
  - `[BOS]`: 2
  - `[EOS]`: 3
- Main asset: `tokenizer_assets/bpe_tokenizer.json`.
- Metadata: `tokenizer_assets/tokenizer_metadata.json`.

The exported tokenizer was loaded from the original tokenizer JSON, and a text encode/decode test reproduced the sample sentence exactly.

---

## Training configuration

The original four-layer training configuration recorded:

| Setting | Value |
|---|---:|
| Batch size | 32 |
| Maximum steps | 3,000 |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Seed in original config | 42 |
| Context length | 128 |

The saved original checkpoint contained step, model state, model configuration, and training/validation loss history. The original optimizer type and learning-rate schedule were not recorded in the available training configuration.

For the later **controlled architecture ablation**, AdamW and a constant learning rate were explicitly selected for both architectures. They should not be described as recovered facts about the original training run.

---

## Research experiments and results

The following results are from the recorded project experiments. Evaluation protocols differ between experiments; metrics from different protocols should not be compared as though they were measured on the same test sample or hardware.

### Initial language-modeling baseline

The initial full-test evaluation reported:

| Model | Test cross-entropy | Perplexity | Next-token accuracy |
|---|---:|---:|---:|
| Mini-GPT, 4 layers | 5.1073 | 165.22 | 18.90% |
| Bigram baseline | 5.6649 | 288.56 | 14.78% |

This was an initial baseline comparison and is separate from the later controlled three-seed depth-ablation experiment.

### KV-cache experiment

A KV-cache implementation was added to the Mini-GPT inference path.

Correctness checks:

- Cached and uncached model logits matched exactly in the equivalence test (maximum absolute difference: `0.0`).
- Cached and uncached greedy generation produced identical sequences in the tested generation check.

Benchmark configuration:

- Prompt length: 32 tokens.
- Generated tokens: 64.
- Warm-up runs: 3.
- Timed runs: 10.
- Device: Tesla T4.

| Metric | Uncached | Cached |
|---|---:|---:|
| Median latency | 0.1721 s | 0.1802 s |
| Throughput | 371.97 tokens/s | 355.11 tokens/s |

The measured speedup was approximately `0.955x`; the cache was about 4.75% slower in this particular small-model benchmark. The implementation passed the correctness checks, but this experiment did not demonstrate a speed improvement for the tested setup.

### Quantization experiments

The environment used for the GPU experiments was a Tesla T4 with PyTorch `2.11.0+cu130`. The INT8 dynamic quantization experiment was evaluated on CPU.

#### FP32 vs FP16

Evaluation used the first 100 non-overlapping test blocks, with 127 targets per block (12,700 targets total). Generation benchmarked 64 new tokens after warm-up, using repeated timed runs.

| Metric | FP32 | FP16 |
|---|---:|---:|
| Cross-entropy | 4.981503 | 4.981483 |
| Perplexity | 145.6932 | 145.6903 |
| Accuracy | 19.6929% | 19.6772% |
| Model tensor storage | 6.995 MB | 3.498 MB |
| Generation throughput | 385.33 tokens/s | 372.48 tokens/s |

The FP16 representation used approximately half the model tensor storage and preserved nearly the same evaluation quality. In this benchmark it was approximately 3.3% slower than FP32. Greedy generation sequences matched. These throughput values were measured on the Tesla T4.

#### Dynamic INT8 on CPU

The FP32 and INT8 CPU comparison used the same 100 test blocks and 12,700 targets.

| Metric | FP32 CPU | INT8 CPU |
|---|---:|---:|
| Cross-entropy | 4.981503 | 4.982704 |
| Perplexity | 145.6932 | 145.8682 |
| Accuracy | 19.6929% | 19.6142% |
| Serialized state-dict size | 7.013 MB | 5.752 MB |
| Generation throughput | 178.80 tokens/s | 173.26 tokens/s |

INT8 reduced serialized state-dict size by approximately 18%, with a small change in evaluation metrics. In this CPU benchmark it was approximately 3.1% slower. Greedy sequences matched in the recorded run.

**Hardware comparison warning:** Do not compare the FP16 Tesla T4 throughput directly against the INT8 CPU throughput. They were measured on different devices and are not a controlled precision-only comparison.

#### INT4 feasibility

The TorchAO INT4 API was present in the environment, but successful INT4 quantization was not achieved.

Recorded issues included:

- A dtype mismatch: the quantization operation expected float32 zero-point tensors but encountered bfloat16.
- A CPU attempt failed because the INT4 packing operator was not implemented for the CPU backend.
- A retry using an explicit floating-point zero-point domain encountered a CUDA `named symbol not found` error in the environment.

INT4 is therefore documented as an unsuccessful compatibility experiment. No INT4 quality, size, or speed results are claimed.

### Controlled depth ablation

A controlled comparison was performed between a four-layer baseline and a two-layer model. Both architectures were trained from scratch under the same defined protocol.

Training protocol:

- Dataset and tokenizer: same WikiText-103 tokenized splits and BPE tokenizer.
- Batch size: 32.
- Context length: 128.
- Training steps: 3,000.
- Optimizer: AdamW.
- Learning rate: 0.0003, constant.
- Weight decay: 0.01.
- Gradient clipping: 1.0.
- Matched seeds: 42, 123, and 2026.
- Same sampled training-window sequence within each matched seed pair.
- Validation evaluation every 250 steps, using the same fixed validation batches.
- Best checkpoint selected by validation loss.
- Test evaluation used the entire test split, totaling 324,084 next-token targets for every run.

#### Aggregate test results (mean ± sample standard deviation)

| Metric | 4 layers | 2 layers |
|---|---:|---:|
| Cross-entropy | **5.10479 ± 0.00495** | 5.13428 ± 0.00188 |
| Perplexity | **164.811 ± 0.816** | 169.741 ± 0.319 |
| Accuracy | **18.778% ± 0.129 pp** | 18.419% ± 0.049 pp |
| Parameters | 1,833,728 | 1,437,184 |
| Checkpoint size | 10.918 MB | 9.398 MB |

The two-layer model had approximately 21.6% fewer parameters and smaller checkpoints, but its mean test cross-entropy was higher by approximately 0.02948. The four-layer model had lower test cross-entropy in all three matched seed comparisons.

#### Individual test cross-entropy by seed

| Seed | 4 layers | 2 layers |
|---:|---:|---:|
| 42 | 5.10962 | 5.13255 |
| 123 | 5.10503 | 5.13399 |
| 2026 | 5.09973 | 5.13628 |

The selected Chatty Mini-GPT research checkpoint is the four-layer run with seed 123 because it had the lowest validation loss among the three four-layer runs. It is stored as `models/best_model.pt` in the Chatty folder.

---

## Repository structure

The deployment repository should have this structure:

```text
Chatty/
├── app.py
├── assistant_backend.py
├── generation.py
├── model.py
├── tokenizer.py
├── requirements.txt                # Chatty web-app dependencies
├── requirements-research.txt       # Mini-GPT research dependencies
├── README.md
├── .gitignore
│
├── models/
│   └── best_model.pt
│
├── tokenizer_assets/
│   ├── bpe_tokenizer.json
│   └── tokenizer_metadata.json
│
├── .streamlit/
│   └── config.toml
│
└── assets/                         # optional; current robot is inline SVG
```

The Qwen3 GGUF file is downloaded/cached by `llama.cpp` and is not stored inside this repository. Do not commit the `.venv` directory, private conversation exports, secrets, or local caches. The main `requirements.txt` is intentionally kept small for the web app; `requirements-research.txt` contains the heavier Mini-GPT research dependencies.

---

## Local installation and execution

Tested local environment:

- Windows 64-bit.
- Python `3.13.7`.
- PyTorch `2.11.0+cpu` in the local virtual environment.
- Streamlit `1.66.1`.
- Tokenizers `0.23.3`.
- `llama.cpp` WinGet package `b11977`.

### 1. Clone the repository

```cmd
git clone https://github.com/YOUR_GITHUB_USERNAME/Chatty.git
cd Chatty
```

### 2. Create and activate a virtual environment

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Chatty web-app dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The deployed Chatty interface only needs Streamlit, Requests, and language detection. PyTorch and the BPE `tokenizers` package are separated into `requirements-research.txt` because the current Qwen-powered chat page does not import the Mini-GPT research modules. To run Mini-GPT research or its standalone generation tests, install the additional dependencies with:

```cmd
pip install -r requirements-research.txt
```

### 4. Install llama.cpp

```cmd
winget install llama.cpp
```

Open a new CMD window after installation. If `llama-server` is not recognized, use the full executable path under the WinGet Packages directory.

### 5. Start Qwen3 in a dedicated terminal

```cmd
llama-server -hf QuantFactory/Qwen3-0.6B-GGUF:Q4_K_M --host 127.0.0.1 --port 8080 --ctx-size 1024 --threads 2 --batch-size 128 --ubatch-size 64 --n-gpu-layers 0 --jinja
```

Keep this terminal open. Verify the model server from another terminal:

```cmd
curl http://127.0.0.1:8080/v1/models
```

### 6. Start Chatty in a second terminal

```cmd
cd path\to\Chatty
.venv\Scripts\activate.bat
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

The Qwen server must remain running while Chatty is used.

---

## Testing

The following checks were performed during development:

1. **Mini-GPT export:** loaded the exported architecture and the selected checkpoint; original and exported logits matched with maximum absolute difference `0.0` in the verification test.
2. **Tokenizer export:** vocabulary size 8,000; the example text encoded and decoded successfully; special token IDs were confirmed as 0, 1, 2, and 3.
3. **Mini-GPT generation:** the selected checkpoint loaded on CUDA in Colab and generated a token continuation.
4. **Windows Mini-GPT test:** the checkpoint loaded using CPU PyTorch, with 4 layers and 1,833,728 parameters; generation executed successfully.
5. **Qwen server:** `/v1/models` returned the loaded QuantFactory Qwen3 model.
6. **Qwen chat API:** `/v1/chat/completions` returned a natural conversational introduction when tested with a system prompt and non-thinking mode.
7. **Backend wrapper:** `assistant_backend.py` successfully connected to the running local Qwen server and returned assistant text and token usage.

---

## GitHub setup

Create a new GitHub repository named `Chatty`. The repository can be public if you are comfortable sharing the source code and the small Mini-GPT checkpoint. Do not include private data or secrets.

From the Chatty project directory, initialize and push the repository:

```cmd
git init
git add .
git status
git commit -m "Initial Chatty application and Mini-GPT research artifacts"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/Chatty.git
git push -u origin main
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username. Review the output of `git status` before committing to ensure that `.venv`, `.env`, secrets, private conversation exports, and local cache files are not included.

If the GitHub repository was created with an initial README or license file, either clone that repository first and copy the project files into it, or reconcile the histories before pushing.

The selected Mini-GPT checkpoint is about 10.9 MB, below GitHub's 25 MiB per-file browser upload limit. The Qwen3 GGUF file is not part of this repository.

---

## Streamlit Community Cloud deployment

Streamlit Community Cloud can deploy an app from a GitHub repository by selecting the repository, branch, and entrypoint file. The expected entrypoint is `app.py`. The cloud deployment workflow also allows selection of the Python version and configuration of secrets.

**Do not deploy the current local-only backend as a working public chatbot yet.** The current `assistant_backend.py` uses:

```python
BASE_URL = "http://127.0.0.1:8080"
```

On Streamlit Community Cloud, this URL points to the cloud app container itself. It does not point back to the developer's Windows computer. The public app therefore cannot use the Qwen server running on the developer's computer.

Before deployment, choose and implement one of these architectures:

1. **A separately hosted Qwen inference endpoint:** configure `assistant_backend.py` to call the hosted endpoint, store any API key in Streamlit Secrets, and never commit that key to GitHub.
2. **Run the Qwen inference server in the same cloud environment:** package or download a compatible Linux inference runtime and GGUF model at startup, start the server as a managed subprocess, and validate that memory and CPU limits are sufficient.

Streamlit Community Cloud's documented resource limits are approximately 2 CPU cores and 2.7 GB memory at the upper end; actual resources can vary. The Qwen GGUF file size alone does not represent its full runtime memory requirement. Test the selected cloud architecture before promising public availability.

When the backend is cloud-ready, deploy from [Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in and select **Create app**.
2. Choose the Chatty GitHub repository.
3. Select branch `main`.
4. Set the main file path to `app.py`.
5. Select a supported Python version (the current Community Cloud default is Python 3.12; validate dependencies against the selected version).
6. Add required credentials under Advanced settings / Secrets if the chosen hosted backend requires them.
7. Deploy and inspect the application logs.

Never commit `.streamlit/secrets.toml`, API keys, access tokens, or passwords. Use the platform's Secrets interface for deployment credentials.

---

## Known limitations and next steps

- **Cloud inference is not configured yet.** The local Qwen endpoint only works while `llama-server` is running on the same computer as Chatty.
- **Conversation persistence is incomplete.** Conversation data is session-based and can be lost when the session ends. JSON and Markdown export are available; persistent private storage and user accounts are not implemented.
- **Grammar correction is not implemented.** Language detection and language preferences do not constitute a grammar-correction engine.
- **The Mini-GPT is not instruction-tuned.** Its text continuations can be incoherent or unrelated to a conversational prompt.
- **The Mini-GPT is small.** It has about 1.83 million parameters and should be considered an educational/research model.
- **INT4 quantization was unsuccessful** in the tested PyTorch/TorchAO/CUDA environment; no INT4 performance claim is made.
- **KV-cache speedup was not demonstrated** for the tested small model and benchmark settings, although correctness checks passed.
- **Qwen3-0.6B is also a small model.** Its answer quality and multilingual performance should be evaluated with a broader test set before making strong claims.
- **Public multi-user privacy needs more work.** Add authentication and a per-user persistent storage solution before using Chatty for sensitive conversations.

### Planned improvements

1. Configure a cloud-accessible Qwen inference backend.
2. Add persistent, private conversation storage.
3. Implement local or hosted grammar correction for English, French, and Arabic.
4. Add broader multilingual and conversational evaluation tests.
5. Add automated tests for generation and conversation import/export.
6. Document deployment costs, latency, and resource consumption after cloud testing.

---

## Technology stack and references

### Technologies

- Python
- PyTorch
- Streamlit
- Hugging Face Tokenizers
- `langdetect`
- `requests`
- `llama.cpp`
- GGUF quantized model format

### Model and platform references

- [QuantFactory Qwen3-0.6B-GGUF model repository](https://huggingface.co/QuantFactory/Qwen3-0.6B-GGUF)
- [llama.cpp project](https://github.com/ggml-org/llama.cpp)
- [Streamlit Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit Community Cloud dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Streamlit Community Cloud secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Streamlit Community Cloud resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

---

**Chatty — From a Mini-GPT research experiment to a conversational web application.**
