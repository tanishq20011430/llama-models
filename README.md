<p align="center">
  <img src="/Llama_Repo.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/meta-Llama"> Models on Hugging Face</a>&nbsp | <a href="https://ai.meta.com/blog/"> Blog</a>&nbsp |  <a href="https://llama.meta.com/">Website</a>&nbsp | <a href="https://llama.meta.com/get-started/">Get Started</a>&nbsp
<br>

---

# Llama Models

Llama is an accessible, open large language model (LLM) designed for developers, researchers, and businesses to build, experiment, and responsibly scale their generative AI ideas. Part of a foundational system, it serves as a bedrock for innovation in the global community. A few key aspects:
1. **Open access**: Easy accessibility to cutting-edge large language models, fostering collaboration and advancements among developers, researchers, and organizations
2. **Broad ecosystem**: Llama models have been downloaded hundreds of millions of times, there are thousands of community projects built on Llama and platform support is broad from cloud providers to startups - the world is building with Llama!
3. **Trust & safety**: Llama models are part of a comprehensive approach to trust and safety, releasing models and tools that are designed to enable community collaboration and encourage the standardization of the development and usage of trust and safety tools for generative AI

Our mission is to empower individuals and industry through this opportunity while fostering an environment of discovery and ethical AI advancements. The model weights are licensed for researchers and commercial entities, upholding the principles of openness.

## Llama Models

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-models)](https://pypi.org/project/llama-models/)
[![Discord](https://img.shields.io/discord/1257833999603335178)](https://discord.gg/TZAAYNVtrU)

|  **Model** | **Launch date** | **Model sizes** | **Context Length** | **Tokenizer** | **Acceptable use policy**  |  **License** | **Model Card** |
| :----: | :----: | :----: | :----:|:----:|:----:|:----:|:----:|
| Llama 2 | 7/18/2023 | 7B, 13B, 70B | 4K | Sentencepiece | [Use Policy](models/llama2/USE_POLICY.md) | [License](models/llama2/LICENSE) | [Model Card](models/llama2/MODEL_CARD.md) |
| Llama 3 | 4/18/2024 | 8B, 70B | 8K | TikToken-based | [Use Policy](models/llama3/USE_POLICY.md) | [License](models/llama3/LICENSE) | [Model Card](models/llama3/MODEL_CARD.md) |
| Llama 3.1 | 7/23/2024 | 8B, 70B, 405B | 128K | TikToken-based | [Use Policy](models/llama3_1/USE_POLICY.md) | [License](models/llama3_1/LICENSE) | [Model Card](models/llama3_1/MODEL_CARD.md) |
| Llama 3.2 | 9/25/2024 | 1B, 3B | 128K | TikToken-based | [Use Policy](models/llama3_2/USE_POLICY.md) | [License](models/llama3_2/LICENSE) | [Model Card](models/llama3_2/MODEL_CARD.md) |
| Llama 3.2-Vision | 9/25/2024 | 11B, 90B | 128K | TikToken-based | [Use Policy](models/llama3_2/USE_POLICY.md) | [License](models/llama3_2/LICENSE) | [Model Card](models/llama3_2/MODEL_CARD_VISION.md) |

## Download

To download the model weights and tokenizer:

1. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/).
2. Read and accept the license.
3. Once your request is approved you will receive a signed URL via email.
4. Install the [Llama CLI](https://github.com/meta-llama/llama-stack): `pip install llama-stack`. (**<-- Start Here if you have received an email already.**)
5. Run `llama model list` to show the latest available models and determine the model ID you wish to download. **NOTE**:
If you want older versions of models, run `llama model list --show-all` to show all the available Llama models.

6. Run: `llama download --source meta --model-id CHOSEN_MODEL_ID`
7. Pass the URL provided when prompted to start the download.

Remember that the links expire after 24 hours and a certain amount of downloads. You can always re-request a link if you start seeing errors such as `403: Forbidden`.

## Running the models

You need to install the following dependencies (in addition to the `requirements.txt` in the root directory of this repository) to run the models:
```
pip install torch fairscale fire blobfile
```

After installing the dependencies, you can run the example scripts (within `llama_models/scripts/` sub-directory) as follows:
```bash
#!/bin/bash

CHECKPOINT_DIR=~/.llama/checkpoints/Meta-Llama3.1-8B-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun llama_models/scripts/example_chat_completion.py $CHECKPOINT_DIR
```

The above script should be used with an Instruct (Chat) model. For a Base model, use the script `llama_models/scripts/example_text_completion.py`. Note that you can use these scripts with both Llama3 and Llama3.1 series of models.

For running larger models with tensor parallelism, you should modify as:
```bash
#!/bin/bash

NGPUS=8
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun \
  --nproc_per_node=$NGPUS \
  llama_models/scripts/example_chat_completion.py $CHECKPOINT_DIR \
  --model_parallel_size $NGPUS
```

For more flexibility in running inference (including running FP8 inference), please see the [`Llama Stack`](https://github.com/meta-llama/llama-stack) repository.


## Access to Hugging Face

We also provide downloads on [Hugging Face](https://huggingface.co/meta-llama), in both transformers and native `llama3` formats. To download the weights from Hugging Face, please follow these steps:

- Visit one of the repos, for example [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
- Read and accept the license. Once your request is approved, you'll be granted access to all Llama 3.1 models as well as previous versions. Note that requests used to take up to one hour to get processed.
- To download the original native weights to use with this repo, click on the "Files and versions" tab and download the contents of the `original` folder. You can also download them from the command line if you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3.1-8B-Instruct
```

**NOTE** The original native weights of meta-llama/Meta-Llama-3.1-405B would not be available through this HugginFace repo.


- To use with transformers, the following [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) snippet will download and cache the weights:

  ```python
  import transformers
  import torch

  model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

  pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
  )
  ```

## Installations

You can install this repository as a [package](https://pypi.org/project/llama-models/) by just doing `pip install llama-models`

## Responsible Use

Llama models are a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
To help developers address these risks, we have created the [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).

## Issues

Please report any software “bug” or other problems with the models through one of the following means:
- Reporting issues with the model: [https://github.com/meta-llama/llama-models/issues](https://github.com/meta-llama/llama-models/issues)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)


## Questions

For common questions, the FAQ can be found [here](https://llama.meta.com/faq), which will be updated over time as new questions arise.


### Automated Update - Sat Feb  1 06:21:35 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:25:58 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:37:26 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:42:39 UTC 2025 🚀


### Automated Update - Sat Feb  1 06:54:15 UTC 2025 🚀


### Automated Update - Sat Feb  1 12:13:04 UTC 2025 🚀


### Automated Update - Sun Feb  2 00:41:15 UTC 2025 🚀


### Automated Update - Sun Feb  2 12:12:49 UTC 2025 🚀


### Automated Update - Mon Feb  3 00:40:10 UTC 2025 🚀


### Automated Update - Mon Feb  3 12:14:59 UTC 2025 🚀


### Automated Update - Tue Feb  4 00:39:01 UTC 2025 🚀


### Automated Update - Tue Feb  4 12:15:45 UTC 2025 🚀


### Automated Update - Wed Feb  5 00:39:13 UTC 2025 🚀


### Automated Update - Wed Feb  5 12:15:42 UTC 2025 🚀


### Automated Update - Thu Feb  6 00:39:34 UTC 2025 🚀


### Automated Update - Thu Feb  6 12:15:48 UTC 2025 🚀


### Automated Update - Fri Feb  7 00:39:22 UTC 2025 🚀


### Automated Update - Fri Feb  7 12:15:08 UTC 2025 🚀


### Automated Update - Sat Feb  8 00:38:23 UTC 2025 🚀


### Automated Update - Sat Feb  8 12:13:27 UTC 2025 🚀


### Automated Update - Sun Feb  9 00:42:12 UTC 2025 🚀


### Automated Update - Sun Feb  9 12:13:22 UTC 2025 🚀


### Automated Update - Mon Feb 10 00:40:39 UTC 2025 🚀


### Automated Update - Mon Feb 10 12:15:20 UTC 2025 🚀


### Automated Update - Tue Feb 11 00:39:34 UTC 2025 🚀


### Automated Update - Tue Feb 11 12:15:39 UTC 2025 🚀


### Automated Update - Wed Feb 12 00:39:25 UTC 2025 🚀


### Automated Update - Wed Feb 12 12:15:25 UTC 2025 🚀


### Automated Update - Thu Feb 13 00:39:48 UTC 2025 🚀


### Automated Update - Thu Feb 13 12:15:25 UTC 2025 🚀


### Automated Update - Fri Feb 14 00:39:21 UTC 2025 🚀


### Automated Update - Fri Feb 14 12:15:15 UTC 2025 🚀


### Automated Update - Sat Feb 15 00:38:48 UTC 2025 🚀


### Automated Update - Sat Feb 15 12:13:23 UTC 2025 🚀


### Automated Update - Sun Feb 16 00:43:20 UTC 2025 🚀


### Automated Update - Sun Feb 16 12:16:00 UTC 2025 🚀


### Automated Update - Mon Feb 17 00:42:11 UTC 2025 🚀


### Automated Update - Mon Feb 17 12:15:49 UTC 2025 🚀


### Automated Update - Tue Feb 18 00:39:07 UTC 2025 🚀


### Automated Update - Tue Feb 18 12:15:40 UTC 2025 🚀


### Automated Update - Wed Feb 19 00:39:33 UTC 2025 🚀


### Automated Update - Wed Feb 19 12:15:11 UTC 2025 🚀


### Automated Update - Thu Feb 20 00:40:08 UTC 2025 🚀


### Automated Update - Thu Feb 20 12:15:46 UTC 2025 🚀


### Automated Update - Fri Feb 21 00:40:03 UTC 2025 🚀


### Automated Update - Fri Feb 21 12:15:12 UTC 2025 🚀


### Automated Update - Sat Feb 22 00:38:34 UTC 2025 🚀


### Automated Update - Sat Feb 22 12:13:00 UTC 2025 🚀


### Automated Update - Sun Feb 23 00:43:06 UTC 2025 🚀


### Automated Update - Sun Feb 23 12:13:27 UTC 2025 🚀


### Automated Update - Mon Feb 24 00:41:45 UTC 2025 🚀


### Automated Update - Mon Feb 24 12:15:56 UTC 2025 🚀


### Automated Update - Tue Feb 25 00:40:36 UTC 2025 🚀


### Automated Update - Tue Feb 25 12:15:45 UTC 2025 🚀
