# ART

**ART** (Audio Reasoning Tasks) is a benchmark for assessing the ability of multimodal models to solve problems that require reasoning over audio signal.

## Tasks

- **Audio Arithmetics:** Performing simple arithmetic reasoning with regard to the sounds heard.

    Example: *Are there as many bell rings as there are cat meows?*

- **Audio Transformation Detection:** Recognizing whether one recording is a transformed version of the other.

    Example: *Is the first recording a sped up version of the second recording?*

- **Cross-Recording Language Identification:** Comparison of the languages spoken in the recordings.

    Example: *Is Berlin the capital of the country this speaker comes from?*

- **Cross-Recording Speaker Identification:** Comparison of the speakers in the recordings.

    Example: *Is the same person heard speaking on both recordings?*

- **Selective Text Inference:** Inference based on some uttered content which is selected on the basis of the properties of some of the speakers.

    Example: *Is green the answer to the question asked by a man?*

- **Sound Reasoning:** Reasoning based on the recognized sound.

    Example: *Is the animal that makes the following sound bigger than a horse?*

- **Speech Features Comparison:** Comparison of two recordings regarding speech features present in them.

    Example: *Is the second recording the same text but read with a Scottish accent?*

- **Text and Sound Reasoning:** Questions that require both sound features and text understanding to be answered.

    Example: *Is the person talking about the following sound?*

- **Text and Temporal Localization Reasoning:** Questions that require both noises from localization (surroundings) and text understanding to be answered.

    Example: *Does the speaker describe the acoustic scene that they are in?*

## Samples

All speech samples are synthetic and do not constitute recordings of real individuals. To generate
instances of each task, we used three types of audio recordings - questions, utterances and sounds.

- For synthesizing questions we used samples from the LJ Speech dataset.

    Keith Ito and Linda Johnson. 2017. The LJ speech dataset. <https://keithito.com/LJ-Speech-Dataset/>. License: `Public Domain`.

- Samples for the tasks that required additional utterances spoken by different speakers were selected from the GLOBE dataset.

    Wenbin Wang, Yang Song, and Sanjay Jha. 2024b. GLOBE: A high-quality English corpus with global accents for zero-shot speaker adaptive text-to-speech. arXiv:2406.14875. <https://huggingface.co/datasets/MushanW/GLOBE>. License: `CC0-1.0`.

- The Cross-Recording Language Identification task required utterances in languages other than English. For this purpose, we used the VoxPopuli dataset.

    Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux. 2021. VoxPopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 993–1003, Online. Association for Computational Linguistics. <https://huggingface.co/datasets/facebook/voxpopuli>. License: `CC0-1.0`. The raw data is collected from 2009-2020 European Parliament event recordings. See European Parliament's [legal notice](https://www.europarl.europa.eu/legal-notice/en/) for the raw data.

- Other sounds such as tunes, background noises and animal sounds were manually selected from samples available under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) in the Freesound collection.

    Frederic Font, Gerard Roma, and Xavier Serra. 2013. Freesound technical demo. In *Proceedings of the 21st ACM International Conference on Multimedia*, MM ’13, pages 411–412, New York, NY, USA. Association for Computing Machinery. <https://freesound.org/>.

Both source code and data in this repository are licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The use of the dataset for impersonating real individuals or creating misleading content is strictly prohibited.

## Usage

### Installation

First, clone this repository to your local machine and install the required packages:

```bash
git clone https://github.com/SamsungLabs/ART.git
cd ART/
uv sync
```

### vLLM inference

We provide a script for running inference on a selected model using vLLM. It allows you to quickly generate outputs.

For a server at `http://localhost:8000/v1`, replace `YOUR_SERVED_MODEL` below:

```bash
uv run infer_vllm.py --model-name YOUR_SERVED_MODEL
```

The script sends the audio question and an instruction to answer `Yes` or `No`.

#### Arguments

You can override any of the default parameters directly from the command line.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--model-name` | Required | Model name exposed by the server. |
| `--dataset-dir` | `dataset/` beside the scripts | Directory containing task folders. |
| `--output-path` | `./outputs.json` | Prediction JSON file; parent directories are created. |
| `--overwrite` | Off | Discard previous predictions and start a new run. |
| `--save-every` | `100` | Save after this many successful new samples per task; must be positive. |
| `--base-url` | `http://localhost:8000/v1` | Server API base URL. |
| `--api-key` | `OPENAI_API_KEY` environment variable, otherwise `EMPTY` | Server API key. |
| `--max-completion-tokens` | `512` | Maximum generated tokens per request; must be positive. |
| `--descriptive` | Off | Allow free-form answers; these are not supported by the exact-match scoring protocol. |
| `--log-level` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

#### Outputs and resuming

The prediction file maps sample IDs to model responses:

```json
{
    "aa_4160014": "Yes"
}
```

Inference also writes a sidecar, such as `outputs/model-outputs.json.run.json`,
recording the model, endpoint, generation settings, and a SHA-256 fingerprint of
local metadata and audio. Fingerprinting reads all dataset files before inference.
Keep the sidecar with its predictions - resuming requires matching run metadata.
Existing predictions without the sidecar cannot be resumed. To use a different
model, settings, or dataset, choose a new `--output-path` or pass `--overwrite`.

Generated files under `outputs/` and `results/`, the default `outputs.json`, and run
sidecars are excluded by `.gitignore`. Prefer those directories for custom runs.

### Evaluation

```bash
uv run eval.py outputs.json
```

Evaluation uses only local metadata. It does not require audio decoding or a running
server. Pass `--dataset-dir PATH` to evaluate against a custom directory.

The evaluator reports per-task and overall prediction coverage and accuracy.
Missing predictions count as incorrect. Overall accuracy is the number of correct
answers divided by the total number of metadata records, rather than an unweighted
average of task scores. Unknown prediction IDs are ignored with a warning. Missing,
empty, malformed, or invalid prediction files, and files with no matching IDs, are
rejected. A partial prediction file with matching IDs is scored normally; check the
coverage report before comparing results.

The final line has the form `overall_accuracy=0.7500`. Other reports are logged to
standard error. The evaluator needs the prediction file, not its run sidecar.

## Citation

```bibtex
@inproceedings{christop-etal-2026-benchmark,
    title = "A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models",
    author = "Christop, Iwona  and
      Czy{\.z}nikiewicz, Mateusz  and
      Sk{\'o}rzewski, Pawe{\l}  and
      Bondaruk, {\L}ukasz  and
      Kubiak, Jakub  and
      Lewandowski, Marcin  and
      Kubis, Marek",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.42/",
    doi = "10.18653/v1/2026.eacl-long.42",
    pages = "953--983",
    ISBN = "979-8-89176-380-7",
    abstract = "The present benchmarks for testing the audio modality of multimodal large language models concentrate on testing various audio tasks such as speaker diarization or gender identification in isolation. Whether a multimodal model can answer the questions that require reasoning skills to combine audio tasks of different categories cannot be verified with their use. To address this issue, we propose Audio Reasoning Tasks (ART), a new benchmark for assessing the ability of multimodal models to solve problems that require reasoning over audio signal."
}
```
