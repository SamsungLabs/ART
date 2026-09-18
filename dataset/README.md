# Dataset

ART contains nine tasks with 1,000 samples per task (9,000 total). Each task has
500 `yes` answers and 500 `no` answers. Audio files are mono FLAC at 16 kHz.

Each task directory contains its audio and a `metadata.json` file:

```text
dataset/
├── audio_arithmetics/
│   ├── metadata.json
│   ├── aa_4160014.flac
│   └── ...
├── audio_transformation_detection/
│   ├── metadata.json
│   └── ...
└── ...
```

Each metadata file is a JSON array of records with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `instance_id` | string | Unique sample ID; maps to `<instance_id>.flac` in the same directory. |
| `template_id` | string | Identifier of the question template. |
| `question_id` | string | Identifier of the instantiated question. |
| `question` | string | Text of the question asked in the audio. |
| `slots` | object | Task-specific template values, such as sounds, counts, speakers, or utterances. |
| `answer` | string | Ground-truth answer (`yes` or `no`). |

Instance IDs are unique across all tasks. Keep each record with its matching audio
file when distributing or selecting samples.
