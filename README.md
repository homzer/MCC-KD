# MCC-KD: Multi-CoT Consistent Knowledge Distillation (Llama Code Section)

1. We provide training sets, validation sets, test sets, and extracted teacher rationales for datasets including GSM8K, CSQA, SVAMP, and ASDiv in the data directory.
(Note that only the GSM8K dataset contains relatively complete rationales. Currently, the provided extracted rationales are only a part of our dataset, and we are still in the process of organizing more data.)

2. The training scripts are in the project's root directory, named `train.py` and `train_mcc.py`. The corresponding shell script files for running are located in the scripts/ directory as `train.sh` and `train_mcc.sh`.
(Currently, only a training version with a parallel quantity of 1 is provided.)

3. Model configuration files need to be placed in the config directory, and the weight file, such as the llama-7B weight file `consolidated.00.pth`, should be placed in the config/7B directory.
(Note that the model parameter file to be downloaded is the original llama version, not the Hugging Face version, and the model parameter file name should contain the suffix `.pth`.)

4. MCC-KD requires ensuring the diversity of rationales and finding a common answer span. The code for this data processing part is still being organized. The process is relatively straightforward, and you can also write it yourself.
Make sure to include "indices" to record the starting and ending indices of the common answer span. It should look something like the following:

```json
[
  {
    "instruction": "...",
    "output": [
        "rationale1",
        "rationale2"
    ],
    "label": "...",
    "indices": [
        [
            start_answer_idx_of_rationale1,
            end_answer_idx_of_rationale1
        ],
        [
            start_answer_idx_of_rationale2,
            end_answer_idx_of_rationale2
        ]
    ]
  }  
]
```


