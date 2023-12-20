import re

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset
from src.generator import Generator
from src.model import DistributedModule
from src.tokenizer import Tokenizer
from src.utils import json_dump


class DistributedEvaluator:
    ANSWER_TYPE_NUMERIC = "NUMERIC"
    ANSWER_TYPE_CHOICE = "CHOICE"
    NUMERIC_TASKS = ['GSM8K', 'SVAMP', 'ASDiv']
    CHOICE_TASKS = ['CSQA']

    def __init__(self, model: DistributedModule, tokenizer: Tokenizer):
        self.model = model
        self.generator = Generator(self.model, tokenizer)

    def _get_answer_type(self, task: str):
        if task in self.NUMERIC_TASKS:
            return self.ANSWER_TYPE_NUMERIC
        elif task in self.CHOICE_TASKS:
            return self.ANSWER_TYPE_CHOICE
        else:
            raise ValueError(
                f"Unrecognized task `{task}`; "
                f"Currently supported {self.NUMERIC_TASKS+self.CHOICE_TASKS}")

    def evaluate(self,
                 task: str,
                 label_file,
                 output_file,
                 batch_size,
                 max_seq_len,
                 temperature: float,
                 top_p: float):
        print("Evaluating.........")
        answer_type = self._get_answer_type(task)

        def extract_predict(_output, _answer_type) -> list:
            _output = _output.strip()
            # only count for the last line
            endline = _output.split('\n')[-1]
            if _answer_type == self.ANSWER_TYPE_NUMERIC:
                matches = re.findall(r'(-?\d+)(,?\d+)?(\.\d+)?', endline)
            elif _answer_type == self.ANSWER_TYPE_CHOICE:
                matches = re.findall(r'[A-G]', endline)
            else:
                raise ValueError(
                    f"Unrecognized answer type: `{_answer_type}`; "
                    f"Currently supported [`{self.ANSWER_TYPE_NUMERIC}`|`{self.ANSWER_TYPE_CHOICE}`]")
            predict = []
            for match in matches:
                predict.append(''.join(match))
            return predict

        def evaluate_accuracy(_datalist, _answer_type):
            _hit = 0
            for _data in _datalist:
                if _answer_type in [self.ANSWER_TYPE_NUMERIC, self.ANSWER_TYPE_CHOICE]:
                    if _data['label'] in _data['predict'][-1:]:
                        _hit += 1
                else:
                    raise ValueError(f"Unknown answer type: {_answer_type}")
            _accuracy = _hit / len(_datalist)
            return _accuracy

        dataset = JsonDataset(label_file)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        datalist = []
        for data in tqdm(data_loader):
            results = self.generator.generate(
                prompts=data['instruction'],
                max_gen_len=max_seq_len,
                temperature=temperature,
                top_p=top_p)
            for i, result in enumerate(results):
                datalist.append(dict(
                    instruction=result['instruction'],
                    output=result['output'],
                    predict=extract_predict(result['output'], answer_type),
                    label=data['label'][i]))
        accuracy = evaluate_accuracy(datalist, answer_type)
        print(f"Accuracy: {accuracy}")
        if output_file is not None:
            json_dump(datalist, f"{output_file}-{round(float(accuracy), 4)}.log")
        return accuracy

    def load(self, ckpt_dir):
        self.model.load(ckpt_dir)
