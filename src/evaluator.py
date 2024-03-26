import collections
import re
from typing import Union

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PairwiseDataset, JsonDataset
from src.entities import Timer, AverageMeter
from src.generator import GeneratorForCausalLM, GeneratorForVerifier
from src.math import math_equal
from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, Verifier, ParallelVerifier
from src.tokenizers.tokenizer import Tokenizer


class SolverEvaluator:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            batch_size: int,
            max_seq_len: int
    ):
        self.generator = GeneratorForCausalLM(model, tokenizer, max_seq_len)
        self.evaluators = {
            "GSM8K": GSM8KEvaluator,
            "MATH": MATHEvaluator,
            "MMLU": MMLUEvaluator,
            "ARC": MultiChoicesEvaluator,
            "CSQA": MultiChoicesEvaluator,
            "BBH": BBHEvaluator,
            "AGIEval": MultiChoicesEvaluator
        }
        self.batch_size = batch_size

    def forward(self, task: str, dataset: JsonDataset, t: float = 0.0, p: float = 1.0):
        print(f"Evaluating {task}.........")
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        evaluator = self.evaluators[task]()

        datalist = []
        timer = Timer(len(dataloader))
        for data in tqdm(dataloader):
            timer.step()
            outs = self.generator.forward(data['instruction'], t=t, p=p)
            for i, out in enumerate(outs):
                datalist.append(dict(
                    instruction=out['instruction'],
                    output=out['output'],
                    label=data['label'][i]
                ))
            print(outs[0]['instruction'] + outs[0]['output'])
            print("---" * 50)

        for data in datalist:  # TODO MATH
            data['predict'] = evaluator.forward(data['output'], data['label'] if task != "MATH" else None)

        Output = collections.namedtuple('Output', ['acc', 'datalist', 'missing', 'correct'])
        return Output(acc=evaluator.accuracy, datalist=datalist, missing=evaluator.miss, correct=evaluator.correct)


class VerifierEvaluator:
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            batch_size: int,
            max_seq_len: int
    ):
        self.generator = GeneratorForVerifier(model, tokenizer, max_seq_len)
        self.meter = AverageMeter()
        self.batch_size = batch_size

    def forward(self, dataset: PairwiseDataset):
        print("Evaluating ...")
        dataset.randomize = False
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        self.meter.reset()
        datalist = []
        for data in tqdm(dataloader):
            chosen_outs = self.generator.forward(data['instruction'], data['chosen'])
            rejected_outs = self.generator.forward(data['instruction'], data['rejected'])
            for i in range(len(data['instruction'])):
                c_reward = chosen_outs.scores[i]
                r_reward = rejected_outs.scores[i]
                datalist.append(dict(
                    instruction=data['instruction'][i],
                    chosen=data['chosen'][i],
                    rejected=data['rejected'][i],
                    chosen_reward=c_reward,
                    rejected_reward=r_reward,
                    chosen_tokens_rewards=chosen_outs.tokens_scores[i],
                    rejected_tokens_rewards=rejected_outs.tokens_scores[i]
                ))
                self.meter.forward(1 if c_reward > r_reward else 0)
        Output = collections.namedtuple('Output', ['acc', 'datalist'])
        return Output(acc=self.meter.avg, datalist=datalist)


# ================================================================================ #


class Evaluator:
    def __init__(self):
        self.meter = AverageMeter()
        self.miss = 0
        self.correct = 0

    @property
    def accuracy(self):
        if self.meter.step == 0:
            print("Warning, nothing to compute, returning accuracy is zero.")
        return self.meter.avg

    def format_label(self, label: str) -> str:
        raise NotImplementedError

    def forward(self, output: str, label: str = None) -> str:
        raise NotImplementedError

    def reset(self):
        self.meter.reset()
        self.miss = 0
        self.correct = 0


class GSM8KEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.float = r"(-?\d+)(,?\d+)?(\.\d+)?"
        self.patterns = [
            r'(?:Therefore|therefore)(.*)\n?',
            r'(?:So|so)(.*)\n?',
        ]
        self.numeric_words = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12"
        }

    def words_to_numbers(self, text: str):
        """ replace `One`, `two` with `1`, `2` etc in a text. """
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for (
            word
        ) in self.numeric_words.keys()) + r')\b', re.IGNORECASE)
        # Replace numeric words with their corresponding numbers
        converted_text = pattern.sub(
            lambda match: self.numeric_words[match.group().lower()],
            text
        )
        return converted_text

    def format_label(self, label: str) -> str:
        return re.sub(r',|\.0+$', "", label)

    def extract_numbers(self, text: str):
        results = []
        text = self.words_to_numbers(text)
        number_matches = re.findall(self.float, text)
        for nm in number_matches:
            results.append(self.format_label("".join(nm)))
        return results

    def forward(self, output: str, label: str = None) -> str:
        final_result = ""
        # splitting
        output = output.strip().split("\n")
        for out in output:
            results = []
            for pattern in self.patterns:
                # looking forward "Therefore" or "So" sentence
                matches = re.findall(pattern, out)
                for match in matches:
                    results.extend(self.extract_numbers(match))
            if len(results) == 0:
                results.extend(self.extract_numbers(out))
            if len(results) != 0:
                final_result = results[-1]

        # evaluation
        if len(final_result) == 0:
            self.miss += 1
        elif label is not None:
            if self.format_label(label) == final_result:
                self.meter.forward(1)
                self.correct += 1
            else:
                self.meter.forward(0)

        return final_result


class MATHEvaluator(Evaluator):
    def __init__(self, escape_error: bool = True):
        super().__init__()
        self.boxed = "boxed"
        self.escape_error = escape_error

    def extract_answer(self, text: str) -> str:
        a = ""
        if self.boxed in text:
            ans = text.split('boxed')[-1]
            if len(ans) == 0:
                return ""
            elif ans[0] == '{':
                stack = 1
                for c in ans[1:]:
                    if c == '{':
                        stack += 1
                        a += c
                    elif c == '}':
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()

        a = a.replace(" ", "")
        a = re.sub(r"\\mathbf", "", a)
        a = re.sub(r"^\\text", "", a)
        a = re.sub(r"^\w=", "", a)
        a = re.sub(r"\\left|\\right|\\!|\\%|\\\$|", "", a)
        a = re.sub(r"\\text{.*\n*.*}", "", a)
        a = re.sub(r"\^{?\\circ}?", "", a)
        a = re.sub(r"\\mbox{.*}", "", a)
        a = re.sub(r"\\\\", r"\\", a)
        a = re.sub(r"\\$", "", a)
        return a

    def format_label(self, label: str) -> str:
        return re.sub(r'\s', "", label)

    def forward(self, output: str, label: str = None) -> str:
        result = self.extract_answer(output)
        if label is not None:
            label = self.format_label(label)
            if len(result) == 0:
                self.miss += 1
            else:
                is_equal = False
                if self.escape_error:
                    try:
                        is_equal = math_equal(result, label)
                    except:
                        pass
                else:
                    is_equal = math_equal(result, label)

                if is_equal:
                    self.meter.forward(1)
                    self.correct += 1
                else:
                    self.meter.forward(0)

        return result


class MMLUEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def format_label(self, label: str) -> str:
        match = re.search(r'([ABCD])\.', label)
        return match.group(1) if match else None

    def forward(self, output: str, label: str = None) -> str:
        answer = None
        answers = re.findall(r'([ABCD])\.', output)
        if label is not None:
            if len(answers) == 0:
                self.miss += 1
            else:
                answer = answers[-1]
                if answer == self.format_label(label):
                    self.meter.forward(1)
                    self.correct += 1
                else:
                    self.meter.forward(0)
        return answer


class MultiChoicesEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.label_patterns = [
            r'\b([A-N])\b'
        ]

    def get_label_pattern(self, label: str) -> str:
        for pattern in self.label_patterns:
            match = re.search(pattern, label)
            if match:
                return pattern
        print(f"Warning: Unrecognized label format: '{label}'")
        return self.label_patterns[-1]

    def forward(self, output: str, label: str = None) -> str:
        answer = None
        final_results = []
        pattern = self.get_label_pattern(label)
        matches = re.findall(pattern, output)
        for match in matches:
            assert type(match) is str
            answer = match
            final_results.append(match.lower())

        # evaluation
        if len(final_results) != 0 and label is not None:
            self.meter.forward(1) if (
                    self.format_label(label) in final_results[-1:]
            ) else self.meter.forward(0)

        if len(final_results) == 0:
            self.miss += 1

        return answer

    def format_label(self, label: str):
        matches = re.findall(self.get_label_pattern(label), label)
        return matches[0].lower() if len(matches) != 0 else None


class BBHEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.label_patterns = [
            r'\b((?:True)|(?:False)|(?:true)|(?:false))\b',
            r'\b((?:Invalid)|(?:Valid)|(?:invalid)|(?:valid))\b',
            r'\b((?:Yes)|(?:No)|(?:yes)|(?:no))\b',
            r'\b([A-Z])\b',
        ]

    def get_label_pattern(self, label: str) -> str:
        for pattern in self.label_patterns:
            match = re.search(pattern, label)
            if match:
                return pattern
        # raise ValueError('Unrecognized label format: ', label)
        print(f"Warning: Unrecognized label format: '{label}'")
        return self.label_patterns[-1]

    def forward(self, output: str, label: str = None) -> str:
        answer = None
        final_results = []
        pattern = self.get_label_pattern(label)
        matches = re.findall(pattern, output)
        for match in matches:
            assert type(match) is str
            answer = match
            final_results.append(match.lower())

        # evaluation
        if len(final_results) != 0 and label is not None:
            self.meter.forward(1) if (
                    self.format_label(label) in final_results[-1:]
            ) else self.meter.forward(0)

        if len(final_results) == 0:
            self.miss += 1

        return answer

    def format_label(self, label: str):
        matches = re.findall(self.get_label_pattern(label), label)
        return matches[0].lower() if len(matches) != 0 else None
