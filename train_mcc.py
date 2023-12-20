import os
import random

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset
from src.model_lora import LoraModelArgs
from src.model_lora_fast import FastLoraLLaMA
from src.tokenizer import Tokenizer
from src.trainer import DistributedTrainer
from src.utils import setup_model_parallel


def training_with_mcc(trainer: DistributedTrainer, data: dict, alpha: float, n=2):
    indices = random.sample([i for i in range(n)], 2)
    outputs = trainer.train_mcc(
        instructions=data['instruction'],
        outputs1=data['output'][indices[0]],
        outputs2=data['output'][indices[1]],
        indices1=data['indices'][indices[0]],
        indices2=data['indices'][indices[1]],
        alpha=alpha
    )
    if trainer.step % 50 == 0:
        print(f'step {trainer.step} ----------------------------------')
        print(f"Info: ", outputs.info)
        print("CE LOSS: ", outputs.ce_loss.item())
        print("KL LOSS: ", alpha * outputs.kl_loss.item())
        predict = trainer.predict(
            outputs.logits1, data['instruction'], data['output'][indices[0]]
        )[0]
        print(predict['instruction'] + predict['output'])


def main(
        task: str,
        model_type: str,
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        log_dir: str = "log",
        diversity: int = 2,
        eval_batch_size: int = 128,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        accumulation_steps: int = 2,
        lr: float = 1e-5,
        epochs: int = 1,
        alpha: float = 0.1,
        lora_rank: int = 16,
        tokenizer_path: str = 'config/tokenizer.model',
        seed: int = None
):
    local_rank, world_size = setup_model_parallel(
        use_float16=True, seed=seed)
    params = LoraModelArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank).from_json(f"config/{model_type}/params.json")
    model = FastLoraLLaMA(params)
    dataset = JsonDataset(filename=train_file)
    data_loader = DataLoader(dataset, batch_size=max_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = DistributedTrainer(
        model=model,
        tokenizer=Tokenizer(tokenizer_path),
        optimizer=optimizer,
        accumulation_steps=accumulation_steps,
        log_dir=os.path.join(log_dir, f"{task}-{model_type}-seed-{1 if seed is None else seed}"),
        eval_batch_size=eval_batch_size
    )
    trainer.load(ckpt_dir)
    best = 0
    for epoch in range(epochs):
        if epoch == -1:
            best = trainer.evaluate(
                task=task,
                label_file=f'data/{task}/dev.json',
                output_file=f'{task}-alpha-{alpha}-init',
            )
        for data in tqdm(data_loader):
            training_with_mcc(trainer, data, alpha, n=diversity)
        acc = trainer.evaluate(
            task=task,
            label_file=f'data/{task}/dev.json',
            output_file=f'{task}-alpha-{alpha}-epoch-{epoch}',
        )
        if acc > best:
            trainer.save(save_dir)
            best = acc

    trainer.load(save_dir)
    trainer.evaluate(
        task=task,
        label_file=f"data/{task}/test.json",
        output_file=f"{task}-alpha-{alpha}-final"
    )


if __name__ == "__main__":
    fire.Fire(main)
