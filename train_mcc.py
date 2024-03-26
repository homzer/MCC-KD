import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MultiOutputsConsistentDataset, JsonDataset
from src.evaluator import SolverEvaluator
from src.models.modeling_utils import get_parallel_model
from src.trainer import ParallelSolverMccDistillTrainer
from src.utils import setup_model_parallel, json_dump


def run(
        task: str,
        model_type: str,
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        label_file: str,
        log_dir: str = "log",
        eval_batch_size: int = 384,
        max_seq_len: int = 512,
        max_batch_size: int = 6,
        accumulation_steps: int = 1,
        T: float = 1.0,
        lr: float = 1e-5,
        epochs: int = 10,
        alpha: float = 0.1,
        lora_rank: int = 128,
        tokenizer_file: str = None,
        config_file: str = None,
        use_float16: bool = False,
        use_bfloat16: bool = False,
        seed: int = None
):
    local_rank, world_size = setup_model_parallel(
        use_float16=use_float16, use_bfloat16=use_bfloat16, seed=seed
    )

    if tokenizer_file is None:
        tokenizer_file = os.path.join(ckpt_dir, 'tokenizer.model')
    if config_file is None:
        config_file = os.path.join(ckpt_dir, 'params.json')
    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank
    )

    dataset = MultiOutputsConsistentDataset(train_file)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelSolverMccDistillTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len,
        accumulation_steps=accumulation_steps
    )
    trainer.load(ckpt_dir)

    evaluator = SolverEvaluator(
        model=model,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        max_seq_len=max_seq_len
    )

    for epoch in range(epochs):
        for data in tqdm(dataloader):
            trainer_outputs = trainer.distill(
                instructions=data['instruction'],
                outputs_a=data['output_a'],
                outputs_b=data['output_b'],
                indices_a=data['indices_a'],
                indices_b=data['indices_b'],
                alpha=alpha,
                T=T
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} ----------------------------------')
                print("CE LOSS: ", trainer_outputs.ce_loss.item())
                print("KL LOSS: ", trainer_outputs.kl_loss.item())
                trainer.predict(trainer_outputs.logits_a, data['instruction'], data['output_a'])
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))
        outputs = evaluator.forward(task, JsonDataset(label_file))
        print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
        json_dump(outputs.datalist, os.path.join(
            log_dir, f'results-epoch-{epoch + 1}-{round(outputs.acc, 4)}.json'), indent=4
        )


if __name__ == '__main__':
    fire.Fire(run)
