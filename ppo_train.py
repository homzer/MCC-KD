import gc
import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset
from src.entities import Timer
from src.evaluator import SolverEvaluator
from src.models.llama import LoraLlamaVerifier, LoraLlama
from src.models.modeling_args import LoraLlamaArgs
from src.ppo.buffer import CriticRolloutBuffer, RolloutBuffer, ActorRolloutBuffer
from src.ppo.collector import CriticBufferCollector, ActorBufferCollector
from src.ppo.trainer import ParallelActorTrainerForCausalLM, ParallelCriticTrainerForCausalLM
from src.tokenizers import LlamaTokenizer
from src.utils import setup_model_parallel, set_barrier, json_dump


def run(
        actor_ckpt_dir: str,
        actor_config_file: str,
        actor_save_dir: str,
        critic_ckpt_dir: str,
        critic_config_file: str,
        critic_save_dir: str,
        reward_model_ckpt_dir: str,
        reward_model_config_file: str,
        task: str,
        train_file: str,
        label_file: str,
        log_dir: str,
        lora_rank: int = 16,
        max_batch_size: int = 4,
        max_buffer_size: int = 96,
        max_seq_len: int = 512,
        epochs: int = 1,
        inner_epochs: int = 2,
        lr: float = 1e-5,
        tokenizer_path: str = None,
):
    assert actor_ckpt_dir != critic_save_dir
    tokenizer_path = tokenizer_path if tokenizer_path else 'config/tokenizer.model'
    dataset = JsonDataset(f=train_file)
    dataloader = DataLoader(dataset, batch_size=max_buffer_size)

    local_rank, world_size = setup_model_parallel()
    tokenizer = LlamaTokenizer(tokenizer_path)
    actor_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(actor_config_file)
    critic_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(critic_config_file)
    reward_model_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(reward_model_config_file)

    for epoch in range(epochs):
        actor = LoraLlama(actor_args)
        actor.init_weights()
        actor.load(actor_ckpt_dir if epoch == 0 else os.path.join(actor_save_dir, f"epoch-{epoch}"))

        # Evaluation
        actor_evaluator = SolverEvaluator(actor, tokenizer, max_buffer_size, max_seq_len)
        eval_outputs = actor_evaluator.forward(task, JsonDataset(label_file))
        print("Evaluate Accuracy: ", eval_outputs.acc, "Missing: ", eval_outputs.missing)
        os.makedirs(log_dir, exist_ok=True)
        json_dump(eval_outputs.datalist, os.path.join(
            log_dir, f'results-epoch-{epoch}-{round(eval_outputs.acc, 4)}.json'
        ), indent=4)

        actor_buffer_collector = ActorBufferCollector(actor, tokenizer, max_seq_len)
        actor_rollout_buffer = ActorRolloutBuffer()
        print('Actor buffer collecting ...')
        timer = Timer(len(dataloader))
        for data in tqdm(dataloader):
            timer.step()
            actor_rollout_buffer.extend(
                actor_buffer_collector.forward(data['instruction'])
            )
            print(data['instruction'][-1])
            print(tokenizer.decode(actor_rollout_buffer.actions[-1][actor_rollout_buffer.action_masks[-1]].tolist()))

        actor.cpu()
        del actor
        del actor_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        critic = LoraLlamaVerifier(critic_args)
        critic.init_weights()
        critic.load(critic_ckpt_dir if epoch == 0 else os.path.join(critic_save_dir, f"epoch-{epoch}"))
        critic_buffer_collector = CriticBufferCollector(critic, tokenizer, max_seq_len)
        critic_rollout_buffer = CriticRolloutBuffer()
        print('Critic buffer collecting ...')
        for data in actor_rollout_buffer.get(max_buffer_size):
            critic_rollout_buffer.extend(
                critic_buffer_collector.forward(
                    data.instructions, data.actions, data.action_masks
                )
            )

        critic.cpu()
        del critic
        del critic_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        reward_model = LoraLlamaVerifier(reward_model_args)
        reward_model.init_weights()
        reward_model.load(reward_model_ckpt_dir)
        reward_buffer_collector = CriticBufferCollector(reward_model, tokenizer, max_seq_len)
        reward_rollout_buffer = CriticRolloutBuffer()
        print('Reward buffer collecting ...')
        for data in actor_rollout_buffer.get(max_buffer_size):
            reward_rollout_buffer.extend(
                reward_buffer_collector.forward(
                    data.instructions, data.actions, data.action_masks
                )
            )

        reward_model.cpu()
        del reward_model
        del reward_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        rollout_buffer = RolloutBuffer(
            obs=actor_rollout_buffer.obs,
            actions=actor_rollout_buffer.actions,
            rewards=reward_rollout_buffer.scores,
            values=critic_rollout_buffer.scores,
            action_logits=actor_rollout_buffer.action_logits,
            action_masks=actor_rollout_buffer.action_masks
        )

        torch.save({
            'obs': rollout_buffer.obs[: max_buffer_size],
            'actions': rollout_buffer.actions[: max_buffer_size],
            'values': rollout_buffer.values[: max_buffer_size],
            'rewards': rollout_buffer.rewards[: max_buffer_size],
            'action_masks': rollout_buffer.action_masks[: max_buffer_size],
            'advantages': rollout_buffer.advantages[: max_buffer_size],
            'returns': rollout_buffer.returns[: max_buffer_size]
        }, f'buffer_{epoch}.bin')

        actor = LoraLlama(actor_args)
        actor.init_weights()
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.1 * lr if epoch == 0 else lr)
        actor_trainer = ParallelActorTrainerForCausalLM(actor, actor_optimizer)
        actor_trainer.load_model(actor_ckpt_dir) if (
                epoch == 0
        ) else actor_trainer.load(os.path.join(actor_save_dir, f"epoch-{epoch}"))
        print('Actor training ...')
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size):
                outputs = actor_trainer.forward(data)
                if actor_trainer.step % 100 == 0:
                    print(f'--------- STEP {actor_trainer.step} OF {len(rollout_buffer) // max_batch_size} ---------')
                    print('Loss: ', outputs.loss)
        actor_trainer.save(os.path.join(actor_save_dir, f"epoch-{epoch + 1}"))

        actor.cpu()
        del actor
        del actor_optimizer
        del actor_trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        critic = LoraLlamaVerifier(critic_args)
        critic.init_weights()
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        critic_trainer = ParallelCriticTrainerForCausalLM(critic, critic_optimizer)
        critic_trainer.load_model(critic_ckpt_dir) if (
                epoch == 0
        ) else critic_trainer.load(os.path.join(critic_save_dir, f"epoch-{epoch}"))
        print('Critic training ...')
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size):
                outputs = critic_trainer.forward(data)
                if critic_trainer.step % 100 == 0:
                    print(f'--------- STEP {critic_trainer.step} OF {len(rollout_buffer) // max_batch_size} ---------')
                    print('Loss: ', outputs.loss)
        critic_trainer.save(os.path.join(critic_save_dir, f"epoch-{epoch + 1}"))

        critic.cpu()
        del critic
        del critic_optimizer
        del critic_trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
