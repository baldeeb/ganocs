import pathlib as pl
from tqdm import tqdm

import wandb

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from utils.load_save import save_model
from utils.evaluation.wrapper import eval
from utils.multiview.wrapper import MultiviewLossFunctor


# TODO: discard and pass device to collate_fn
def targets2device(targets, device):
    for i in range(len(targets)): 
        for k in ['masks', 'labels', 'boxes']: 
            targets[i][k] = targets[i][k].to(device)
    return targets

def numpify(data):
    if isinstance(data, dict):
        return {k: numpify(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpify(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return data


@hydra.main(version_base=None, config_path='./config', config_name='base')
def run(cfg: DictConfig) -> None:
    
    # Data
    training_dataloader = hydra.utils.instantiate(cfg.data.training)
    testing_dataloader = hydra.utils.instantiate(cfg.data.testing)

    # Logger
    if cfg.log: 
        wandb.init(project=cfg.project_name, name=cfg.run_name, config=cfg)
        log = wandb.log
    else: log = lambda x: None

    # Model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.model.load:
        model.load_state_dict(torch.load(cfg.model.load))
        print(f'Loaded {cfg.model.load}')  # TODO: log this in a better way
    model.to(cfg.device).train()

    # (Optional) Multiview consistency loss
    if 'multiview' in cfg.data:
        mvc_dataloader = hydra.utils.instantiate(cfg.data.multiview.loader)
        multiview_loss = MultiviewLossFunctor(mvc_dataloader, model,
                                              cfg.model.multiview.loss_weight,
                                              cfg.model.multiview.loss_mode,
                                              cfg.device)
    else: multiview_loss = lambda: {}

    # Optimizer
    optim_cfg = cfg.optimization
    parameters = model.parameters(keys=optim_cfg.parameters)
    optimizer = hydra.utils.instantiate(optim_cfg.optimizer, params=parameters)

    # Training
    for epoch in tqdm(range(cfg.num_epochs), desc='Training Epoch Loop'):
        for batch_i, (images, targets) in tqdm(enumerate(training_dataloader), 
                                               total=int(len(training_dataloader)
                                                         /training_dataloader.batch_size),
                                               leave=False, desc='Training Batch Loop'):
            images = [im.to(cfg.device) for im in images]
            targets = targets2device(targets, cfg.device)       # TODO: Discard and pass device to collate_fn
            
            losses = model(images, targets)                     # Forward pass
            losses.update(multiview_loss())                     # (Optional) Add multiview loss
            loss = sum(losses.values())                         # Sum losses
            
            optimizer.zero_grad()
            loss.backward()                                     # Backward pass
            optimizer.step()
            
            log(numpify(losses))
            log({'loss': loss})
            if batch_i % cfg.batches_before_eval == 0:          # Eval
                eval(model, testing_dataloader, cfg.device, 
                     cfg.num_eval_batches, log=log)
            
            if cfg.batches_before_save and batch_i + 1 % cfg.batches_before_save == 0:
                save_model(model, pl.Path(cfg.checkpoint_dir)/f'{cfg.run_name}_{epoch}_{batch_i}.pth',
                           retain_n=cfg.get('retain_n_checkpoints', None))
            
        save_model(model, pl.Path(cfg.checkpoint_dir)/f'{cfg.run_name}_{epoch+1}.pth',
                   retain_n=cfg.get('retain_n_checkpoints', None))
        log({'epoch': epoch+1})


if __name__ == '__main__':
    run()