import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from utils.evaluation.wrapper import eval
import logging

# TODO: discard and pass device to collate_fn
def targets2device(targets, device):
    for i in range(len(targets)): 
        for k in ['masks', 'labels', 'boxes']: 
            targets[i][k] = targets[i][k].to(device)
    return targets



@hydra.main(version_base=None, config_path='./config', config_name='eval')
def run(cfg: DictConfig) -> None:
    
    # Data
    testing_dataloader = hydra.utils.instantiate(cfg.data.testing)

    # Logger
    if cfg.log: 
        wandb.init(**cfg.logger, config=OmegaConf.to_container(cfg))
        log = wandb.log
    else: log = lambda x: None

    # Model
    model = hydra.utils.instantiate(cfg.model)
    assert cfg.model.load.path, 'No model to load...'
    map_loc = 'cpu' if cfg.device=="cpu" else None 
    state_dict = torch.load(cfg.model.load.path, 
                            map_location=map_loc)
    model.load_state_dict(state_dict)
    logging.info(f'Loaded {cfg.model.load}')
    model.to(cfg.device).eval()

    eval(model, testing_dataloader, cfg.device,cfg.mAP_configs,
         num_batches=cfg.num_eval_batches,log=log) 


if __name__ == '__main__':
    run()