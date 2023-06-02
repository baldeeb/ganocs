from habitat_datagen_util.utils.dataset import HabitatDataset
from torch.utils.data import DataLoader
import torch 
from tqdm import tqdm
import wandb
import pathlib as pl
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from utils.evaluation.wrapper import eval
from utils.load_save import save_model

@hydra.main(version_base=None, config_path='./config', config_name='base')
def run(cfg: DictConfig) -> None:

    def targets2device(targets, device):
        for i in range(len(targets)): 
            for k in ['masks', 'labels', 'boxes']: 
                targets[i][k] = targets[i][k].to(device)
        return targets

    # Data
    training_data = HabitatDataset(cfg.data.training.dir)
    with open_dict(cfg):
        collate_fn = hydra.utils.instantiate(cfg.data.training.loader.pop('collate_fn'))
    training_dataloader = DataLoader(training_data, **cfg.data.training.loader, collate_fn=collate_fn)
    test_data = HabitatDataset(cfg.data.testing.dir)
    with open_dict(cfg):
        collate_fn = hydra.utils.instantiate(cfg.data.testing.loader.pop('collate_fn'))
    testing_dataloader = DataLoader(test_data, **cfg.data.testing.loader, collate_fn=collate_fn)

    if 'multiview_consistency_data' in cfg.data:
        mvc_cfg = cfg.data.multiview_consistency_data
        mvc_dataset = hydra.utils.instantiate(mvc_cfg.dataset)
        mvc_collate_fn = hydra.utils.instantiate(mvc_cfg.collate_fn)
        multiview_consistency_data = hydra.utils.instantiate(mvc_cfg.loader, 
                                                            collate_fn=mvc_collate_fn,
                                                            dataset=mvc_dataset)
    else: multiview_consistency_data = None

    # Model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.model.load:
        model.load_state_dict(torch.load(cfg.model.load))
        print(f'Loaded {cfg.model.load}')
    model.to(cfg.device).train()

    # Optimizer
    optim_cfg = cfg.optimization
    parameters = model.parameters(keys=optim_cfg.parameters)
    optimizer = hydra.utils.instantiate(optim_cfg.optimizer, 
                                        params=parameters)

    # Logger
    wandb.init(project=cfg.project_name, 
               name=cfg.run_name,
               config=cfg)

    # Pretrained wieghts eval
    eval(model, testing_dataloader, cfg.device, 
         test_data.intrinsic(), cfg.data.testing.batches)

    for epoch in tqdm(range(cfg.num_epochs)):
        for batch_i, (images, targets) in enumerate(training_dataloader):
            images = images.to(cfg.device)
            targets = targets2device(targets, cfg.device)
            losses = model(images, targets)
            if multiview_consistency_data is not None:
                mv_loss = multiview_loss(multiview_consistency_data, 
                                         model,
                                         cfg.multiview_consistency_loss_weight,
                                         cfg.device) 
                losses.update(mv_loss)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log(losses)
            wandb.log({'loss': loss})
            if batch_i % cfg.batches_before_eval == 0:
                eval(model, testing_dataloader, cfg.device, 
                     test_data.intrinsic(), cfg.data.testing.batches) 
        save_model(model, pl.Path(cfg.checkpoint_dir)/f'{cfg.run_name}_{epoch}.pth')
        wandb.log({'epoch': epoch})


def multiview_loss(dataloader, model, weight, device):

    def targets2device(targets, device):
        for i in range(len(targets)): 
            for k in ['masks', 'labels', 'boxes']: 
                targets[i][k] = targets[i][k].to(device)
        return targets

    init_training_mode = model.roi_heads._training_mode
    model.roi_heads.training_mode('multiview')
    loss = []
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets2device(targets, device)
        losses = model(images, targets)
        loss.append(sum(losses.values()))
    model.roi_heads.training_mode(init_training_mode)
    return {'multiview': ( sum(loss) / len(loss) ) * weight }

if __name__ == '__main__':
    run()