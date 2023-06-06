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

    # Model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.model.load:
        model.load_state_dict(torch.load(cfg.model.load))
        print(f'Loaded {cfg.model.load}')
    model.to(cfg.device).train()

    # (Optional) Multiview consistency loss
    if 'multiview_consistency_data' in cfg.data:
        mvc_cfg = cfg.data.multiview_consistency_data
        mvc_dataset = hydra.utils.instantiate(mvc_cfg.dataset)
        mvc_collate_fn = hydra.utils.instantiate(mvc_cfg.collate_fn)
        mvc_dataloader = hydra.utils.instantiate(mvc_cfg.loader, 
                                                 collate_fn=mvc_collate_fn,
                                                 dataset=mvc_dataset)
        multiview_loss = MultiviewLossFunctor(mvc_dataloader,
                                              model,
                                              cfg.multiview_consistency_loss_weight,
                                              cfg.device)
    else: multiview_loss = lambda: {}

    # Optimizer
    optim_cfg = cfg.optimization
    parameters = model.parameters(keys=optim_cfg.parameters)
    optimizer = hydra.utils.instantiate(optim_cfg.optimizer, params=parameters)

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
            losses = model(images, targets)                 # Forward pass
            losses.update(multiview_loss())                 # Add multiview loss
            loss = sum(losses.values())                     # Sum losses
            optimizer.zero_grad()
            loss.backward()                                 # Backward pass
            optimizer.step()
            wandb.log(losses)
            wandb.log({'loss': loss})
            if batch_i % cfg.batches_before_eval == 0:      # Eval
                eval(model, testing_dataloader, cfg.device, 
                     test_data.intrinsic(), cfg.data.testing.batches) 
        save_model(model, pl.Path(cfg.checkpoint_dir)/f'{cfg.run_name}_{epoch}.pth')
        wandb.log({'epoch': epoch})


class MultiviewLossFunctor:

    def __init__(self, 
                 dataloader:DataLoader, 
                 model:torch.nn.Module, 
                 weight:float, 
                 device:str):
        self.dataloader = dataloader
        self.data_iter = self.dataloader.__iter__()
        self.model = model
        self.weight = weight
        self.device = device

    def _get_data(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            print('WARNING: Multiview dataloader ran out of data. Resetting.')
            self.data_iter = self.dataloader.__iter__()
            return next(self.data_iter)

    @staticmethod
    def _targets2device(targets, device):
            for i in range(len(targets)): 
                for k in ['masks', 'labels', 'boxes']: 
                    targets[i][k] = targets[i][k].to(device)
            return targets
    
    def __call__(self):
        init_training_mode = self.model.roi_heads._training_mode
        self.model.roi_heads.training_mode('multiview')
        images, targets = self._get_data()
        images = images.to(self.device)
        targets = self._targets2device(targets, self.device)
        losses = self.model(images, targets)
        self.model.roi_heads.training_mode(init_training_mode)
        l = (sum(losses.values()) / len(losses)) * self.weight
        return {'multiview': l}



if __name__ == '__main__':
    run()