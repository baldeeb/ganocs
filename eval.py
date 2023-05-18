from habitat_datagen_util.utils.dataset import HabitatDataset
from habitat_datagen_util.utils.collate_tools import collate_fn
from torch.utils.data import DataLoader
import torch 
from tqdm import tqdm
import wandb
import pathlib as pl

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path='./config', config_name='base')
def run_eval(cfg: DictConfig) -> None:

    def targets2device(targets, device):
        for i in range(len(targets)): 
            for k in ['masks', 'labels', 'boxes']: 
                targets[i][k] = targets[i][k].to(device)
        return targets

    # Data
    training_data = HabitatDataset(cfg.data.training.dir)
    training_dataloader = DataLoader(training_data, 
                            **cfg.data.training.loader,
                            collate_fn=collate_fn)
    # TODO:
    # test_data = HabitatDataset(cfg.data.testing.dir)
    # testing_dataloader = DataLoader(test_data, 
    #                         **cfg.data.training.loader,
    #                         collate_fn=collate_fn)


    # Model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.model.load:
        model.load_state_dict(torch.load(cfg.model.load))
        print(f'Loaded {cfg.model.load}')
    model.to(cfg.device).train()

    # Optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # Logger
    wandb.init(project=cfg.project_name, 
               name=cfg.run_name,
               config=cfg)

    for epoch in tqdm(range(cfg.num_epochs)):
        for itr, (images, targets) in enumerate(training_dataloader):
            images = images.to(cfg.device)
            targets = targets2device(targets, cfg.device)
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            wandb.log(losses)
            wandb.log({'loss': loss})

            # Save image and NOCS every 10 iterations
            if itr % 10 == 0: 
                with torch.no_grad():
                    model.eval()
                    r = model(images)
                    if sum([v.numel() for k, v in r[0].items()]) != 0:
                        _printable = lambda a: a.permute(1,2,0).detach().cpu().numpy()
                        wandb.log({'image': wandb.Image(_printable(images[0]))})
                        if 'nocs' in r[0]:
                            wandb.log({'nocs':wandb.Image(_printable(r[0]['nocs'][0]))})
                    model.train()
        torch.save(model.state_dict(), 
                   cfg.checkpoint_dir/f'{cfg.run_name}{epoch}.pth')
        wandb.log({'epoch': epoch})


if __name__ == '__main__':
    run_eval()