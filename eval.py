from habitat_datagen_util.utils.dataset import HabitatDataset
from habitat_datagen_util.utils.collate_tools import collate_fn
from torch.utils.data import DataLoader
import torch 
from tqdm import tqdm
import wandb
import pathlib as pl
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from utils.visualization import draw_3d_boxes

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
    test_data = HabitatDataset(cfg.data.testing.dir)
    testing_dataloader = DataLoader(test_data, 
                            **cfg.data.testing.loader,
                            collate_fn=collate_fn)


    # Model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.model.load:
        model.load_state_dict(torch.load(cfg.model.load))
        print(f'Loaded {cfg.model.load}')
    model.to(cfg.device).train()

    # Optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, 
                                        params=model.parameters())

    # Logger
    wandb.init(project=cfg.project_name, 
               name=cfg.run_name,
               config=cfg)

    # Pretrained wieghts eval
    eval(model, testing_dataloader, cfg.device, test_data.intrinsic(), cfg.data.testing.batches) 

    for epoch in tqdm(range(cfg.num_epochs)):
        for batch_i, (images, targets) in enumerate(training_dataloader):
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
            if batch_i % cfg.batches_before_eval == 0:
                eval(model, testing_dataloader, cfg.device, test_data.intrinsic(), cfg.data.testing.batches) 

        torch.save(model.state_dict(), 
                   cfg.checkpoint_dir/f'{cfg.run_name}{epoch}.pth')
        wandb.log({'epoch': epoch})

def l2_image_loss(pred_nocs, gt_nocs, gt_masks, device='cpu'):
    '''
    Args:
        pred_nocs: (N, 3, H, W)
        gt_nocs: (3, H, W)
        gt_mask: (N, H, W)'''
    pred_nocs =  pred_nocs.to(device)
    gt_nocs   =  gt_nocs.to(device)
    gt_masks  =  gt_masks.to(device)
    pred = pred_nocs * gt_masks[:, None]
    gt = gt_nocs[None] * gt_masks[:, None]
    return torch.mean(torch.norm(pred - gt, dim=1)).item()

def eval(model, dataloader, device, intrinsic, n_batches):
    with torch.no_grad():
        model.eval()
        loss = []
        for batch_i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            results = model(images)
            for result, target in zip(results, targets):
                loss.append(l2_image_loss(result['nocs'], 
                                          target['nocs'], 
                                          target['masks'],
                                          device=device))
            if n_batches is not None and batch_i >= n_batches:
                wandb.log({'eval_nocs_loss': sum(loss)/len(loss)})
                for result, image, target in zip(results, images, targets):
                    if np.prod(result['nocs'].shape) == 0: continue
                    img = draw_box(image*255.0, 
                                   result['masks'][0] > 0.5, 
                                   result['nocs'][0],
                                   np.array(target['depth']), 
                                   intrinsic)
                    _printable = lambda a: a.permute(1,2,0).detach().cpu().numpy()
                    wandb.log({'image': wandb.Image(img),
                            'nocs':  wandb.Image(_printable(result['nocs'][0])),})
                break
        model.train()
    

from utils.align import align
def draw_box(image, mask, nocs, depth, intrinsic):
    _to_ndarray = lambda a : a.clone().detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    transforms, scales, _ = align(_to_ndarray(mask), _to_ndarray(nocs), 
                                  _to_ndarray(depth), _to_ndarray(intrinsic))
    img = draw_3d_boxes(_to_ndarray(image.permute(1,2,0)), _to_ndarray(transforms[0]), 
                        _to_ndarray(scales[0]), _to_ndarray(intrinsic))
    return img

if __name__ == '__main__':
    run_eval()