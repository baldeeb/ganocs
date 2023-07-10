import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./config/data", config_name="original_nocs")
def run(cfg: DictConfig):
    
    dataloader = hydra.utils.instantiate(cfg.training)
    # base = hydra.utils.instantiate(cfg.base)
    testing = hydra.utils.instantiate(cfg.testing)
    
    print(f'dataset has {len(dataloader)} elements')
    print(f'testing dataset has {len(testing)} elements')

    for i, data in enumerate(dataloader):
        print(f'iter: {i} - len: {len(data[1])}')
    print('Done')



@hydra.main(version_base=None, config_path="./config", config_name="original_data")
def compare_datasets(cfg: DictConfig):
    
    # hydra.utils.instantiate(cfg.data.testing),


    loaders = [
        hydra.utils.instantiate(cfg.data.training),
        hydra.utils.instantiate(cfg.data.testing),
        hydra.utils.instantiate(cfg.data.base),
    ]
  
    for loader_idx, loader in enumerate(loaders):
        print(f'loader: {loader_idx} contains {len(loader)} elements')
        for data_idx, data in enumerate(loader):
            print(f'num images {len(data[0])}, num lebels for image {len(data[1][0])}')
            for l_idx, (lk, lv) in enumerate(data[1][0].items()):
                if lv is None: 
                    print(f'{lk} = None')
                    continue
                print(f'{lk}, length: {lv.shape}')
                print(f'\ttype: {type(lv)}')
                try: print(f'\t min: {min(lv)}, max: {max(lv)}')
                except: 
                    try: print(f'\t min: {lv.min()}, max: {lv.max()}')
                    except: pass
                try: print(f'\tdata type: {lv.dtype}')
                except: pass
                
            break
        print('*'*10)


def init_datasets(cfg):

    def init_set(set_cfg):
        dataset = hydra.utils.instantiate(set_cfg.loader)
        if set_cfg.type == 'real':
            dataset.load_real_scenes(set_cfg.dataset_dir)
        if set_cfg.type == 'synthetic':
            dataset.load_camera_scenes(set_cfg.dataset_dir)
        dataset.prepare(set_cfg.class_map)
        return dataset

    # init_set(cfg['training']['datasets']['real'])
    init_set(cfg.training.datasets.real)
    init_set(cfg.training.datasets.camera)

if __name__ == "__main__":
    # run()
    compare_datasets()