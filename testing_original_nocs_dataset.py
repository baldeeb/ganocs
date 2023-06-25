import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./config/data", config_name="original_nocs")
def run(cfg: DictConfig):
    
    dataloader = hydra.utils.instantiate(cfg.training)
    # print(OmegaConf.to_yaml(set_cfg))
    for data in dataloader:
        print(len(data))
    print('Done')


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
    run()