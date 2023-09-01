import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

@hydra.main(version_base=None, config_path="./config", config_name="original_data")
def testing_termination(cfg: DictConfig):
    

    datasets = {
        'training': hydra.utils.instantiate(cfg.data.training),
        'testing': hydra.utils.instantiate(cfg.data.testing),
        'habitat': hydra.utils.instantiate(cfg.data.base),
    }
    def __test(x, n):
        print(f'{n} data length {len(x)}')
        for _ in tqdm(x, total=len(x)):
            pass
        print('Done')

    for k, v in datasets.items():
        __test(v, k)

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
                print(f'{lk}, {lv.shape}  -  {type(lv)}')
                try: print(f'\t min: {min(lv)}, max: {max(lv)}')
                except: 
                    try: print(f'\t min: {lv.min()}, max: {lv.max()}')
                    except: pass
                try: print(f'\tdata type: {lv.dtype}')
                except: pass
                
            break

        print('boxes')
        print(data[1][0]['boxes'])
        print('*'*10)



@hydra.main(version_base=None, config_path="./config", config_name="original_data")
def debug_iterators(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.data.training),

    for i, data in enumerate(dataset[0]):
        print(f'iter: {i} - len: {len(data[0])}')


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




import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='./config', config_name='original_data', version_base=None)
def verify_data(cfg: DictConfig):
    L = hydra.utils.instantiate(cfg.data.training)
    # L = hydra.utils.instantiate(cfg.data.testing)
    # L = hydra.utils.instantiate(cfg.data.base)
    
    
    # plot image with annotated boxes and object classes
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def plot_image(img, boxes, labels, scores=None):
        fontsize, font_color = 12, 'red'
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if scores is not None:
                ax.text(x1, y1, f'{labels[i]}: {scores[i]:.2f}', color=font_color, fontsize=fontsize)
            else:
                ax.text(x1, y1, f'{labels[i]}', color=font_color, fontsize=fontsize)
        plt.show()


    for i, data in enumerate(L):
        img = data[0][0].permute(1,2,0)
        boxes = data[1][0]['boxes']
        labels = data[1][0]['labels']
        plot_image(img, boxes, labels)
        if i > 5: break



@hydra.main(version_base=None, config_path="./config", config_name="original_data")
def simple_test(cfg: DictConfig):
    datasets = {
        'training': hydra.utils.instantiate(cfg.data.training),
        # 'testing': hydra.utils.instantiate(cfg.data.testing),
        # 'habitat': hydra.utils.instantiate(cfg.data.base),
    }
    print("here")
    def __test(x, n):
        print(f'{n} data length {len(x)}')
        for _ in tqdm(x, total=len(x)):
            pass
        print('Done')

    for k, v in datasets.items():
        __test(v, k)
        break


from datasets.original_nocs.visualization_utils import visualize_original_nocs_data_point
import matplotlib.pyplot as plt
import torch

@hydra.main(version_base=None, config_path="./config", config_name="original_data")
def visualize_data(cfg: DictConfig):
    datasets = {
        'training': hydra.utils.instantiate(cfg.data.training),
        # 'testing': hydra.utils.instantiate(cfg.data.testing),
        # 'habitat': hydra.utils.instantiate(cfg.data.base),
    }

    for k, dataset in datasets.items():
        print(f'{k} data length {len(dataset)}')
        B = dataset._batch_size
        for data in tqdm(dataset, total=len(dataset)):
            images, information = data[0], data[1]
            images = [i.cpu().numpy() if isinstance(i, torch.Tensor) else i for i in images]
            images = [i.transpose(1,2,0) for i in images]            
            for info in information:
                for k in info.keys():
                    if isinstance(info[k], torch.Tensor): 
                        info[k] = info[k].cpu().numpy()

            fig, axs_grid = plt.subplots(B, 3)
            for img, info, axs in zip(images, information, axs_grid):
                depth, masks, nocs, labels, boxes, scales, camera_pose, intrinsics, _ = tuple(info.values())
                visualize_original_nocs_data_point(axs, img, depth, masks, boxes, labels, nocs)
            # fig, axs = plt.subplots(1, 3)
            # draw_2d_boxes_with_labels(axs[0], image, masks, boxes, labels)
            # axs[1].imshow(depth)
            # axs[2].imshow(nocs.transpose(1,2,0))
            plt.show()
            if input('press enter for another "n" for next dataset: ') == 'n': break


if __name__ == "__main__":
    # testing_termination()
    # compare_datasets()
    # debug_iterators()
    # verify_data()
    # simple_test()

    visualize_data()