import sys
sys.path.insert(0, '/home/baldeeb/Code/pytorch-NOCS')

from models.nocs import get_nocs_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from utils.load_save import load_nocs
from torch.optim import Adam
import torch


def targets2device(targets, device):
    for i in range(len(targets)): 
        for k in ['masks', 'labels', 'boxes']: 
            targets[i][k] = targets[i][k].to(device)
    return targets

def run():
    CHKPT_DIR = "/home/baldeeb/Code/pytorch-NOCS/checkpoints/nocs_classification/2023-05-25_13-46-18/one_class_frozen_backbone_4.pth"
    device= 'cuda:1'

    model_cfg = {'maskrcnn_weights': MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                 'nocs_num_bins':    32,
                 'nocs_loss_mode':   'classification',
                 'multiheaded_nocs': True,}

    # model = get_nocs_resnet50_fpn(**model_cfg)
    model = load_nocs(CHKPT_DIR, **model_cfg)

    model.to(device).train()

    if False:
        from habitat_datagen_util.utils.dataset import HabitatDataset
        from habitat_datagen_util.utils.collate_tools import collate_fn
        # DATA_DIR = "/home/baldeeb/Code/pytorch-NOCS/data/habitat/train"  # larger dataset
        DATA_DIR = '/home/baldeeb/Code/pytorch-NOCS/data/habitat/generated/200of100scenes_26selectChairs/test'
        dataset = HabitatDataset(DATA_DIR)
    elif False:
        from datasets.spot_dataset import SpotDataset, collate_fn 
        DATA_DIR = '/home/baldeeb/Code/bd_spot_wrapper/data/output/front_left'
        dataset = SpotDataset(DATA_DIR)
    else:
        from datasets.rosbag_dataset import RosbagReader, collate_fn
        bag_path = '/media/baldeeb/ssd2/Data/kinect/images_poses_camerainfo.bag'
        topic_names = {'/rtabmap/rtabmap/localization_pose': 'pose',
               '/k4a/depth_to_rgb/camera_info': 'intrinsics',
               '/k4a/depth_to_rgb/image_raw': 'depth',
               '/k4a/rgb/image_raw': 'color',}
        dataset = RosbagReader(bag_path, 
                               topics=list(topic_names.keys()),
                               topic_name_map=topic_names)
        
    dataloader = DataLoader(dataset, 
                            batch_size=6, 
                            collate_fn=collate_fn)
    optimizer = Adam(model.parameters(), lr=1e-4)

    model.roi_heads.training_mode('multiview')
    for itr, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        # TODO: remove - temp 
        import cv2 
        for i, temp_image in enumerate(images):
            cv2.imwrite(f'./data/temp/test_image{i}.png', temp_image.permute(1,2,0).cpu().numpy()*255)


        targets = targets2device(targets, device)
        losses = model(images, targets)
        loss = sum(losses.values())
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        
        print(losses['multiview_consistency_loss'])


if __name__ == '__main__':
    # habitat_run()
    # spot_run()
    run()