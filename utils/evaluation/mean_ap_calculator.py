import numpy as np
from utils.evaluation.tools import compute_mAP,compute_ap_from_matches_scores
import matplotlib.pyplot as plt
import os
import wandb

class MeanAveragePrecisionCalculator:
    def __init__(self, config_dict):
        self.synset_names = config_dict.synset_names
        self.num_classes = len(self.synset_names)
        self.degree_thresholds = range(
            config_dict.get('degree_thresholds_start'), 
            config_dict.get('degree_thresholds_end'), 
            config_dict.get('degree_thresholds_step')
        )
        self.shift_thresholds = np.linspace(
            config_dict.get('shift_thresholds_start'), 
            config_dict.get('shift_thresholds_end'), 
            config_dict.get('shift_thresholds_steps')
        ) * config_dict.get('shift_thresholds_multiplier')
        self.iou_3d_thresholds = np.linspace(
            config_dict.get('iou_3d_thresholds_start'), 
            config_dict.get('iou_3d_thresholds_end'), 
            config_dict.get('iou_3d_thresholds_steps')
        )
        self.degree_thres_list = list(self.degree_thresholds) + [360]
        self.num_degree_thres = len(self.degree_thres_list)

        self.shift_thres_list = list(self.shift_thresholds) + [100]
        self.num_shift_thres = len(self.shift_thres_list)

        self.iou_thres_list = list(self.iou_3d_thresholds)
        self.num_iou_thres = len(self.iou_thres_list)

        self.iou_3d_aps = np.zeros((self.num_classes + 1, self.num_iou_thres))
        self.iou_pred_matches_all = [np.zeros((self.num_iou_thres, 0)) for _ in range(self.num_classes)]
        self.iou_pred_scores_all  = [np.zeros((self.num_iou_thres, 0)) for _ in range(self.num_classes)]
        self.iou_gt_matches_all   = [np.zeros((self.num_iou_thres, 0)) for _ in range(self.num_classes)]
        
        self.pose_aps = np.zeros((self.num_classes + 1, self.num_degree_thres, self.num_shift_thres))
        self.pose_pred_matches_all = [np.zeros((self.num_degree_thres, self.num_shift_thres, 0)) for _  in range(self.num_classes)]
        self.pose_gt_matches_all  = [np.zeros((self.num_degree_thres, self.num_shift_thres, 0)) for _  in range(self.num_classes)]
        self.pose_pred_scores_all = [np.zeros((self.num_degree_thres, self.num_shift_thres, 0)) for _  in range(self.num_classes)]
        self.use_matches_for_pose = config_dict.get('use_matches_for_pose')
        self.iou_pose_thres = config_dict.get('iou_pose_thres')
        self.print_result=config_dict.get('print_result')


    def computing_mAP(self, result, target, device):
        # Implement the logic to compute mAP using result and target
        compute_mAP(result,target,device,self.synset_names,self.iou_thres_list,self.iou_pred_matches_all,self.iou_pred_scores_all,
                                self.iou_gt_matches_all,self.use_matches_for_pose,self.iou_pose_thres,self.degree_thres_list, 
                                self.shift_thres_list,self.pose_pred_matches_all,self.pose_pred_scores_all,self.pose_gt_matches_all)
                    
    def get_mAP_dict(self, summary=False):
        # iou_dict = {}
        # iou_dict['thres_list'] = self.iou_thres_list
        for cls_id in range(1, self.num_classes):
            # class_name = self.synset_names[cls_id]
            for s, iou_thres in enumerate(self.iou_thres_list):
                self.iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(self.iou_pred_matches_all[cls_id][s, :],
                                                                    self.iou_pred_scores_all[cls_id][s, :],
                                                                    self.iou_gt_matches_all[cls_id][s, :])    

        self.iou_3d_aps[-1, :] = np.mean(self.iou_3d_aps[1:-1, :], axis=0)
        
        # iou_dict['aps'] = self.iou_3d_aps

        for i, degree_thres in enumerate(self.degree_thres_list):                
            for j, shift_thres in enumerate(self.shift_thres_list):
                for cls_id in range(1, self.num_classes):
                    cls_pose_pred_matches_all = self.pose_pred_matches_all[cls_id][i, j, :]
                    cls_pose_gt_matches_all = self.pose_gt_matches_all[cls_id][i, j, :]
                    cls_pose_pred_scores_all = self.pose_pred_scores_all[cls_id][i, j, :]
                    self.pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all, 
                                                                            cls_pose_pred_scores_all, 
                                                                            cls_pose_gt_matches_all)
                self.pose_aps[-1, i, j] = np.mean(self.pose_aps[1:-1, i, j])
        
        results = {}
        run_idx = None if summary else -1

        results["3D IoU at 25"]    = self.get_3d_ious(0.25, run_idx=run_idx)
        results["3D IoU at 50"]    = self.get_3d_ious(0.5,  run_idx=run_idx)

        results["5 degree, 5cm"]   = self.get_pose_aps(5, 5 ,  run_idx=run_idx)
        results["5 degree, 10cm"]  = self.get_pose_aps(5, 10,  run_idx=run_idx)
        results["10 degree, 5cm"]  = self.get_pose_aps(10, 5,  run_idx=run_idx)
        results["10 degree, 10cm"] = self.get_pose_aps(10, 10, run_idx=run_idx)
        results["15 degree, 5cm"]  = self.get_pose_aps(15, 5,  run_idx=run_idx)
        results["15 degree, 10cm"] = self.get_pose_aps(15, 10, run_idx=run_idx)

        if self.print_result:
            for k, v in results.items():
                print(f'{k}: {v:.1f}')

        return results

    def plot_map_curves(self,log:callable=wandb.log):
        # draw iou 3d AP vs. iou thresholds
        fig_iou = plt.figure()
        ax_iou = plt.subplot(111)
        plt.ylabel('AP')
        plt.ylim((0, 1))
        plt.xlabel('3D IoU thresholds')
        log_dir='/home/kdesingh/chada022/torch_nocs/checkpoints/map_curves'
        iou_output_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.png'.format(self.iou_thres_list[0], self.iou_thres_list[-1]))
        iou_dict_pkl_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.pkl'.format(self.iou_thres_list[0], self.iou_thres_list[-1]))

        iou_dict = {}
        iou_dict['thres_list'] = self.iou_thres_list

        for cls_id in range(1, self.num_classes):
            class_name = self.synset_names[cls_id]
            ax_iou.plot(self.iou_thres_list, self.iou_3d_aps[cls_id, :], label=class_name)
        ax_iou.plot(self.iou_thres_list, self.iou_3d_aps[-1, :], label='mean')
        ax_iou.legend()
        ax_iou.set_title("IoU vs AP")
        ax_iou.set_xlabel("IoU Threshold")
        ax_iou.set_ylabel("Average Precision")

        # Log the plot to wandb
        log({"IoU_vs_AP": wandb.Image(fig_iou)})
        plt.close(fig_iou)

        for cls_id in range(1, self.num_classes):
            class_name = self.synset_names[cls_id]
            print(class_name)
            # print(np.amin(aps[i, :, :]), np.amax(aps[i, :, :]))
        
            #ap_image = cv2.resize(pose_aps[cls_id, :, :]*255, (320, 320), interpolation = cv2.INTER_LINEAR)
            fig_iou = plt.figure()
            ax_iou = plt.subplot(111)
            plt.ylabel('Rotation thresholds/degree')
            plt.ylim((self.degree_thres_list[0], self.degree_thres_list[-2]))
            plt.xlabel('translation/cm')
            plt.xlim((self.shift_thres_list[0], self.shift_thres_list[-2]))
            plt.imshow(self.pose_aps[cls_id, :-1, :-1], cmap='jet', interpolation='bilinear')

            output_path = os.path.join(log_dir,'AP_{}_{}-{}degree_{}-{}cm.png'.format(class_name, 
                                                                                    self.degree_thres_list[0], self.degree_thres_list[-2], 
                                                                                    self.shift_thres_list[0], self.shift_thres_list[-2]))
            output_name= 'AP_{}_{}-{}degree_{}-{}cm'.format(class_name,self.degree_thres_list[0], self.degree_thres_list[-2], 
                                                                            self.shift_thres_list[0], self.shift_thres_list[-2])
            plt.colorbar()
            log({output_name: wandb.Image(fig_iou)})
            plt.close(fig_iou)        
        
        #ap_mean_image = cv2.resize(pose_aps[-1, :, :]*255, (320, 320), interpolation = cv2.INTER_LINEAR) 
        
        fig_pose = plt.figure()
        ax_pose = plt.subplot(111)
        plt.ylabel('Rotation thresholds/degree')
        plt.ylim((self.degree_thres_list[0], self.degree_thres_list[-2]))
        plt.xlabel('translation/cm')
        plt.xlim((self.shift_thres_list[0], self.shift_thres_list[-2]))
        plt.imshow(self.pose_aps[-1, :-1, :-1], cmap='jet', interpolation='bilinear')
        output_path = os.path.join(log_dir,'mAP_{}-{}degree_{}-{}cm.png'.format(self.degree_thres_list[0], self.degree_thres_list[-2], 
                                                                                self.shift_thres_list[0], self.shift_thres_list[-2]))
        output_name='mAP_{}-{}degree_{}-{}cm'.format(self.degree_thres_list[0], self.degree_thres_list[-2], 
                                                                                self.shift_thres_list[0], self.shift_thres_list[-2])
        plt.colorbar()
        #plt.savefig(output_path)
        log({output_name: wandb.Image(fig_pose)})
        plt.close(fig_pose)

        
        fig_rot = plt.figure()
        ax_rot = plt.subplot(111)
        plt.ylabel('AP')
        plt.ylim((0, 1.05))
        plt.xlabel('translation/cm')
        for cls_id in range(1, self.num_classes):
            class_name = self.synset_names[cls_id]
            print(class_name)
            ax_rot.plot(self.shift_thres_list[:-1], self.pose_aps[cls_id, -1, :-1], label=class_name)
        
        ax_rot.plot(self.shift_thres_list[:-1], self.pose_aps[-1, -1, :-1], label='mean')
        output_path = os.path.join(log_dir,'mAP_{}-{}cm.png'.format(self.shift_thres_list[0], self.shift_thres_list[-2]))
        output_name='mAP_{}-{}cm.png'.format(self.shift_thres_list[0], self.shift_thres_list[-2])
        ax_rot.legend()
        #fig_rot.savefig(output_path)
        log({output_name: wandb.Image(fig_rot)})
        plt.close(fig_rot)

        fig_trans = plt.figure()
        ax_trans = plt.subplot(111)
        plt.ylabel('AP')
        plt.ylim((0, 1.05))

        plt.xlabel('Rotation/degree')
        for cls_id in range(1, self.num_classes):
            class_name = self.synset_names[cls_id]
            ax_trans.plot(self.degree_thres_list[:-1], self.pose_aps[cls_id, :-1, -1], label=class_name)

        ax_trans.plot(self.degree_thres_list[:-1], self.pose_aps[-1, :-1, -1], label='mean')
        output_path = os.path.join(log_dir,'mAP_{}-{}degree.png'.format(self.degree_thres_list[0], self.degree_thres_list[-2]))
        output_name='mAP_{}-{}degree.png'.format(self.degree_thres_list[0], self.degree_thres_list[-2])
        ax_trans.legend()
        fig_trans.savefig(output_path)
        log({output_name: wandb.Image(fig_trans)})
        plt.close(fig_trans)



    def get_pose_aps(self, rot, shift, run_idx=None):
        deg = self.degree_thres_list.index(rot)
        cm = self.shift_thres_list.index(shift)
        if run_idx: return self.pose_aps[run_idx, deg, cm] * 100
        else: return np.mean(self.pose_aps[:, deg, cm]) * 100

    def get_3d_ious(self, thresh, run_idx=None):
        t = self.iou_thres_list.index(thresh)
        if run_idx: return self.iou_3d_aps[run_idx, t] * 100
        else: return np.mean(self.iou_3d_aps[:, t]) * 100