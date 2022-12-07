import torch
import os
import os.path as osp
import shutil

# def load_state_dict(weights, map_location=None):
#     """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
#     ckpt = torch.load(weights, map_location=map_location)
#     state_dict = ckpt['model'].float().state_dict()
#     return model


def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """ Save checkpoint to the disk."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + '.pt')
    torch.save(ckpt, filename)
    if is_best:
        best_filename = osp.join(save_dir, 'best_ckpt.pt')
        shutil.copyfile(filename, best_filename)


if __name__ =="__main__":
    ckpt = torch.load("/home/ubuntu/yxd/yolov6s.pt")
    print(ckpt)
    weights = ckpt['model'].float().state_dict()
    torch.save(weights, "/home/ubuntu/yxd/xgen.pt")
    print("yolov6s2 done")