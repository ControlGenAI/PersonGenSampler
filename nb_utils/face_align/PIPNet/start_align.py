import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.append('./../../')
from nb_utils.face_align.PIPNet.alignment.dataloader import FolderAlignCrop
from nb_utils.face_align.PIPNet.alignment.gen_pickle import gen_pickle_abs


def set_random_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_arr(tensor):
    arr = ((tensor + 1.) * 127.5).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return arr


def main(args):
    set_random_seed(0)
    torch.multiprocessing.set_start_method('spawn')

    folder_dataset = FolderAlignCrop(
        args.in_folder,
        image_size=args.out_size,
        align_mode=args.align_mode
    )
    batch_size = args.batch_size
    folder_loader = DataLoader(
        folder_dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )

    os.makedirs(args.out_folder, exist_ok=True)

    img_cnt = 0
    batch_idx = 0
    dataset_list = []
    for batch in tqdm(folder_loader, desc="Detect and Align"):
        batch_idx += 1
        i_t = batch["target_image"]
        has_lmk_box = batch["has_lmk_box"]
        img_name = batch["img_name"]

        arr_t = tensor_to_arr(i_t)
        for b in range(batch_size):
            if not has_lmk_box[b]:
                print('skip image %d' % (batch_idx - 1))
                continue
            img_t = Image.fromarray(arr_t[b])
            if not args.pickle_only:
                img_t.save(os.path.join(args.out_folder, img_name[b]))
            dataset_list.append(os.path.join(args.out_folder, img_name[b]))
            img_cnt += 1

    if args.out_pickle is None:
        args.out_pickle = "{}.pickle".format(args.out_folder)
    gen_pickle_abs(args.out_folder, args.out_pickle, dataset_list=dataset_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, required=True)
    parser.add_argument("--pickle_only", action='store_true', default=False)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--out_pickle", type=str, default=None)
    parser.add_argument("--out_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--align_mode", type=str, default="ffhq", choices=['ffhq', 'set1', 'arcface'])
    opt = parser.parse_args()

    main(opt)
