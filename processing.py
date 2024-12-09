import argparse
import logging
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
import json

def imgs2pickle(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False, dataset='CASIAB') -> None:
    """Reads a group of images and saves the data in pickle format and as individual resized and cropped images.

    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    sinfo = img_groups[0]
    img_paths = img_groups[1]
    to_pickle = []
    for img_file in sorted(img_paths):
        if verbose:
            logging.debug(f'Reading sid {sinfo[0]}, seq {sinfo[1]}, view {sinfo[2]} from {img_file}')

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

        if dataset == 'GREW':
            to_pickle.append(img.astype('uint8'))
            continue

        if img.sum() <= 10000:
            if verbose:
                logging.debug(f'Image sum: {img.sum()}')
            logging.warning(f'{img_file} has no data.')
            continue

        # Crop blank space
        y_sum = img.sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
        img = img[y_top: y_btm + 1, :]

        # Resize while maintaining aspect ratio
        ratio = img.shape[1] / img.shape[0]
        img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)

        # Center cropping
        x_csum = img.sum(axis=0).cumsum()
        x_center = None
        for idx, csum in enumerate(x_csum):
            if csum > img.sum() / 2:
                x_center = idx
                break

        if not x_center:
            logging.warning(f'{img_file} has no center.')
            continue

        half_width = img_size // 2
        left = x_center - half_width
        right = x_center + half_width
        if left <= 0 or right >= img.shape[1]:
            left += half_width
            right += half_width
            _ = np.zeros((img.shape[0], half_width))
            img = np.concatenate([_, img, _], axis=1)

        cropped_img = img[:, left: right].astype('uint8')
        to_pickle.append(cropped_img)

        # Save the resized and cropped image as a file
        image_save_path = os.path.join(output_path, *sinfo, f"{Path(img_file).stem}_cropped.png")
        os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
        cv2.imwrite(image_save_path, cropped_img)

    # Save the images as a pickle file
    if to_pickle:
        to_pickle = np.asarray(to_pickle)
        dst_path = os.path.join(output_path, *sinfo)
        os.makedirs(dst_path, exist_ok=True)
        pkl_path = os.path.join(dst_path, f'{sinfo[2]}.pkl')
        if verbose:
            logging.debug(f'Saving {pkl_path}...')
        pickle.dump(to_pickle, open(pkl_path, 'wb'))
        logging.info(f'Saved {len(to_pickle)} valid frames to {pkl_path}.')

    if len(to_pickle) < 5:
        logging.warning(f'{sinfo} has less than 5 valid data.')

def pretreat(input_path: Path, output_path: Path, img_size: int = 64, workers: int = 4, verbose: bool = False, dataset: str = 'CASIAB') -> None:
    """Reads a flat dataset of images and saves the data in pickle format and as processed images.

    Args:
        input_path (Path): Path containing raw image files.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    # Group all images together (single group)
    img_paths = list(input_path.glob('*.png'))
    if not img_paths:
        logging.warning(f"No images found in {input_path}.")
        return

    logging.info(f'Total files found: {len(img_paths)}')

    img_groups = {('all', 'all', 'all'): img_paths}  # Treat all images as a single group

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='group')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(
            partial(
                imgs2pickle,
                output_path=output_path,
                img_size=img_size,
                verbose=verbose,
                dataset=dataset,
            ),
            img_groups.items(),
        ):
            progress.update(1)
    logging.info('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', default='test_data/test_human_images_results', type=str, help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='processed_results', type=str, help='Output path of pickled dataset.')
    parser.add_argument('-l', '--log_file', default='./pretreatment.log', type=str, help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('-n', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('-r', '--img_size', default=64, type=int, help='Image resizing size. Default 64')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Display debug info.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')

    pretreat(input_path=Path(args.input_path), output_path=Path(args.output_path), img_size=args.img_size, workers=args.n_workers, verbose=args.verbose)
