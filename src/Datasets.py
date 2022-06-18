import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
import Utils
import random
import zipfile
from io import BytesIO


class BaseDataset:
    def __init__(self, samples, zip_path, tokenizer, mask_prob=0.0, mask_ratio=0.0, is_test=False, fix_length=256):
        self.samples = samples
        self.length = len(samples)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.max_frame = 32
        self.handles = zipfile.ZipFile(zip_path, 'r')
        self.test_mode = is_test
        self.fix_length = fix_length

    def __len__(self):
        return self.length

    def fetch_video_feat(self, vid):
        raw_feats = np.load(BytesIO(self.handles.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        return feat, mask

    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        labels = [self.samples[idx]["labels"]]
        video_input, video_mask = self.fetch_video_feat(self.samples[idx]["vid"])
        # mask argument
        mask_inds = None
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
        if len(input_ids) < self.fix_length:
            input_ids += [self.tokenizer.pad_token_id] * (self.fix_length - len(input_ids))
            attention_mask += [0] * (self.fix_length - len(attention_mask))
        assert len(input_ids) == len(attention_mask)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        if mask_inds is not None:
            tmp = np.array([-100] * self.fix_length)
            input_ids = input_ids.numpy()
            tmp[mask_inds] = input_ids[mask_inds]
            input_ids[mask_inds] = self.tokenizer.mask_token_id
            no_mask_inds = np.where(tmp >= 21128)
            input_ids[no_mask_inds] = tmp[no_mask_inds]
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "video_input": torch.tensor(video_input, dtype=torch.float),
            "video_mask": torch.tensor(video_mask, dtype=torch.long),
        }


class PretrainDatasets(BaseDataset):
    def __init__(self, samples, zip_path, tokenizer, mask_ratio=0.0, is_test=False, fix_length=256):
        super(PretrainDatasets, self).__init__(samples, zip_path, tokenizer, 1.0, mask_ratio, is_test, fix_length)
    
    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        input_ids_r = input_ids.copy()
        attention_mask = self.samples[idx]["attention_mask"]
        video_input, video_mask = self.fetch_video_feat(self.samples[idx]["vid"])
        labels = np.array([-100] * self.fix_length)
        # mask argument
        all_inds = np.arange(1, len(input_ids) - 1)
        n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
        np.random.shuffle(all_inds)
        mask_inds = all_inds[:n_mask]
        if len(input_ids) < self.fix_length:
            input_ids += [self.tokenizer.pad_token_id] * (self.fix_length - len(input_ids))
            input_ids_r += [self.tokenizer.pad_token_id] * (self.fix_length - len(input_ids_r))
            attention_mask += [0] * (self.fix_length - len(attention_mask))
        assert len(input_ids) == len(attention_mask)
        # input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = np.array(input_ids)
        # labels = torch.tensor(labels, dtype=torch.long)
        labels[mask_inds] = input_ids[mask_inds]
        input_ids[mask_inds] = self.tokenizer.mask_token_id
        no_mask_inds = np.where(labels >= 21128)
        input_ids[no_mask_inds] = labels[no_mask_inds]
        labels[no_mask_inds] = -100
        # logging.info(labels)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_ids_r": torch.tensor(input_ids_r, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "video_input": torch.tensor(video_input, dtype=torch.float),
            "video_mask": torch.tensor(video_mask, dtype=torch.long),
        }


class BaseDatasetsValid(BaseDataset):
    def __init__(self, samples, zip_path, tokenizer, mask_ratio=0.0, is_test=True, fix_length=256):
        super(BaseDatasetsValid, self).__init__(samples, zip_path, tokenizer, 0, 0, True, fix_length)
    
    def __getitem__(self, idx):
        input_ids = self.samples[idx]["input_ids"]
        attention_mask = self.samples[idx]["attention_mask"]
        video_input, video_mask = self.fetch_video_feat(self.samples[idx]["vid"])
        if len(input_ids) < self.fix_length:
            input_ids += [self.tokenizer.pad_token_id] * (self.fix_length - len(input_ids))
            attention_mask += [0] * (self.fix_length - len(attention_mask))
        assert len(input_ids) == len(attention_mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "video_input": torch.tensor(video_input, dtype=torch.float),
            "video_mask": torch.tensor(video_mask, dtype=torch.long),
        }