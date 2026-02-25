from data.dataset_base import EmbeddingDataset, DatasetBase
import numpy as np
from typing import Tuple
from omegaconf import DictConfig
from pathlib import Path
import polars as pl

class N24News(DatasetBase):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.load_data(dataset_path)

    def load_data(self, dataset_path: str) -> None:
        json_path = Path(dataset_path) / "news" / "nytimes_dataset.json"
        table = pl.read_json(json_path)
     
        table = table.with_columns(
            pl.col("image_id").map_elements(
                # CHANGED "images" to "imgs" here:
                lambda x: str(Path(dataset_path) / "imgs" / f"{x}.jpg"), return_dtype=pl.String
            ).alias("image_path")
        )
        
        table = table.with_row_index(name="image_idx").with_columns(
            pl.col("headline").alias("captions")
        ).with_row_index(name="caption_idx")
        
        self.table = table

class N24NewsRetrievalDataset(N24News, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        N24News.__init__(self, dataset_path=task_config.dataset_path)
        self.image_paths = self.table.select("image_path").to_series().to_list()
        
        if task_config.generate_embedding:
            self.captions = self.table.select("captions").to_series().to_list()
        else:
            EmbeddingDataset.__init__(self, split=task_config.split)
            self.captions = self.table.select("captions").to_series().to_list()
            
       
            self.num_caption_per_image = 1
            self.num_image_per_caption = 1
            self.img_caption_mapping = self.table.select("image_idx", "caption_idx").rename({"image_idx": "image_id", "caption_idx": "caption_id"}).to_dicts()
            
            self.load_two_encoder_data(
                hf_repo_id=task_config.hf_repo_id, 
                hf_img_embedding_name=task_config.hf_img_embedding_name, 
                hf_text_embedding_name=task_config.hf_text_embedding_name
            )
            if self.split in ["train", "large"]:
                self.set_train_test_split_index(train_test_ratio=task_config.train_test_ratio, seed=task_config.seed)
                self.get_training_paired_embeddings()
    
    def get_training_paired_embeddings(self) -> None:
        text_emb, image_emb = [], []
        
        if self.split == "train":
            for item in self.img_caption_mapping:
                text_emb.append(self.text_embeddings[item["caption_id"]].reshape(1, -1))
                image_emb.append(self.image_embeddings[item["image_id"]].reshape(1, -1))
        elif self.split == "large" and self.train_idx is not None:
            for idx in self.train_idx:
                text_emb.append(self.text_embeddings[self.img_caption_mapping[idx]["caption_id"]].reshape(1, -1))
                image_emb.append(self.image_embeddings[self.img_caption_mapping[idx]["image_id"]].reshape(1, -1))
        else:
            raise ValueError("Please set split to 'train' or get the train/test split index first.")
                
        self.support_embeddings["train_image"] = np.concatenate(image_emb, axis=0)
        self.support_embeddings["train_text"] = np.concatenate(text_emb, axis=0)

    def get_test_data(self):
        self.text_to_image_gt_ids = {}
        self.image_to_text_gt_ids = {}
        
        if self.split == "large":
            val_text_idx = []
            for idx, image_id in enumerate(self.val_idx):
                val_text_idx.append(self.img_caption_mapping[image_id]["caption_id"])
                self.image_to_text_gt_ids[idx] = [idx]
                self.text_to_image_gt_ids[idx] = [idx]
            val_image_embeddings = self.image_embeddings[self.val_idx]
            val_text_embeddings = self.text_embeddings[val_text_idx]

        elif self.split == "val":
            val_image_embeddings = self.image_embeddings
            val_text_embeddings = self.text_embeddings
            for item in self.img_caption_mapping:
                self.image_to_text_gt_ids[item["image_id"]] = [item["caption_id"]]
                self.text_to_image_gt_ids[item["caption_id"]] = [item["image_id"]] 
        else:
            raise ValueError("Please set split to 'train', 'val' or 'large'.")

        return val_image_embeddings, val_text_embeddings, self.image_to_text_gt_ids, self.text_to_image_gt_ids

    def get_support_embeddings(self):
        return self.support_embeddings
