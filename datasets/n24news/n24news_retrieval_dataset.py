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
     
        # 1. Map Image Paths
        table = table.with_columns(
            pl.col("image_id").map_elements(
                lambda x: str(Path(dataset_path) / "imgs" / f"{x}.jpg"), return_dtype=pl.String
            ).alias("image_path")
        )
        
        # Ensure text columns are strings to prevent concat errors if any are null
        table = table.with_columns([
            pl.col("headline").fill_null("").cast(pl.String),
            pl.col("abstract").fill_null("").cast(pl.String),
            pl.col("caption").fill_null("").cast(pl.String)
        ])
        
        # 2. Base table for images (Length = N)
        self.image_table = table.with_row_index(name="image_idx")
        
        # 3. Exploded table for captions (Length = 3N)
        # Concat the 3 columns into a list, then explode so each image gets 3 rows
        self.caption_table = self.image_table.select(
            "image_idx",
            pl.concat_list(["headline", "abstract", "caption"]).alias("captions")
        ).explode("captions").with_row_index(name="caption_idx")
        
        self.table = table # Keeping original table just in case

class N24NewsRetrievalDataset(N24News, EmbeddingDataset):
    def __init__(self, task_config: DictConfig):
        N24News.__init__(self, dataset_path=task_config.dataset_path)
        
        # Images come from the base table (N distinct paths)
        self.image_paths = self.image_table.select("image_path").to_series().to_list()
        
        if task_config.generate_embedding:
            # Captions come from the exploded table (3N distinct captions)
            self.captions = self.caption_table.select("captions").to_series().to_list()
        else:
            EmbeddingDataset.__init__(self, split=task_config.split)
            self.captions = self.caption_table.select("captions").to_series().to_list()
            
            # Update to reflect the 3-to-1 relationship
            self.num_caption_per_image = 3
            self.num_image_per_caption = 1
            
            # Create the mapping list
            self.img_caption_mapping = self.caption_table.select(
                "image_idx", "caption_idx"
            ).rename({"image_idx": "image_id", "caption_idx": "caption_id"}).to_dicts()
            
            # NEW: Helper dict for O(1) lookups since mapping list index no longer equals image_id
            self.image_to_captions = {}
            for item in self.img_caption_mapping:
                img_id = item["image_id"]
                self.image_to_captions.setdefault(img_id, []).append(item["caption_id"])
            
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
            for img_id in self.train_idx:
                # Use the helper dict to find all 3 caption IDs for this image ID
                for cap_id in self.image_to_captions[img_id]:
                    text_emb.append(self.text_embeddings[cap_id].reshape(1, -1))
                    image_emb.append(self.image_embeddings[img_id].reshape(1, -1))
        else:
            raise ValueError("Please set split to 'train' or get the train/test split index first.")
                
        self.support_embeddings["train_image"] = np.concatenate(image_emb, axis=0)
        self.support_embeddings["train_text"] = np.concatenate(text_emb, axis=0)

    def get_test_data(self):
        self.text_to_image_gt_ids = {}
        self.image_to_text_gt_ids = {}
        
        if self.split == "large":
            val_text_idx = []
            for local_img_idx, img_id in enumerate(self.val_idx):
                cap_ids = self.image_to_captions[img_id]
                
                # Append the 3 caption indices for this validation image
                local_cap_start = len(val_text_idx)
                val_text_idx.extend(cap_ids)
                local_cap_end = len(val_text_idx)
                
                # Map the local test indices for retrieval metrics
                local_cap_ids = list(range(local_cap_start, local_cap_end))
                self.image_to_text_gt_ids[local_img_idx] = local_cap_ids
                for c_id in local_cap_ids:
                    self.text_to_image_gt_ids[c_id] = [local_img_idx]
                    
            val_image_embeddings = self.image_embeddings[self.val_idx]
            val_text_embeddings = self.text_embeddings[val_text_idx]

        elif self.split == "val":
            val_image_embeddings = self.image_embeddings
            val_text_embeddings = self.text_embeddings
            for item in self.img_caption_mapping:
                img_id = item["image_id"]
                cap_id = item["caption_id"]
                self.image_to_text_gt_ids.setdefault(img_id, []).append(cap_id)
                self.text_to_image_gt_ids.setdefault(cap_id, []).append(img_id)
        else:
            raise ValueError("Please set split to 'train', 'val' or 'large'.")

        return val_image_embeddings, val_text_embeddings, self.image_to_text_gt_ids, self.text_to_image_gt_ids

    def get_support_embeddings(self):
        return self.support_embeddings
