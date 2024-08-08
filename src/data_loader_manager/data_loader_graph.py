import functools
import json
import logging
import os
import os.path as osp
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.transforms as T
from easydict import EasyDict
from torch_geometric.data import HeteroData
from torch_geometric.datasets import (
    DBLP,
    Amazon,
    AMiner,
    Coauthor,
    FB15k_237,
    Planetoid,
    WebKB,
    WikiCS,
    WikipediaNetwork,
    WordNet18RR,
)
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddMetaPaths, RandomNodeSplit
from torch_geometric.utils import to_torch_sparse_tensor

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from utils.dirs import load_file
from utils.functions import create_split_masks

sys.dont_write_bytecode = True


logger = logging.getLogger(__name__)


class DataLoaderForGraph(DataLoaderWrapper):
    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)

    def LoadTwitterData(self, module_config):
        data_path = osp.join(self.config.DATA_FOLDER, module_config.config.path)
        save_or_load_path = osp.join(
            data_path, "processed", f"{module_config.config.save_or_load_name}.pt"
        )
        data_dict = EasyDict({})
        if (
            osp.exists(save_or_load_path)
            and module_config.option == "default"
            and not self.config.reset_data
        ):
            data = torch.load(save_or_load_path)
        else:
            os.makedirs(osp.join(data_path, "processed"), exist_ok=True)
            raw_data_path = osp.join(data_path, "raw")

            raw_edges = load_file(osp.join(raw_data_path, "edges.json"))
            keyword_dict = load_file(osp.join(raw_data_path, "keyword_embeddings.pkl"))
            tweet_dict = load_file(osp.join(raw_data_path, "tweet_embeddings.pkl"))
            user_dict = load_file(osp.join(raw_data_path, "user_embeddings.pkl"))
            user_labels = load_file(osp.join(raw_data_path, "labels.json"))

            user_ids, user_embeddings = user_dict["ids"], user_dict["embeddings"]
            tweet_ids, tweet_embeddings = tweet_dict["ids"], tweet_dict["embeddings"]
            keyword_ids, keyword_embeddings = (
                keyword_dict["ids"],
                keyword_dict["embeddings"],
            )
            ### ID Mapping ###
            id_mapping = {}
            count_dict = {"user": 0, "tweet": 0, "keyword": 0}
            for id in user_ids + tweet_ids + keyword_ids:
                node_type = id.split("_")[0]
                id_mapping[id] = count_dict[node_type]
                count_dict[node_type] += 1

            ### Preprocess the labels ###
            label_mapping = {"Negative": 0, "Seller": 1, "Buyer": 2, "Related": 3}
            labels = []
            for user_id, label in user_labels.items():
                assert id_mapping[user_id] == len(labels)
                labels.append(label_mapping[label])

            ### Preprocess the edges ###
            relations = defaultdict(set)

            if "build_metapath_for_MetaPath2Vec" in module_config.config.preprocess:
                for edge in raw_edges:
                    src_node_id = id_mapping[edge["source_id"]]
                    tar_node_id = id_mapping[edge["target_id"]]
                    relation_type = edge["relation"]
                    src_node_type = edge["source_id"].split("_")[0]
                    tar_node_type = edge["target_id"].split("_")[0]
                    triple = relation_type.split("-")
                    assert src_node_type == triple[0] and tar_node_type == triple[2]

                    relations[(src_node_type, "to", tar_node_type)].add(
                        (src_node_id, tar_node_id)
                    )
                    relations[(tar_node_type, "to", src_node_type)].add(
                        (tar_node_id, src_node_id)
                    )
            else:
                for edge in raw_edges:
                    src_node_id = id_mapping[edge["source_id"]]
                    tar_node_id = id_mapping[edge["target_id"]]
                    relation_type = edge["relation"]
                    src_node_type = edge["source_id"].split("_")[0]
                    tar_node_type = edge["target_id"].split("_")[0]
                    triple = relation_type.split("-")
                    assert src_node_type == triple[0] and tar_node_type == triple[2]

                    relations[(src_node_type, triple[1] + "-->", tar_node_type)].add(
                        (src_node_id, tar_node_id)
                    )
                    relations[(tar_node_type, "<--" + triple[1], src_node_type)].add(
                        (tar_node_id, src_node_id)
                    )
            edge_index_dict = {}
            for key, edges in relations.items():
                src_node_type, relation_type, tar_node_type = key
                src_node_ids, tar_node_ids = zip(*edges)
                src_node_ids = torch.tensor(src_node_ids)
                tar_node_ids = torch.tensor(tar_node_ids)
                tmp_edge_index = torch.stack([src_node_ids, tar_node_ids], dim=0)
                sorted_indices = torch.argsort(tmp_edge_index[0, :])
                sorted_edge_index = tmp_edge_index[:, sorted_indices]
                unique_edges = torch.unique(sorted_edge_index, dim=1)
                edge_index_dict[key] = unique_edges

            ###  Initialize the data object ###
            data = HeteroData()
            data["user"].x = torch.tensor(user_embeddings)
            data["keyword"].x = torch.tensor(keyword_embeddings)
            data["tweet"].x = torch.tensor(tweet_embeddings)
            data["user"].y = torch.tensor(labels)

            for (src_type, relation_type, dst_type), edges in edge_index_dict.items():
                data[src_type, relation_type, dst_type].edge_index = edges.contiguous()

            if "build_baseline" in module_config.config.preprocess:
                edge_types = data.edge_types
                labels = data["user"].y
                pos = [[], []]
                # pos_dict = defaultdict(int)
                nei_t = [set() for i in range(data["user"].num_nodes)]
                nei_k = [set() for i in range(data["user"].num_nodes)]
                for edge_type in edge_types:
                    if edge_type[0] == "user" and edge_type[2] == "user":
                        row, col = data[edge_type].edge_index
                        for row_id, col_id in zip(row, col):
                            pos[0].append(row_id.item())
                            pos[1].append(col_id.item())
                            pos[0].append(col_id.item())
                            pos[1].append(row_id.item())
                    elif edge_type[0] == "user" and edge_type[2] == "tweet":
                        row, col = data[edge_type].edge_index
                        for row_id, col_id in zip(row, col):
                            nei_t[row_id].add(col_id.item())
                    elif edge_type[2] == "user" and edge_type[0] == "tweet":
                        row, col = data[edge_type].edge_index
                        for row_id, col_id in zip(row, col):
                            nei_t[col_id].add(row_id.item())
                    elif edge_type[0] == "user" and edge_type[2] == "keyword":
                        row, col = data[edge_type].edge_index
                        for row_id, col_id in zip(row, col):
                            nei_k[row_id].add(col_id.item())
                    elif edge_type[2] == "user" and edge_type[0] == "keyword":
                        row, col = data[edge_type].edge_index
                        for row_id, col_id in zip(row, col):
                            nei_k[col_id].add(row_id.item())
                # pos = [torch.LongTensor(list(p)) for p in pos]
                pos = torch.LongTensor(pos)
                pos = to_torch_sparse_tensor(pos).coalesce()
                nei_k = [torch.LongTensor(list(p)) for p in nei_k]
                nei_t = [torch.LongTensor(list(p)) for p in nei_t]
                # assert len(pos) == data["user"].num_nodes
                assert len(nei_k) == data["user"].num_nodes
                assert len(nei_t) == data["user"].num_nodes
                data["user"].pos = pos
                data["user"].nei_index = {
                    "keyword": nei_k,
                    "tweet": nei_t,
                }

            if "build_metapath_from_config" in module_config.config.preprocess:
                metapaths = [
                    [(src, rel, dst) for src, rel, dst in metapath]
                    for metapath in module_config.config.metapaths
                ]
                data = AddMetaPaths(
                    metapaths,
                    max_sample=5,
                    drop_orig_edge_types=False,
                    keep_same_node_type=True,
                )(data)
            elif "build_metapath_drop_orig" in module_config.config.preprocess:
                metapaths = [
                    [(src, rel, dst) for src, rel, dst in metapath]
                    for metapath in module_config.config.metapaths
                ]
                data = AddMetaPaths(
                    metapaths,
                    max_sample=5,
                    drop_orig_edge_types=True,
                    keep_same_node_type=True,
                )(data)
                for edge_type in data.edge_types:
                    if not edge_type[1].startswith("metapath"):
                        del data[edge_type]

            data.target_node_type = "user"
            torch.save(data, save_or_load_path)
            logger.info(data.metadata())
            logger.info(f"Data saved to: {save_or_load_path}")

        dataname = module_config.config.name
        data_dict[dataname.lower()] = data
        self.data.update(data_dict)

    def LoadPositionEmb(self, model_config):
        data_path = osp.join(self.config.DATA_FOLDER, model_config.path)
        file_name = model_config.config.file_name
        save_or_load_path = osp.join(data_path, file_name)
        assert osp.isfile(save_or_load_path), f"{save_or_load_path} does not exist"
        position_emb = torch.load(save_or_load_path)

        use_column = model_config.use_column
        node_type = model_config.config.node_type
        self.data[use_column][node_type].p_x = position_emb
        print()

    def LoadBinaryData(self, module_config):
        use_column = module_config.use_column
        target_node_type = self.config.train.additional.target_node_type
        data = self.data[use_column]
        labels = data[target_node_type].y
        labels = labels > 0
        data[target_node_type].y = labels.long()
        self.data[use_column] = data

    def LoadSplits(self, module_config):
        option = module_config.option
        use_column = module_config.use_column
        target_node_type = self.config.train.additional.target_node_type
        save_or_load_path = osp.join(
            self.config.DATA_FOLDER,
            module_config.path,
            f"split_{module_config.use_column}_{self.config.num_runs}_{module_config.split_ratio.train}_{module_config.split_ratio.valid}_{module_config.split_ratio.test}.pt",
        )
        if (
            osp.exists(save_or_load_path)
            and option == "default"
            and not self.config.reset_data
        ):
            loaded_split_masks = torch.load(save_or_load_path)
            assert len(loaded_split_masks) == self.config.num_runs
        else:
            y_true = self.data[use_column][target_node_type].y
            loaded_split_masks = {}
            for current_run in range(self.config.num_runs):
                loaded_split_masks[current_run] = create_split_masks(
                    y_true, **module_config.split_ratio
                )
            torch.save(loaded_split_masks, save_or_load_path)
        self.splits[use_column] = loaded_split_masks[self.config.current_run]
   
    def LoadDataLoader(self, module_config):
        option = module_config.option
        use_column = module_config.use_column
        target_node_type = self.config.train.additional.target_node_type

        for mode in module_config.config.keys():
            for data_config in module_config.config[mode]:
                use_split = data_config.split
                dataset_type = data_config.dataset_type
                if option == "skip":
                    data_loader = self.data[use_column].clone()
                    loader_name = f"{mode}/{dataset_type}.{use_split}"
                    self.data_loaders[mode][loader_name] = data_loader
                    self.data_loaders[mode][loader_name][
                        target_node_type
                    ].mask = self.splits[use_column][use_split]
                else:
                    data_loader = self.data[use_column].clone()
                    data_loader[target_node_type].mask = self.splits[use_column][
                        use_split
                    ]
                    data_loader = DataLoader([data_loader])
                    loader_name = f"{mode}/{dataset_type}.{use_split}"
                    self.data_loaders[mode][loader_name] = data_loader
                logger.info(
                    f"[Data Statistics]: {mode} data loader: {loader_name} {len(data_loader)}"
                )
