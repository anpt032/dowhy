import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from dowhy.causal_prediction_selfcode.algorithms.utils import mmd_compute


class Regularizer:

    def __init__(self, E_conditioned, ci_test, kernel_type, gamma):
        self.E_conditioned = E_conditioned
        self.ci_test = ci_test
        self.kernel_type = kernel_type
        self.gamma = gamma

    def mmd(self, x, y):
        return mmd_compute(x, y, self.kernel_type, self.gamma)
    
    def unconditional_reg(self, classifs, attribute_labels, num_envs, E_eq_A=False):
        penalty = 0

        if E_eq_A:
            if self.E_conditioned is False:
                for i in range(num_envs):
                    for j in range(i+1, num_envs):
                        penalty += self.mmd(classifs[i], classifs[j])
        else:
            if self.E_conditioned:
                for i in range(num_envs):
                    unique_attr_labels = torch.unique(attribute_labels[i])
                    unique_attr_label_indices = []
                    for label in unique_attr_labels:
                        label_ind = [ind for ind, j in enumerate(attribute_labels[i]) if j == label]
                        unique_attr_label_indices.append(label_ind)

                    nulabels = unique_attr_labels.shape[0]
                    for aidx in range(nulabels):
                        for bidx in range(aidx+1, nulabels):
                            penalty += self.mmd(
                                classifs[i][unique_attr_label_indices[aidx]],
                                classifs[i][unique_attr_label_indices[bidx]],
                            )
            else: 
                overall_nmb_indices, nmb_id = [], []
                for i in range(num_envs):
                    unique_attrs = torch.unique(attribute_labels[i])
                    unique_attr_indices = []
                    for attr in unique_attrs:
                        attr_ind = [ind for ind, j in enumerate(attribute_labels[i]) if j == attr]
                        unique_attr_indices.append(attr_ind)
                        overall_nmb_indices.append(attr_ind)
                        nmb_id.append(i)

                nuattr = len(overall_nmb_indices)
                for aidx in range(nuattr):
                    for bidx in range(aidx+1, nuattr):
                        a_nmb_id = nmb_id[aidx]
                        b_nmb_id = nmb_id[bidx]
                        penalty += self.mmd(
                            classifs[a_nmb_id][overall_nmb_indices[aidx]],
                            classifs[b_nmb_id][overall_nmb_indices[bidx]],
                        )

        return  penalty
    
    def conditional_reg(self, classifs, attribute_labels, conditioning_subset, num_envs, E_eq_A=False):
        penalty = 0

        if E_eq_A:
            if self.E_conditioned is False:
                overall_group_vindices = {}
                overall_group_eindices = {}

                for i in range(num_envs):
                    conditioning_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                    conditioning_subset_i_uniform = [
                        ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in conditioning_subset_i
                    ]
                    grouping_data = torch.cat(conditioning_subset_i_uniform, 1)
                    assert grouping_data.min() >= 0, "Group numbers cannot be negative."
                    cardinality = 1 + torch.max(grouping_data, dim=0)[0]
                    cumprod = torch.cumprod(cardinality, dim=0)
                    n_groups = cumprod[-1].item()

                    if torch.cuda.is_available():
                        factors = torch.cat((torch.tensor([1], device='cuda'), cumprod[:-1].to('cuda')))
                        factors = factors.double()

                        group_indices = grouping_data.double().cuda() @ factors
                    else:
                        factors_np = np.concatenate(([1], cumprod[:-1]))
                        factors = torch.from_numpy(factors_np)

                        group_indices = grouping_data @ factors

                    for group_idx in range(n_groups):
                        group_idx_indices = [
                            gp_idx for gp_idx in range(len(group_indices)) if group_indices[gp_idx] == group_idx
                        ]

                        if group_idx not in overall_group_vindices:
                            overall_group_vindices[group_idx] = {}
                            overall_group_eindices[group_idx] = {}

                        unique_attrs = torch.unique(
                            attribute_labels[i][group_idx_indices]
                        )
                        unique_attr_indices = []
                        for attr in unique_attrs:
                            if attr not in overall_group_vindices[group_idx]:
                                overall_group_vindices[group_idx][attr] = []
                                overall_group_eindices[group_idx][attr] = []
                            single_attr = []
                            for group_idx_indices_attr in group_idx_indices:
                                if attribute_labels[i][group_idx_indices_attr] == attr:
                                    single_attr.append(group_idx_indices_attr)
                            overall_group_vindices[group_idx][attr].append(single_attr)
                            overall_group_eindices[group_idx][attr].append(i)
                            unique_attr_indices.append(single_attr)

                for (group_label) in (overall_group_vindices):
                    tensors_list = []
                    for attr in overall_group_vindices[group_label]:
                        attrs_list = []
                        if overall_group_vindices[group_label][attr] != []:
                            for il_ind, indices_list in enumerate(overall_group_vindices[group_label][attr]):
                                attrs_list.append(
                                    classifs[overall_group_eindices[group_label][attr][il_ind]][indices_list]
                                )
                        if len(attrs_list) > 0:
                            tensor_attrs = torch.cat(attrs_list, 0)
                            tensors_list.append(tensor_attrs)

                    nuattr = len(tensors_list)
                    for aidx in range(nuattr):
                        for bidx in range(aidx+1, nuattr):
                            penalty += self.mmd(tensors_list[aidx], tensors_list[bidx])
        else:
            if self.E_conditioned:
                for i in range(num_envs):
                    conditioning_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                    conditioning_subset_i_uniform = [
                        ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in conditioning_subset_i
                    ]
                    grouping_data = torch.cat(conditioning_subset_i_uniform, 1)
                    assert grouping_data.min() >= 0, "Group numbers cannot be negative."
                    cardinality = 1 + torch.max(grouping_data, dim=0)[0]
                    cumprod = torch.cumprod(cardinality, dim=0)
                    n_groups = cumprod[-1].item()

                    if torch.cuda.is_available():
                        factors = torch.cat((torch.tensor([1], device='cuda'), cumprod[:-1].to('cuda')))
                        factors = factors.double()

                        group_indices = grouping_data.double().cuda() @ factors
                    else:
                        factors_np = np.concatenate(([1], cumprod[:-1]))
                        factors = torch.from_numpy(factors_np)

                        group_indices = grouping_data @ factors

                    for group_idx in range(n_groups):
                        group_idx_indices = [
                            gp_idx for gp_idx in range(len(group_indices)) if group_indices[gp_idx] == group_idx
                        ]
                        unique_attrs = torch.unique(
                            attribute_labels[i][group_idx_indices]
                        )
                        unique_attr_indices = []
                        for attr in unique_attrs:
                            single_attr = []
                            for group_idx_indices_attr in group_idx_indices:
                                if attribute_labels[i][group_idx_indices_attr] == attr:
                                    single_attr.append(group_idx_indices_attr)
                            unique_attr_indices.append(single_attr)
                        
                        nuattr = unique_attrs.shape[0]
                        for aidx in range(nuattr):
                            for bidx in range(aidx+1, nuattr):
                                penalty += self.mmd(
                                    classifs[i][unique_attr_indices[aidx]], 
                                    classifs[i][unique_attr_indices[bidx]]
                                )
            
            else:
                overall_group_vindices = {}
                overall_group_eindices = {}

                for i in range(num_envs):
                    conditioning_subset_i = [subset_var[i] for subset_var in conditioning_subset]
                    conditioning_subset_i_uniform = [
                        ele.unsqueeze(1) if ele.dim() == 1 else ele for ele in conditioning_subset_i
                    ]
                    grouping_data = torch.cat(conditioning_subset_i_uniform, 1)
                    assert grouping_data.min() >= 0, "Group numbers cannot be negative."
                    cardinality = 1 + torch.max(grouping_data, dim=0)[0]
                    cumprod = torch.cumprod(cardinality, dim=0)
                    n_groups = cumprod[-1].item()

                    if torch.cuda.is_available():
                        factors = torch.cat((torch.tensor([1], device='cuda'), cumprod[:-1].to('cuda')))
                        factors = factors.double()

                        group_indices = grouping_data.double().cuda() @ factors
                    else:
                        factors_np = np.concatenate(([1], cumprod[:-1]))
                        factors = torch.from_numpy(factors_np)

                        group_indices = grouping_data @ factors

                    for group_idx in range(n_groups):
                        group_idx_indices = [
                            gp_idx for gp_idx in range(len(group_indices)) if group_indices[gp_idx] == group_idx
                        ]

                        if group_idx not in overall_group_vindices:
                            overall_group_vindices[group_idx] = {}
                            overall_group_eindices[group_idx] = {}

                        unique_attrs = torch.unique(
                            attribute_labels[i][group_idx_indices]
                        )
                        unique_attr_indices = []
                        for attr in unique_attrs:
                            if attr not in overall_group_vindices[group_idx]:
                                overall_group_vindices[group_idx][attr] = []
                                overall_group_eindices[group_idx][attr] = []
                            single_attr = []
                            for group_idx_indices_attr in group_idx_indices:
                                if attribute_labels[i][group_idx_indices_attr] == attr:
                                    single_attr.append(group_idx_indices_attr)
                            overall_group_vindices[group_idx][attr].append(single_attr)
                            overall_group_eindices[group_idx][attr].append(i)
                            unique_attr_indices.append(single_attr)

                for (group_label) in (overall_group_vindices):
                    tensors_list = []
                    for attr in overall_group_vindices[group_label]:
                        attrs_list = []
                        if overall_group_vindices[group_label][attr] != []:
                            for il_ind, indices_list in enumerate(overall_group_vindices[group_label][attr]):
                                attrs_list.append(
                                    classifs[overall_group_eindices[group_label][attr][il_ind]][indices_list]
                                )
                        if len(attrs_list) > 0:
                            tensor_attrs = torch.cat(attrs_list, 0)
                            tensors_list.append(tensor_attrs)

                    nuattr = len(tensors_list)
                    for aidx in range(nuattr):
                        for bidx in range(aidx + 1, nuattr):
                            penalty += self.mmd(tensors_list[aidx], tensors_list[bidx])

        return penalty
