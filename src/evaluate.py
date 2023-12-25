import torch
import numpy as np
import kaolin.metrics.pointcloud as pt

print(torch.cuda.is_available())

# pack point clouds into tensors
out_point_cloud_tensor = torch.tensor(np.expand_dims(out_point_cloud, 0), device='cuda:0', dtype=torch.float32)
gt_point_cloud_tensor = torch.tensor(np.expand_dims(gt_point_cloud, 0), device='cuda:0', dtype=torch.float32)
out_point_cloud_edge_tensor = torch.tensor(np.expand_dims(out_point_cloud_edge, 0), device='cuda:0', dtype=torch.float32)
gt_point_cloud_edge_tensor = torch.tensor(np.expand_dims(gt_point_cloud_edge, 0), device='cuda:0', dtype=torch.float32)

# Chamfer distance
eval_metrics.chamfer_distance = pt.chamfer_distance(out_point_cloud_tensor, gt_point_cloud_tensor)

# F1-score
eval_metrics.f1_score = pt.f_score(out_point_cloud_tensor, gt_point_cloud_tensor)

# Edge Chamfer distance
eval_metrics.edge_chamfer_distance = pt.chamfer_distance(out_point_cloud_edge_tensor, gt_point_cloud_edge_tensor)

# Edge F1-score
eval_metrics.edge_f1_score = pt.f_score(out_point_cloud_edge_tensor, gt_point_cloud_edge_tensor)
