import torch
import torch.nn as nn
from torch import sqrt, abs

mse_loss = nn.MSELoss()

def bboxes_loss(logits, labels):
    #bb_square = (logits[:,:4] - labels[:,:4]) ** 2
    
    loc_sq = (logits[:,:2] - labels[:,:2]) ** 2
    #size_sq = (logits[:,2:4] - labels[:,2:4]) ** 2
    size_sq = (sqrt(abs(logits[:,2:4])) - sqrt(abs(labels[:,2:4]))) ** 2
    obj_sq = (logits[:,4] - labels[:,4]) ** 2
    #bbox_loss = torch.mean(loc_sq) + 0.1*torch.mean(size_sq) + 0.01 * torch.mean(obj_sq)
    bbox_loss = torch.mean(loc_sq) + 0.1*torch.mean(size_sq) + 0.1 * torch.mean(obj_sq)

    #print("loc_sq:", loc_sq)
    #print("size_sq:", size_sq)
    #print("obj_sq:", obj_sq)
    #print("bbox_loss:", bbox_loss)
    

    #loc_sq = (logits[:,:2] - labels[:,:2]) ** 2
    #size_sq = (logits[:,2:4] - 0.1) ** 2
    #obj_sq = (logits[:,4] - labels[:,4]) ** 2
    #bbox_loss = torch.mean(loc_sq) + 0.1*torch.mean(size_sq) + 0.01 * torch.mean(obj_sq)

    #bbox_loss = torch.mean(bb_square)
    #loss2 = mse_loss(logits[:,:4], labels[:,:4])
    #print("bbox_loss:", bbox_loss)
    return bbox_loss


if __name__ == "__main__":

    #logits = torch.tensor([[0, 0, 1, 1, 100], [1, 1, 3, 3, 100]], dtype=torch.float32)
    #labels = torch.tensor([[0.1, 0.1, 0.9, 0.9, 100], [1, 1, 3.2, 3.2, 100]], dtype=torch.float32)
    logits = torch.tensor([[0, 0, 5, 5, 100], [0, 0, 5, 5, 100]], dtype=torch.float32)
    labels = torch.tensor([[1, 1, 1, 1, 100], [1, 1, 1, 1, 100]], dtype=torch.float32)

    print(logits)
    print(labels)

    loss = bboxes_loss(logits, labels)
    print("loss:", loss)