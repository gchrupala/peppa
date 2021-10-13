import torch

def identity(x):
    return x

def cosine_matrix(U, V):
    "Returns the matrix of cosine similarity between each row of U and each row of V."
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return torch.matmul(U_norm, V_norm.t())
        


                
