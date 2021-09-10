



class MILNCELoss(torch.nn.Module):
    """The loss implemented is: log(pos/(2 * pos + neg)) = log(pos/(pos + neg/2)) - log(2).
    See: https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/loss.py
         https://github.com/antoine77340/MIL-NCE_HowTo100M/issues/4
    """
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, V, A):
        """Returns MIL-NCE loss.
        Args:
           V: Tensor of embeddings (e.g. video)
           A: Tensor of embeddings (e.g. audio)
        """
        x = torch.matmul(V, A.t())
        x = x.view(V.shape[0], V.shape[0], -1)
        numerator = x * torch.eye(x.shape[0])[:,:,None].to(x.device)
        numerator = numerator.sum(dim=1)
        numerator = torch.logsumexp(numerator, dim=1)
        denominator = torch.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - numerator)

class TripletLoss(torch.nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, V, A):
        """Returns Triplet loss with margin.
        Args:
           V: Tensor of embeddings (e.g. video)
           A: Tensor of embeddings (e.g. audio)
        """
        return contrastive(cosine_matrix(V, A), margin=self.margin)

def contrastive(M, margin=0.2):
    "Returns contrastive margin loss over similarity matrix M."
    E = - M
    D = torch.diag(E)
    C_c = torch.clamp(margin - E + D, min=0)
    C_r = torch.clamp(margin - E + D.view(-1, 1), min=0)
    C = C_c + C_r
    return (C.sum() - torch.diag(C).sum())/C.size(0)**2


def cosine_matrix(U, V):
    "Returns the matrix of cosine similarity between each row of U and each row of V."
    U_norm = U / U.norm(2, dim=1, keepdim=True)
    V_norm = V / V.norm(2, dim=1, keepdim=True)
    return torch.matmul(U_norm, V_norm.t())
        


                
