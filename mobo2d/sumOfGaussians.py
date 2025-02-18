import torch

class sumOfGaussians:
    def __init__(self, d, n_maxima, global_mag, local_mag, alpha, beta, gamma):
        self.d = d
        self.global_mag = global_mag
        self.local_mag = local_mag
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.global_loc = torch.DoubleTensor(d).uniform_(-1,1)
        self.local_loc = [torch.DoubleTensor(d).uniform_(-1,1) for _ in range(n_maxima)]

    def eval(self, X):
        # First ensure all dimensions are correct
        assert X.shape[1] == self.d, f"X must have {self.d} columns (one for each dimension), but got shape {X.shape}."
        assert self.global_loc.shape[0] == self.d, f"global_max_loc must have {self.d} elements, but got shape {self.global_loc.shape}."
        assert all(loc.shape[0] == self.d for loc in self.local_loc), (f"Each local maximum must have {self.d} elements, but one or more maxima do not match.")

        # Now assemble the function
        global_peak = self.global_mag * torch.exp(-torch.norm(X - self.global_loc, dim=1)**2 / self.alpha)
        local_peaks = sum(self.local_mag * torch.exp(-torch.norm(X - loc, dim=1)**2 / self.beta) for loc in self.local_loc)
        background = self.gamma * torch.sin(torch.sum(X * torch.pi, dim = 1))

        return (global_peak + local_peaks + background).reshape(-1,1)