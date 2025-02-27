import torch
import pickle
import time
import csv
import json

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples

from botorch import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

import warnings
from botorch.exceptions.warnings import NumericsWarning, BadInitialCandidatesWarning, InputDataWarning, OptimizationWarning
warnings.filterwarnings("ignore", category = NumericsWarning)
warnings.filterwarnings("ignore", category = BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category = InputDataWarning)
warnings.filterwarnings("ignore", category = OptimizationWarning)
warnings.filterwarnings("ignore", category = RuntimeWarning)

from sumOfGaussians import sumOfGaussians

# Load the individual objective functions
with open('data64/func1.pkl', 'rb') as file:
    func1 = pickle.load(file)
with open('data64/func2.pkl', 'rb') as file:
    func2 = pickle.load(file)

# Use GPU if possible
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Parameters
BOUNDS = torch.tensor([[-1, -1], [1, 1]], **tkwargs)
REFERENCE = torch.tensor([-1,-1], **tkwargs)
NUM_RESTARTS = 20
RAW_SAMPLES = 512

ACQUISITION_FUNCTIONS = ["Random", "qLogEHVI", "qLogNEHVI", "qLogNParEGO"]
EXPERIMENTS = [(1, 100), (5, 20), (10, 10)]  # (batch_size, n_iter)

# Output CSV file
CSV_FILE = "results.csv"

# Use GPU if available

def multi_objective(X):
    # Accepts X as a batch_size x dim tensor
    # Returns a batch_size x 2 tensor
    return torch.cat([func1.eval(X), func2.eval(X)], dim = 1)

def initialize_model(train_X, train_Y):
    models = []
    for i in range(train_Y.shape[-1]):
        train_y = train_Y[..., i:i+1]
        models.append(SingleTaskGP(
            train_X = train_X,
            train_Y = train_y,
            input_transform = Normalize(d = train_X.shape[-1]),
            outcome_transform = Standardize(m = 1)
        ))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def get_acq(acq_name, model, train_X):
    if acq_name == "qLogEHVI":
        pred = model.posterior(train_X).mean
        partitioning = FastNondominatedPartitioning(REFERENCE, pred)
        acq = qLogExpectedHypervolumeImprovement(model, REFERENCE, partitioning)
    elif acq_name == "qLogNEHVI":
        acq = qLogNoisyExpectedHypervolumeImprovement(model, REFERENCE, train_X, prune_baseline = True)
    elif acq_name == "qLogNParEGO":
        acq = qLogNParEGO(model, train_X, prune_baseline = True)
    elif acq_name == "Random":
        acq = None
    else:
        raise ValueError(f"Unknown acquisition function: {acq_name}")
    
    return acq

def step_mobo(acq_name, model, train_X, batch_size):
    if acq_name == "Random":
        candidates = draw_sobol_samples(BOUNDS, batch_size, 1).squeeze(1)
    else:
        acq = get_acq(acq_name, model, train_X)
        candidates, _ = optimize_acqf(
            acq_function = acq,
            bounds = BOUNDS,
            q = batch_size,
            num_restarts = NUM_RESTARTS,
            raw_samples = RAW_SAMPLES, 
            sequential = True,
            options = {"batch_limit": 5, "maxiter": 200},
        )

    new_X = candidates.detach()
    new_Y = multi_objective(candidates)

    return new_X, new_Y

def run_bayesian_opt(acq_name, func, init_X, batch_size, n_iter):
    # Generate the initial objective function data
    init_Y = func(init_X)

    # Initialize the model with the initial data
    mll, model = initialize_model(init_X, init_Y)
    
    # Compute the hypervolume of the initial data
    bd = DominatedPartitioning(ref_point = REFERENCE, Y = init_Y)
    hvs = [bd.compute_hypervolume().item()]
    times = [0.0]

    # Create training data tensors (a bit redundant)
    train_X = init_X
    train_Y = init_Y

    # Bayesian optimization loop
    for i in range(1, n_iter + 1):
        start_time = time.time()

        # Fit the model
        fit_gpytorch_mll(mll)

        # Select new candidates
        new_X, new_Y = step_mobo(acq_name, model, train_X, batch_size)

        # Clear CUDA cache after acquisition optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Append to training tensors
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])

        # Compute hypervolume
        bd = DominatedPartitioning(ref_point = REFERENCE, Y = train_Y)
        hvs.append(bd.compute_hypervolume().item())

        iteration_time = time.time() - start_time
        times.append(iteration_time)

        # Reinitialize models
        mll, model = initialize_model(train_X, train_Y)

        print(f'[{acq_name}, batch={batch_size}, iter={i}/{n_iter}] HV = {hvs[-1]:.5f}, Time = {iteration_time:.2f} sec')

    return hvs, times
    
def save_results(acq_name, batch_size, n_iter, hvs, times):
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([acq_name, batch_size, n_iter, json.dumps(hvs), json.dumps(times)])

def main():
    init_X = torch.load('data64/train.pt').to(**tkwargs)
    func = multi_objective

    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Acquisition", "Batch Size", "Iterations", "Hypervolumes", "Times"])

    for acq_name in ACQUISITION_FUNCTIONS:
        for batch_size, n_iter in EXPERIMENTS:
            hvs, times = run_bayesian_opt(acq_name, func, init_X, batch_size, n_iter)
            save_results(acq_name, batch_size, n_iter, hvs, times)

if __name__ == "__main__":
    main()