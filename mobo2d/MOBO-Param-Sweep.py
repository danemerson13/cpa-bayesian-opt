import torch
import pickle
import time
import csv
import json

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from botorch import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

import warnings
from botorch.exceptions.warnings import NumericsWarning, BadInitialCandidatesWarning, InputDataWarning, OptimizationWarning
warnings.filterwarnings("ignore", category=NumericsWarning)
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)

# --- Load Data ---
with open('data64/func1.pkl', 'rb') as file:
    func1 = pickle.load(file)
with open('data64/func2.pkl', 'rb') as file:
    func2 = pickle.load(file)

# --- Parameters ---
tkwargs = {"dtype": torch.double, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
BOUNDS = torch.tensor([[-1, -1], [1, 1]], **tkwargs)
REFERENCE = torch.tensor([-1, -1], **tkwargs)
BATCH_SIZE = 10
N_ITER = 10
CSV_FILE = "bo_hyperparameter_sweep_results_sequential.csv"

# Default hyperparameter values
DEFAULT_PARAMS = {
    "raw_samples": 256,
    "num_restarts": 10,
    "batch_limit": 5,
    "max_iter": 100
}

# Default hyperparameter values
HYPERPARAMETER_GRID = {
    "raw_samples": [128, 256, 512, 1024],
    "num_restarts": [5, 10, 20, 50],
    "batch_limit": [1, 5, 10, 20],
    "max_iter": [50, 100, 200, 500]
}

# Hyperparameter values to test (others remain default)
RAW_SAMPLES_VALUES = [64, 128, 256, 512, 1024]
NUM_RESTARTS_VALUES = [5, 10, 20, 50]
BATCH_LIMIT_VALUES = [1, 5, 10, 20]
MAX_ITER_VALUES = [50, 100, 200, 500]

# --- Functions ---
def multi_objective(X):
    return torch.cat([func1.eval(X), func2.eval(X)], dim=1)

def initialize_model(train_X, train_Y):
    models = [SingleTaskGP(train_X, train_Y[..., i:i+1],
                           input_transform=Normalize(train_X.shape[-1]),
                           outcome_transform=Standardize(m=1)) for i in range(train_Y.shape[-1])]
    return SumMarginalLogLikelihood(ModelListGP(*models).likelihood, ModelListGP(*models)), ModelListGP(*models)

def step_mobo(model, train_X, raw_samples, num_restarts, batch_limit, max_iter):
    acq = qLogNoisyExpectedHypervolumeImprovement(model, REFERENCE, train_X, prune_baseline=True)
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=BOUNDS,
        q=BATCH_SIZE,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential = True,
        options={"batch_limit": batch_limit, "maxiter": max_iter},
    )
    return candidates.detach(), multi_objective(candidates), candidates

def run_bo_with_hyperparams(init_X, raw_samples, num_restarts, batch_limit, max_iter):
    init_Y = multi_objective(init_X)
    mll, model = initialize_model(init_X, init_Y)

    hvs, times, candidates_list = [DominatedPartitioning(REFERENCE, init_Y).compute_hypervolume().item()], [0.0], [init_X.cpu().tolist()]
    train_X, train_Y = init_X, init_Y

    for i in range(1, N_ITER + 1):
        try:
            start = time.time()
            fit_gpytorch_mll(mll)
            new_X, new_Y, candidates = step_mobo(model, train_X, raw_samples, num_restarts, batch_limit, max_iter)

            # Clear CUDA cache after acquisition optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_X, train_Y = torch.cat([train_X, new_X]), torch.cat([train_Y, new_Y])
            hv = DominatedPartitioning(REFERENCE, train_Y).compute_hypervolume().item()
            hvs.append(hv)
            times.append(time.time() - start)
            candidates_list.append(candidates.cpu().tolist())

            mll, model = initialize_model(train_X, train_Y)
            print(f"[Iter {i}/{N_ITER}] HV={hv:.5f} | raw={raw_samples}, restarts={num_restarts}, limit={batch_limit}, maxiter={max_iter}")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM at Iter {i} | raw={raw_samples}, restarts={num_restarts}, limit={batch_limit}, maxiter={max_iter}. Skipping...")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    return hvs, times, candidates_list

def save_results(params, hvs, times, candidates_list):
    with open(CSV_FILE, mode="a", newline="") as file:
        csv.writer(file).writerow([
            params['raw_samples'],
            params['num_restarts'],
            params['batch_limit'],
            params['max_iter'],
            json.dumps(hvs),
            json.dumps(times),
            json.dumps(candidates_list)
        ])

def hyperparameter_sweep(init_X):
    param_sets = []

    # Create parameter sets by sweeping each hyperparameter individually while others stay default
    for param, values in HYPERPARAMETER_GRID.items():
        for value in values:
            params = DEFAULT_PARAMS.copy()
            params[param] = value
            param_sets.append(params)

    for params in param_sets:
        print(f"\nTesting params: {params}")
        try:
            hvs, times, candidates_list = run_bo_with_hyperparams(init_X, **params)
            save_results(params, hvs, times, candidates_list)
        except Exception as e:
            print(f"Failed with params {params}: {e}")

# --- Main ---
def main():
    init_X = torch.load('data64/train.pt').to(**tkwargs)

    with open(CSV_FILE, mode="w", newline="") as file:
        csv.writer(file).writerow(["raw_samples", "num_restarts", "batch_limit", "max_iter", "hypervolumes", "times", "candidates"])

    hyperparameter_sweep(init_X)
    print("Sweep complete. Results saved.")

if __name__ == "__main__":
    main()