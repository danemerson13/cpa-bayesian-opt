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
warnings.filterwarnings("ignore", category = RuntimeWarning)

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
N_ITER = 20  # Updated number of iterations
CSV_FILE = "bo_replicability_results_sequential.csv"
N_RUNS = 10  # Number of repeated runs

# Default hyperparameters
DEFAULT_PARAMS = {
    "raw_samples": 256,
    "num_restarts": 10,
    "batch_limit": 5,
    "max_iter": 100
}

# --- Functions ---
def multi_objective(X):
    return torch.cat([func1.eval(X), func2.eval(X)], dim=1)

def initialize_model(train_X, train_Y):
    models = [
        SingleTaskGP(
            train_X, train_Y[..., i:i+1],
            input_transform=Normalize(train_X.shape[-1]),
            outcome_transform=Standardize(m=1)
        ) for i in range(train_Y.shape[-1])
    ]
    model_list = ModelListGP(*models)
    return SumMarginalLogLikelihood(model_list.likelihood, model_list), model_list

def step_mobo(model, train_X, raw_samples, num_restarts, batch_limit, max_iter):
    acq = qLogNoisyExpectedHypervolumeImprovement(model, REFERENCE, train_X, prune_baseline=True)
    candidates, _ = optimize_acqf(
        acq_function=acq,   
        bounds=BOUNDS,
        q=BATCH_SIZE,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": batch_limit, "maxiter": max_iter},
        sequential = True
    )
    return candidates.detach(), multi_objective(candidates), candidates

def run_bo(init_X, run_id):
    init_Y = multi_objective(init_X)
    mll, model = initialize_model(init_X, init_Y)

    hvs = [DominatedPartitioning(REFERENCE, init_Y).compute_hypervolume().item()]
    times = [0.0]
    candidates_list = [init_X.cpu().tolist()]
    train_X, train_Y = init_X.clone(), init_Y.clone()

    for i in range(1, N_ITER + 1):
        try:
            start = time.time()
            fit_gpytorch_mll(mll)
            new_X, new_Y, candidates = step_mobo(model, train_X, **DEFAULT_PARAMS)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])

            hv = DominatedPartitioning(REFERENCE, train_Y).compute_hypervolume().item()
            hvs.append(hv)
            times.append(time.time() - start)
            candidates_list.append(candidates.cpu().tolist())

            mll, model = initialize_model(train_X, train_Y)

            print(f"[Run {run_id} | Iter {i}/{N_ITER}] HV={hv:.5f} t={times[-1]:.2f}")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[Run {run_id} | Iter {i}] CUDA OOM encountered. Skipping to next run.")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    return hvs, times, candidates_list

def save_results(run_id, hvs, times, candidates_list):
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([run_id, json.dumps(hvs), json.dumps(times), json.dumps(candidates_list)])

# --- Main ---
def main():
    init_X = torch.load('data64/train.pt').to(**tkwargs)

    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id", "hypervolumes", "times", "candidates"])

    for run_id in range(1, N_RUNS + 1):
        print(f"\n--- Starting Run {run_id}/{N_RUNS} ---")
        try:
            hvs, times, candidates_list = run_bo(init_X, run_id)
            save_results(run_id, hvs, times, candidates_list)
        except Exception as e:
            print(f"Run {run_id} failed: {e}")

    print("\nAll runs complete. Results saved.")

if __name__ == "__main__":
    main()