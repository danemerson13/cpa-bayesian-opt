import torch
import os
import time

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.model import ModelList

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from botorch.optim.optimize import optimize_acqf_discrete_local_search
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# General Parameters
REFERENCE = torch.tensor([-1,-1], **tkwargs)
BATCH_SIZE = 10
MAX_CONC = 6
STEP = 0.5

# Acquition Function Optimization Parameters
NUM_RESTARTS = 10
RAW_SAMPLES = 2048
MAX_BATCH_SIZE = 2048
MAX_TRIES = 100

def load_data(acq_name):# Set path of dir copntaining data
    data_path = 'data'
    # Load the starting data (same for all acq functions)
    for file in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, file)):
            data = torch.load(os.path.join(data_path, file)).to(**tkwargs)

    # Now load the acq function specific data (from all iterations)
    for file in os.listdir(os.path.join(data_path, acq_name)):
        iter_data = torch.load(os.path.join(data_path, acq_name, file)).to(**tkwargs)
        # Append each iteration to the data tensor
        data = torch.cat((data, iter_data), dim=0)

    # Return the data tensor
    train_X = data[:,:-3]
    train_Y = data[:,-3:-1]
    if acq_name == "VarLogNEHVI":
        train_Yvar = data[:,-1].unsqueeze(-1)
    else:
        train_Yvar = None

    return train_X, train_Y, train_Yvar

def init_model(train_X, train_Y, train_Yvar = None):
    # Initialize the model with the training data
    # Account for variance in observations when train_Yvar is not None
    deterministic_model = GenericDeterministicModel(lambda x: x.sum(dim=-1, keepdim=True))
    # Define GP model (on viability)
    gp_model = SingleTaskGP(
        train_X = train_X,
        train_Y = train_Y[:,1].unsqueeze(-1),
        train_Yvar = train_Yvar,
        input_transform = Normalize(train_X.shape[-1]),
        outcome_transform = Standardize(m=1)
    )
    model = ModelList(deterministic_model, gp_model)
    mll = ExactMarginalLogLikelihood(model.models[1].likelihood, model.models[1])
    return mll, model

def get_acqf(acq_name, model, train_X):
    if acq_name == "Random":
        acq = None
    elif acq_name == "LogNEHVI" or acq_name == "VarLogNEHVI":
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model = model,
            ref_point = REFERENCE,
            X_baseline = train_X,
            prune_baseline = True,
            cache_root = False
        )
    elif acq_name == "LogNParEGO":
        acq = qLogNParEGO(
            model = model, 
            X_baseline = train_X, 
            prune_baseline = True, 
            cache_root = False
        )
    else:
        raise ValueError(f"Unknown acquisition function: {acq_name}")
    return acq

def random_sample(train_X):
    samples = []
    while len(samples) < BATCH_SIZE:
        valid_sample = False
        while not valid_sample:
            sample = torch.randint(0, int(1/STEP)*MAX_CONC, (1, train_X.shape[-1]), **tkwargs)*STEP
            if (sample.sum() <= MAX_CONC) and (torch.norm(sample - train_X, dim=1).min() > 0):
                valid_sample = True
        samples.append(sample)
    return torch.cat(samples)

def step(acq_name, model, train_X):
    if acq_name == "Random":
        candidates = random_sample(train_X)
    else:
        acq = get_acqf(acq_name, model, train_X)
        candidates, _ = optimize_acqf_discrete_local_search(
            acq_function = acq,
            q = BATCH_SIZE,
            discrete_choices = [torch.arange(0, MAX_CONC+STEP, STEP, **tkwargs)]*train_X.shape[-1],
            inequality_constraints = [
                (
                    torch.arange(train_X.shape[-1], dtype = torch.int).to(tkwargs['device']),
                    torch.full((train_X.shape[-1],), -1.0, **tkwargs),
                    -MAX_CONC
                )
            ],
            num_restarts = NUM_RESTARTS,
            raw_samples = RAW_SAMPLES,
            max_batch_size = MAX_BATCH_SIZE,
            max_tries = MAX_TRIES,
            unique = True
        )
    return candidates.detach()

def run(acq_name):
    # Load the data for the specific acquisition function
    train_X, train_Y, train_Yvar = load_data(acq_name)
    
    # Create the model
    mll, model = init_model(train_X, train_Y, train_Yvar)
    
    # Fit the GP model
    fit_gpytorch_mll(mll)
    
    # Generate candidates
    start = time.time()
    candidates = step(acq_name, model, train_X)
    end = time.time()
    print(f"Candidate generation for: {acq_name}, Time: {end - start:.2f} seconds")
    
    # Clear CUDA cache after acquisition optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Use the deterministic model and trained GP to get the total concentration and predicted viability
    with torch.no_grad():
        pred_Y = model.posterior(candidates).mean

    # Return the candidates and Y_pred together
    return torch.cat((candidates, pred_Y), dim=-1).cpu().numpy()

