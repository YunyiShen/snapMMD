"""Campaign 9: SnapMMD with the mRNA-only parametric repressilator on the mRNA-protein data (one seed per call).
Mirrors fit/snapmmd/classic/classic_sde.py (Repressilator) except for the data file."""
import importlib.util, os, sys, time
import numpy as np, torch
from snapMMD.dls import MMDLoss, snapMMD, RBF
HERE = os.path.dirname(os.path.abspath(__file__)); A = os.path.abspath(os.path.join(HERE, "..", ".."))   # aistats2027/
spec = importlib.util.spec_from_file_location("classic_models", f"{A}/code/models/classic.py")
models = importlib.util.module_from_spec(spec); spec.loader.exec_module(models)
seed = int(sys.argv[1]); torch.set_num_threads(2)
data = np.load(f"{A}/../data/missingobs/Repressilator_data.npz")
N = int(data["N_steps"]); Xs = [torch.tensor(data["Xs"][i]) for i in range(N - 1)]
dts = torch.tensor(data["dts"]); ts = torch.tensor(data["time_scale"])
y0 = Xs[0].repeat([5, 1])
model = models.repressilator(10., 1., 1., 10., .03)
os.makedirs(f"{HERE}/models", exist_ok=True)
torch.manual_seed(seed)
dls = snapMMD(model, Xs, dts[:-1] / ts, lr=0.05)
mmd = MMDLoss(kernel=RBF())
t0 = time.time(); dls.train(mmd, y0, epochs=800, adaptive_bandwidth=False)
torch.save(model.state_dict(), f"{HERE}/models/Repressilator_model_{seed}.pt")
print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
