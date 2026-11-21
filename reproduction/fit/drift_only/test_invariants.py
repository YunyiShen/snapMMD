"""Small numerical checks for the controlled comparison; no scientific refits."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torchsde

import train
from snapMMD.dls import MMDLoss, RBF, snapMMD


class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_models_and_frozen_diffusion(self):
        for task, dim, scale in [("LV", 2, 9.), ("ReprParam", 3, 10.)]:
            x = torch.tensor([[-1., 0., 2.], [1., 3., 4.]])[:, :dim]
            a = train.make_model(task, "structured", scale, 1)
            b = train.make_model(task, "mlp", scale, 1)
            torch.testing.assert_close(a.g(0., x), b.g(0., x), rtol=0, atol=0)
            torch.testing.assert_close(a.g(0., x) / np.sqrt(scale), 0.02 * x.relu())
            torch.testing.assert_close(a.f(0., x), a.drift.f(0., x), rtol=0, atol=0)
            for model in [a, b]:
                self.assertFalse(model.g(0., x).requires_grad)
                self.assertFalse(any("sigma" in name and p.requires_grad for name, p in model.named_parameters()))
                before = model.g(0., x).clone()
                opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0025)
                model.f(0., x).square().mean().backward()
                opt.step()
                torch.testing.assert_close(before, model.g(0., x), rtol=0, atol=0)

    def test_original_mlp_is_preserved(self):
        # Extract only the old MLP class, avoiding the historical CLI's imports/prints.
        path = train.REPO / "reproduction/fit/fully_neural/MLP.py"
        source = ast.parse(path.read_text())
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "MLP")
        scope = {"nn": nn}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        for task, dim, scale in [("LV", 2, 9.), ("ReprParam", 3, 10.)]:
            torch.manual_seed(1)
            old = scope["MLP"](dim)
            new = train.make_model(task, "mlp", scale, 1)
            x = torch.arange(4 * dim, dtype=torch.float32).reshape(4, dim)
            torch.testing.assert_close(old(x), new.f(0., x), rtol=0, atol=0)
            other = train.make_model(task, "mlp", scale, 2)
            self.assertFalse(torch.equal(new.f(0., x), other.f(0., x)))

    def test_target_is_excluded(self):
        source = train.REPO / "data/classic/LV_data.npz"
        with np.load(source) as z:
            data = {k: z[k].copy() for k in z.files}
        original, times, scale = train.load_training(source)
        data["Xs"][-1] = 1e6
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "changed.npz"
            np.savez(path, **data)
            changed, other_times, other_scale = train.load_training(path)
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
        torch.testing.assert_close(times, other_times, rtol=0, atol=0)
        self.assertEqual(scale, other_scale)
        self.assertEqual(len(times), 10)
        self.assertAlmostEqual(float(times[-1]), 1.)

    def test_original_optimizer_step_is_reproduced(self):
        # Before freezing diffusion, reproduce the released fitter on the same path.
        torch.manual_seed(3)
        xs = torch.rand(3, 6, 2) + 1
        times = torch.tensor([0., .002, .004])
        y0 = xs[0].repeat(2, 1)
        old = train.LotkaVolterra(4.5, .9, .9, .18, .03)
        new = copy.deepcopy(old)
        old_fitter = snapMMD(old, list(xs), times, bm=train.make_brownian(y0, times, 7), lr=.0025)
        old_fitter.train(MMDLoss(RBF()), y0, epochs=1, gradclip=1.)
        criterion, weights = train.training_objective(xs)
        optimizer = torch.optim.Adam(new.parameters(), lr=.0025)
        loss = train.objective(new, y0, times, xs, criterion, weights,
                               train.make_brownian(y0, times, 7), .001)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new.parameters(), 1.)
        optimizer.step()
        for a, b in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_checkpoint_resume_and_brownian_order(self):
        for arm in ["structured", "mlp"]:
            xs = torch.ones(3, 6, 2)
            xs[1] *= 1.1
            xs[2] *= 1.2
            times = torch.tensor([0., .002, .004])
            y0 = xs[0].repeat(2, 1)
            model = train.make_model("LV", arm, 9., 1)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.Adam(params, lr=.0025)
            criterion, weights = train.training_objective(xs)
            bm = train.make_brownian(y0, times, 71)

            def step(m, o, brownian):
                o.zero_grad()
                loss = train.objective(m, y0, times, xs, criterion, weights, brownian, .001)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.)
                o.step()

            step(model, opt, bm)
            with tempfile.TemporaryDirectory() as d:
                file = Path(d) / "checkpoint.pt"
                train.save_checkpoint(file, model, opt, 1, {"test": True})
                state = torch.load(file)
            resumed = train.make_model("LV", arm, 9., 1)
            resumed.load_state_dict(state["model"])
            opt2 = torch.optim.Adam([p for p in resumed.parameters() if p.requires_grad], lr=.0025)
            opt2.load_state_dict(state["optimizer"])
            step(model, opt, bm)
            bm2 = train.make_brownian(y0, times, 71)
            step(resumed, opt2, bm2)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
            bm3 = train.make_brownian(y0, times, 71)
            bm3(.001, .003)
            torch.testing.assert_close(bm(.001, .002), bm3(.001, .002), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
