"""Small numerical checks for the controlled comparison; no scientific refits."""
import ast
import copy
import math
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

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

    def test_learned_diffusion_matches_paper(self):
        for task, dim, scale in [("LV", 2, 9.), ("ReprParam", 3, 10.)]:
            original = (train.LotkaVolterra(4.5, .9, .9, .18, .03) if task == "LV"
                        else train.repressilator(10., 1., 1., 10., .03))
            structured = train.make_model(task, "structured", scale, 1, "learned_state")
            mlp = train.make_model(task, "mlp", scale, 1, "learned_state")
            x = torch.tensor([[-1., 0., 2.], [1., 3., 4.]])[:, :dim]
            torch.testing.assert_close(original.f(0., x), structured.f(0., x), rtol=0, atol=0)
            for model in [structured, mlp]:
                torch.testing.assert_close(original.g(0., x), model.g(0., x), rtol=0, atol=0)
                self.assertEqual(model.sigma.numel(), 1)
                self.assertTrue(model.sigma.requires_grad)
                self.assertAlmostEqual(train.sigma_solver(model) / math.sqrt(scale), .03 / math.sqrt(scale), places=8)
                model.g(0., x).sum().backward()
                self.assertGreater(float(model.sigma.grad), 0.)
            torch.manual_seed(3)
            xs = torch.rand(3, 6, dim) + 1
            times = torch.tensor([0., .002, .004])
            y0 = xs[0].repeat(2, 1)
            criterion, weights = train.training_objective(xs)
            mlp.zero_grad()
            loss = train.objective(mlp, y0, times, xs, criterion, weights,
                                   train.make_brownian(y0, times, 17), .001)
            loss.backward()
            self.assertTrue(torch.isfinite(mlp.sigma.grad))
            self.assertNotEqual(float(mlp.sigma.grad), 0.)
            before = mlp.sigma.detach().clone()
            torch.optim.Adam(mlp.parameters(), lr=.0025).step()
            self.assertFalse(torch.equal(before, mlp.sigma))
            self.assertGreater(train.sigma_solver(mlp), 0.)

    def test_paper_control_update_unclipped(self):
        torch.manual_seed(5)
        xs = torch.rand(3, 6, 3) + 1
        times = torch.tensor([0., .002, .004])
        y0 = xs[0].repeat(2, 1)
        original = train.repressilator(10., 1., 1., 10., .03)
        control = train.make_model("ReprParam", "structured", 10., 1, "learned_state")
        fitter = snapMMD(original, list(xs), times, bm=train.make_brownian(y0, times, 17), lr=.05)
        fitter.train(MMDLoss(RBF()), y0, epochs=1)
        criterion, weights = train.training_objective(xs)
        optimizer = torch.optim.Adam(control.parameters(), lr=.05)
        loss = train.objective(control, y0, times, xs, criterion, weights,
                               train.make_brownian(y0, times, 17), .001)
        loss.backward()
        gradients = [p.grad.clone() for p in control.parameters()]
        train.gradient_norm(list(control.parameters()), None)
        for expected, p in zip(gradients, control.parameters()):
            torch.testing.assert_close(expected, p.grad, rtol=0, atol=0)
        optimizer.step()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, control.state_dict()[name], rtol=0, atol=0)

    def test_cosine_checkpoint_resume(self):
        torch.manual_seed(5)
        xs = torch.rand(3, 6, 2) + 1
        times = torch.tensor([0., .002, .004])
        y0 = xs[0].repeat(2, 1)
        criterion, weights = train.training_objective(xs)
        model = train.make_model("LV", "mlp", 9., 1, "learned_state")
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = train.make_scheduler(optimizer, "cosine", 4, .0001)

        def step(m, opt, sched):
            lr = opt.param_groups[0]["lr"]
            opt.zero_grad()
            loss = train.objective(m, y0, times, xs, criterion, weights,
                                   train.make_brownian(y0, times, 17), .001)
            loss.backward()
            train.gradient_norm(list(m.parameters()), 1.)
            opt.step()
            sched.step()
            return lr

        self.assertEqual(step(model, optimizer, scheduler), .01)
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d) / "checkpoint.pt"
            train.save_checkpoint(checkpoint, model, optimizer, 1, {}, scheduler)
            state = torch.load(checkpoint)
        resumed = train.make_model("LV", "mlp", 9., 1, "learned_state")
        resumed.load_state_dict(state["model"])
        optimizer2 = torch.optim.Adam(resumed.parameters(), lr=.01)
        scheduler2 = train.make_scheduler(optimizer2, "cosine", 4, .0001)
        optimizer2.load_state_dict(state["optimizer"])
        scheduler2.load_state_dict(state["scheduler"])
        for _ in range(3):
            last_used = step(model, optimizer, scheduler)
            self.assertEqual(last_used, step(resumed, optimizer2, scheduler2))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], .0001, places=15)
        self.assertAlmostEqual(last_used, .0001 + .5 * (.01 - .0001) * (1 + math.cos(3 * math.pi / 4)), places=15)
        self.assertEqual(scheduler.state_dict(), scheduler2.state_dict())

    def test_failed_evaluation_retains_returned_cloud(self):
        path = train.REPO / "data/classic/LV_data.npz"
        interp = path.with_name("LV_data_interp_val.npz")
        xs, _, scale = train.load_training(path)
        model = train.make_model("LV", "mlp", scale, 1, "learned_state")
        failed = torch.stack([xs[0], torch.full_like(xs[0], float("nan"))])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(train.torchsde, "sdeint", return_value=failed), self.assertRaises(FloatingPointError):
                train.evaluate(model, path, interp, xs, scale, 1, .001, out)
            self.assertFalse((out / "predictions.npz").exists())
            with np.load(out / "predictions.partial.npz") as z:
                self.assertTrue(np.isnan(z["forecast_first"]).all())
                self.assertEqual(z["forecast_first"].shape, (200, 2))

    def test_learned_launcher_mapping(self):
        launcher = Path(__file__).with_name("rusty_learned_gpu.sbatch")
        env = dict(os.environ, SNAPMMD_DRY_RUN="1", SNAPMMD_REPO=str(train.REPO),
                   SNAPMMD_PYTHON="python", SNAPMMD_RUNS="/tmp/test-runs", SLURM_ARRAY_JOB_ID="123")
        for key in ("SNAPMMD_EPOCHS", "SNAPMMD_STAGE"):
            env.pop(key, None)
        outputs = []
        for index in range(9):
            result = subprocess.run(["bash", str(launcher)], env=dict(env, SLURM_ARRAY_TASK_ID=str(index)),
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            command = shlex.split(result.stdout)
            options = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(options["--diffusion"], "learned_state")
            self.assertEqual(options["--seed"], "1")
            self.assertEqual(options["--device"], "cuda")
            outputs.append(options["--output"])
            if index == 0:
                self.assertEqual((options["--task"], options["--drift"], options["--lr"], options["--epochs"], options["--gradclip"]),
                                 ("ReprParam", "structured", "0.05", "800", "none"))
            else:
                self.assertEqual(options["--task"], "LV" if index % 2 else "ReprParam")
                self.assertEqual(options["--drift"], "mlp")
                self.assertEqual(options["--epochs"], "1000")
                self.assertEqual(float(options["--lr"]), [.0025, .001, .01, .01][(index - 1) // 2])
                self.assertEqual(options["--schedule"], "cosine" if index >= 7 else "constant")
                self.assertEqual(options["--gradclip"], "1.0")
        self.assertEqual(len(set(outputs)), 9)
        bad = subprocess.run(["bash", str(launcher)], env=dict(env, SLURM_ARRAY_TASK_ID="9"),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(bad.returncode, 0)

    def test_cli_schedule_guards(self):
        base = ["train.py", "--task", "LV", "--drift", "mlp", "--seed", "1", "--output", "/tmp/not-created"]
        with patch("sys.argv", base + ["--gradclip", "none", "--schedule", "cosine", "--min-lr", ".0001"]):
            args = train.parse_args()
            self.assertIsNone(args.gradclip)
        for extra in (["--schedule", "cosine", "--min-lr", ".01"], ["--lr", "nan"], ["--gradclip", "0"]):
            with patch("sys.argv", base + extra), self.assertRaises(SystemExit):
                train.parse_args()


if __name__ == "__main__":
    unittest.main()
