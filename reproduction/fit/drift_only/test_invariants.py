"""Small numerical checks for the controlled comparison; no scientific refits."""
import ast
import copy
from contextlib import nullcontext
import json
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

    def test_misspecified_formulas_and_units(self):
        x = torch.tensor([[.3, 1.2], [2., .7]], dtype=torch.float64)
        model = train.make_model("LV", "lv_saturated", 9., 1, "learned_state", 1.5).double()
        alpha, beta, gamma, delta = [float(getattr(model.drift, k).exp())
                                      for k in ["alpha", "beta", "gamma", "delta"]]
        z = x.numpy() + 1e-10
        predation = z[:, 0] * z[:, 1] / (1 + z[:, 0] / 1.5)
        expected = np.column_stack([alpha*z[:, 0]-beta*predation,
                                    delta*predation-gamma*z[:, 1]])
        np.testing.assert_allclose(model.f(0., x).detach(), expected, rtol=1e-14)
        self.assertNotIn("drift.saturation_scale", dict(model.named_parameters()))
        self.assertIn("drift.saturation_scale", model.state_dict())
        # At x=K the interaction is half its bilinear counterpart.
        k = model.drift.saturation_scale
        self.assertEqual(float(k * 2. / (1. + k / k)), float(k))
        # Disabling saturation in test-only algebra must recover the classic drift.
        classic = train.make_model("LV", "structured", 9., 1, "learned_state").double()
        uncapped = np.column_stack([alpha*z[:, 0]-beta*z[:, 0]*z[:, 1],
                                    delta*z[:, 0]*z[:, 1]-gamma*z[:, 1]])
        np.testing.assert_allclose(classic.f(0., x).detach(), uncapped, rtol=1e-14)
        for task, arm, dim, scale in [("LV", "lv_saturated", 2, 9.),
                                       ("ReprParam", "repr_exponential", 3, 10.)]:
            kwargs = {"saturation_scale": 1.5} if dim == 2 else {}
            a = train.make_model(task, arm, scale, 1, "learned_state", **kwargs).double()
            b = train.make_model(task, arm, 2*scale, 1, "learned_state", **kwargs).double()
            values = torch.tensor([[.3, 1.2, 2.], [2., .7, .4]], dtype=torch.float64)[:, :dim]
            torch.testing.assert_close(a.f(0., values)/scale, b.f(0., values)/(2*scale), rtol=2e-7, atol=1e-8)
        x3 = torch.tensor([[.3, 1.2, 2.], [-1., 0., .5]], dtype=torch.float64)
        z3 = x3.numpy().clip(0) + 1e-8
        exp_model = train.make_model("ReprParam", "repr_exponential", 10., 1, "learned_state").double()
        beta, k, gamma = [float(getattr(exp_model.drift, key).exp()) for key in ["beta", "k", "gamma"]]
        expected = np.empty_like(z3)
        for i, parent in enumerate([2, 0, 1]):
            expected[:, i] = beta*np.exp(-z3[:, parent]/k)-gamma*z3[:, i]
        np.testing.assert_allclose(exp_model.f(0., x3).detach(), expected, rtol=1e-14)
        for task, dim, scale in [("LV", 2, 9.), ("ReprParam", 3, 10.)]:
            affine = train.make_model(task, "affine", scale, 1, "learned_state").double()
            values = x3[:, :dim]
            expected = values.numpy() @ affine.drift.weight.detach().numpy().T + affine.drift.bias.detach().numpy()
            np.testing.assert_allclose(affine.f(0., values).detach(), expected, rtol=1e-14)
            self.assertEqual(sum(p.numel() for p in affine.parameters()), dim*(dim+1)+1)

    def test_misspecified_gpu_launcher(self):
        launcher = Path(train.__file__).with_name("rusty_misspecified_gpu.sbatch")
        env = dict(os.environ, SNAPMMD_REPO=str(train.REPO), SNAPMMD_DRY_RUN="1", SLURM_ARRAY_JOB_ID="123")
        for key in ["SNAPMMD_STAGE", "SNAPMMD_EPOCHS", "SNAPMMD_EXPECT_COMMIT"]:
            env.pop(key, None)
        expected = [("LV", "affine"), ("ReprParam", "affine"), ("LV", "lv_saturated"), ("ReprParam", "repr_exponential")]
        outputs = []
        for index, pair in enumerate(expected):
            result = subprocess.run(["bash", str(launcher)], env=dict(env, SLURM_ARRAY_TASK_ID=str(index)),
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            command = shlex.split(result.stdout)
            options = dict(zip(command[2::2], command[3::2]))
            self.assertEqual((options["--task"], options["--drift"]), pair)
            for key, value in [("--epochs", "1000"), ("--lr", "0.01"), ("--gradclip", "1.0"),
                               ("--diffusion", "learned_state"), ("--seed", "1"), ("--device", "cuda"),
                               ("--threads", "1"), ("--schedule", "constant")]:
                self.assertEqual(options[key], value)
            outputs.append(options["--output"])
        self.assertEqual(len(set(outputs)), 4)
        for changes in [{"SLURM_ARRAY_TASK_ID": "4"}, {"SLURM_ARRAY_TASK_ID": "-1"},
                        {"SLURM_ARRAY_TASK_ID": "0", "SNAPMMD_EPOCHS": "8"},
                        {"SLURM_ARRAY_TASK_ID": "0", "SNAPMMD_EPOCHS": "8", "SNAPMMD_STAGE": "misspecified-full-gpu/"},
                        {"SLURM_ARRAY_TASK_ID": "0", "SNAPMMD_STAGE": "../other"},
                        {"SLURM_ARRAY_TASK_ID": "0", "SNAPMMD_EPOCHS": "0", "SNAPMMD_STAGE": "smoke"}]:
            result = subprocess.run(["bash", str(launcher)], env=dict(env, **changes),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode, 0)
        smoke = subprocess.run(["bash", str(launcher)],
                               env=dict(env, SLURM_ARRAY_TASK_ID="0", SNAPMMD_EPOCHS="8", SNAPMMD_STAGE="smoke"),
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        self.assertIn("--epochs 8", smoke.stdout)
        mismatch = subprocess.run(["bash", str(launcher)],
                                  env=dict(env, SLURM_ARRAY_TASK_ID="0", SNAPMMD_DRY_RUN="0", SNAPMMD_EXPECT_COMMIT="0"*40),
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("Checkout changed", mismatch.stderr)

    def test_training_scale_and_new_family_guards(self):
        source = train.REPO / "data/classic/LV_data.npz"
        xs, _, _ = train.load_training(source)
        k = train.training_saturation_scale(xs)
        self.assertEqual(k, float(np.median(xs[..., 0].numpy())))
        with np.load(source) as z:
            data = {key: z[key].copy() for key in z.files}
        data["Xs"][-1] = -1e12
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.npz"
            np.savez(changed, **data)
            other, _, _ = train.load_training(changed)
        self.assertEqual(k, train.training_saturation_scale(other))
        self.assertEqual(train.training_lv_followup_scales(xs), train.training_lv_followup_scales(other))
        for bad in [torch.zeros_like(xs), torch.full_like(xs, float("nan"))]:
            with self.assertRaises(ValueError):
                train.training_saturation_scale(bad)
        for task, arm, diffusion, kval in [("LV", "repr_exponential", "learned_state", None),
                                           ("ReprParam", "lv_saturated", "learned_state", k),
                                           ("LV", "affine", "fixed_true", None),
                                           ("LV", "lv_saturated", "learned_state", None)]:
            with self.assertRaises(ValueError):
                train.make_model(task, arm, 9., 1, diffusion, kval)

    def test_new_family_gradients_and_checkpoint_resume(self):
        for task, arm, dim, scale in [("LV", "affine", 2, 9.), ("ReprParam", "affine", 3, 10.),
                                       ("LV", "lv_saturated", 2, 9.), ("ReprParam", "repr_exponential", 3, 10.),
                                       ("LV", "lv_saturated_calibrated", 2, 9.), ("LV", "lv_logistic_saturated", 2, 9.)]:
            kwargs = {"saturation_scale": 1.5} if arm in train.LV_SATURATED else {}
            if arm in train.LV_FOLLOWUP:
                kwargs["reference_prey"] = 1.3
            if arm == "lv_logistic_saturated":
                kwargs["carrying_capacity"] = 10.
            torch.manual_seed(41)
            xs = torch.rand(3, 6, dim) + .4
            times = torch.tensor([0., .002, .004])
            y0 = xs[0].repeat(2, 1)
            criterion, weights = train.training_objective(xs)
            model = train.make_model(task, arm, scale, 1, "learned_state", **kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)

            def step(m, opt):
                opt.zero_grad()
                loss = train.objective(m, y0, times, xs, criterion, weights,
                                       train.make_brownian(y0, times, 17), .001)
                loss.backward()
                for name, parameter in m.named_parameters():
                    self.assertIsNotNone(parameter.grad, (arm, name))
                    self.assertTrue(torch.isfinite(parameter.grad).all(), (arm, name))
                    self.assertGreater(float(parameter.grad.abs().sum()), 0., (arm, name))
                train.gradient_norm(list(m.parameters()), 1.)
                opt.step()

            step(model, optimizer)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                train.save_checkpoint(path, model, optimizer, 1, {"family": arm})
                state = torch.load(path)
            restored = train.make_model(task, arm, scale, 2, "learned_state", **kwargs)
            restored.load_state_dict(state["model"])
            opt2 = torch.optim.Adam(restored.parameters(), lr=.01)
            opt2.load_state_dict(state["optimizer"])
            step(model, optimizer)
            step(restored, opt2)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
            if arm in train.LV_SATURATED:
                self.assertEqual(float(model.drift.saturation_scale), 1.5)

    def test_calibrated_lv_initialization_and_formulas(self):
        xs, _, scale = train.load_training(train.REPO / "data/classic/LV_data.npz")
        k, ref, capacity = train.training_lv_followup_scales(xs)
        self.assertEqual(k, float(np.median(xs[..., 0].numpy())))
        self.assertEqual(ref, float(np.median(xs[0, :, 0].numpy())))
        self.assertEqual(capacity, 5 * float(xs[..., 0].max()))
        classic = train.make_model("LV", "structured", scale, 1, "learned_state").double()
        at_ref = torch.tensor([[ref, 1.], [ref, 4.]], dtype=torch.float64)
        for arm in sorted(train.LV_FOLLOWUP):
            c = capacity if arm == "lv_logistic_saturated" else None
            kwargs = dict(saturation_scale=k, reference_prey=ref, carrying_capacity=c)
            model = train.make_model("LV", arm, scale, 1, "learned_state", **kwargs).double()
            self.assertEqual(sum(p.numel() for p in model.parameters()), 5)
            self.assertEqual(set(dict(model.drift.named_parameters())), {"alpha", "beta", "gamma", "delta"})
            torch.testing.assert_close(model.f(0., at_ref), classic.f(0., at_ref), rtol=5e-7, atol=5e-7)
            torch.testing.assert_close(model.g(0., at_ref), classic.g(0., at_ref), rtol=0, atol=0)
            values = torch.tensor([[.3, 1.2], [2., .7], [-1., 0.]], dtype=torch.float64)
            z = values.numpy().clip(0) + 1e-10
            a, b, g, d = [float(getattr(model.drift, name).exp()) for name in ("alpha", "beta", "gamma", "delta")]
            qy = z[:, 0] * z[:, 1] / (1 + z[:, 0]/k)
            growth = a * z[:, 0] * (1 if c is None else 1-z[:, 0]/float(model.drift.carrying_capacity))
            expected = np.column_stack([growth-b*qy, d*qy-g*z[:, 1]])
            np.testing.assert_allclose(model.f(0., values).detach(), expected, rtol=1e-14)
            other = train.make_model("LV", arm, 2*scale, 1, "learned_state", **kwargs).double()
            torch.testing.assert_close(model.f(0., values)/scale, other.f(0., values)/(2*scale), rtol=3e-7, atol=1e-8)
            if c is not None:
                high_prey = torch.tensor([[2*c, 1.]], dtype=torch.float64)
                self.assertLess(float(model.f(0., high_prey)[0, 0]), 0.)
        for arm, ref_value, c in [("lv_saturated_calibrated", None, None),
                                  ("lv_logistic_saturated", ref, None),
                                  ("lv_saturated_calibrated", ref, capacity),
                                  ("lv_saturated", ref, None),
                                  ("lv_logistic_saturated", ref, ref),
                                  ("lv_logistic_saturated", ref, float("inf"))]:
            with self.assertRaises(ValueError):
                train.make_model("LV", arm, scale, 1, "learned_state", k, ref_value, c)

    def test_saturated_interaction_reordering(self):
        old = train.SaturatingLV(3.).double()
        new = train.SaturatingLV(3., reference_prey=2.).double()
        with torch.no_grad():
            for name, value in old.named_parameters():
                getattr(new, name).copy_(value)
        values = torch.tensor([[.3, 1.2], [2., .7]], dtype=torch.float64, requires_grad=True)
        outputs, gradients = [], []
        for model in (old, new):
            result = model(values)
            outputs.append(result)
            gradients.append(torch.autograd.grad(result.sum(), (values,) + tuple(model.parameters())))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-14, atol=1e-14)
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=1e-14, atol=1e-14)
        large = torch.tensor([[1e20, 1e20]], dtype=torch.float32)
        self.assertFalse(torch.isfinite(old.float()(large)).all())
        self.assertTrue(torch.isfinite(new.float()(large)).all())

    def test_failure_snapshot_preserves_last_checkpoint(self):
        model = train.make_model("LV", "lv_saturated_calibrated", 9., 1, "learned_state", 3., 2.)
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        states = torch.tensor([[[1., 2.], [3., 4.]],
                               [[float("nan"), float("inf")], [1e20, -2.]]])
        context = dict(phase="training_loss", attempted_epoch=428, last_successful_epoch=427,
                       optimizer_step_started=False, optimizer_step_completed=False,
                       trajectory_model_epoch=427, trajectory_time_units="solver",
                       _trajectory=states, _trajectory_times=torch.tensor([0., 1.]))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            train.save_checkpoint(out/"checkpoint.pt", model, opt, 425, {})
            previous = (out/"checkpoint.pt").read_bytes()
            with torch.no_grad():
                model.drift.alpha.add_(.1)
            train.save_failure(out, model, opt, {}, None, context, FloatingPointError("bad loss"))
            self.assertEqual(previous, (out/"checkpoint.pt").read_bytes())
            record = json.loads(next(out.glob("failure_*.json")).read_text())
            self.assertEqual(record["diagnostic_errors"], [])
            self.assertEqual(record["trajectory_ranges"][1]["nonfinite_entries"], 2)
            self.assertEqual(record["trajectory_ranges"][1]["finite_coordinate_min"], [float(states[1, 1, 0]), -2.])
            state = torch.load(out/record["failure_checkpoint"])
            self.assertTrue(state["diagnostic_only"])
            self.assertNotIn("epoch", state)
            self.assertEqual(state["failure_context"]["last_successful_epoch"], 427)
            torch.testing.assert_close(state["model"]["drift.alpha"], model.drift.alpha)
            with np.load(out/record["trajectory_artifact"]) as z:
                np.testing.assert_equal(z["states"], states.numpy())

    def test_training_failure_phases_and_artifacts(self):
        xs = torch.ones(2, 3, 2)
        times = torch.tensor([0., .001])
        context = {}
        with patch.object(train.torchsde, "sdeint_adjoint", return_value=xs*float("inf")):
            with self.assertRaises(FloatingPointError):
                train.objective(None, xs[0], times, xs, None, None, None, .001, context)
        self.assertEqual(context["phase"], "training_trajectory")
        with tempfile.TemporaryDirectory() as directory:
            for phase in ("training_loss", "gradient_norm", "parameter_check"):
                out = Path(directory)/phase
                argv = ["train.py", "--task", "LV", "--drift", "lv_saturated_calibrated", "--diffusion",
                        "learned_state", "--seed", "1", "--epochs", "1", "--lr", ".01", "--output", str(out)]
                with patch("sys.argv", argv):
                    args = train.parse_args()
                def fake_objective(model, y0, ts, observed, criterion, weights, bm, dt, diagnostics):
                    train.begin_trajectory(diagnostics, "training_loss", ts)
                    diagnostics["_trajectory"] = observed.detach()
                    return torch.tensor(float("nan")) if phase == "training_loss" else model.sigma.square()
                def bad_update(optimizer):
                    with torch.no_grad():
                        optimizer.param_groups[0]["params"][0].fill_(float("nan"))
                injection = (patch.object(train, "gradient_norm", side_effect=RuntimeError("injected gradient failure"))
                             if phase == "gradient_norm" else
                             patch.object(torch.optim.Adam, "step", new=bad_update)
                             if phase == "parameter_check" else nullcontext())
                with patch.object(train, "objective", side_effect=fake_objective), injection, \
                        patch.object(torch, "set_num_interop_threads"), self.assertRaises((FloatingPointError, RuntimeError)):
                    train.run(args)
                record = json.loads(next(out.glob("failure_*.json")).read_text())
                self.assertEqual((record["phase"], record["attempted_epoch"], record["last_successful_epoch"]), (phase, 1, 0))
                self.assertEqual(record["optimizer_step_started"], phase == "parameter_check")
                self.assertEqual(record["optimizer_step_completed"], phase == "parameter_check")
                self.assertTrue((out/"checkpoint_initial.pt").exists())
                saved = torch.load(out/"checkpoint.pt")
                self.assertEqual(saved["epoch"], 0)
                self.assertTrue(all(torch.isfinite(v).all() for v in saved["model"].values()))
                failed = torch.load(out/record["failure_checkpoint"])
                self.assertEqual(bool(torch.isfinite(failed["model"]["sigma"])), phase != "parameter_check")
                self.assertTrue((out/record["trajectory_artifact"]).exists())
                self.assertFalse((out/"summary.json").exists())

    def test_metric_failure_records_the_matching_cloud(self):
        path = train.REPO/"data/classic/LV_data.npz"
        xs, _, scale = train.load_training(path)
        context = {"last_successful_epoch": 1000}
        def fake_solve(model, initial, times, **kwargs):
            return torch.stack([initial*(1+float(t)) for t in times])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(train.torchsde, "sdeint", side_effect=fake_solve), \
                    patch.object(train, "mmd2", return_value={"snap": float("nan")}), \
                    patch.object(train, "emd", return_value=0.), self.assertRaises(FloatingPointError):
                train.evaluate(None, path, path.with_name("LV_data_interp_val.npz"), xs, scale, 1, .001, out, context)
            self.assertEqual(context["phase"], "evaluation_forecast_first_metrics")
            with np.load(out/"predictions.partial.npz") as z:
                np.testing.assert_equal(context["_trajectory"][0].numpy(), z["forecast_first"])
            self.assertFalse((out/"summary.json").exists())

    def test_lv_followup_launcher(self):
        launcher = Path(train.__file__).with_name("rusty_lv_saturation_followup_gpu.sbatch")
        self.assertIn("#SBATCH --array=0-1%2", launcher.read_text())
        env = dict(os.environ, SNAPMMD_REPO=str(train.REPO), SNAPMMD_DRY_RUN="1", SLURM_ARRAY_JOB_ID="123")
        for key in ("SNAPMMD_STAGE", "SNAPMMD_EPOCHS", "SNAPMMD_EXPECT_COMMIT"):
            env.pop(key, None)
        outputs = []
        for index, arm in enumerate(("lv_saturated_calibrated", "lv_logistic_saturated")):
            result = subprocess.run(["bash", str(launcher)], env=dict(env, SLURM_ARRAY_TASK_ID=str(index)),
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            command = shlex.split(result.stdout)
            options = dict(zip(command[2::2], command[3::2]))
            for key, value in [("--task", "LV"), ("--drift", arm), ("--lr", "0.01"), ("--epochs", "1000"),
                               ("--seed", "1"), ("--diffusion", "learned_state"), ("--gradclip", "1.0"),
                               ("--schedule", "constant"), ("--threads", "1"), ("--device", "cuda")]:
                self.assertEqual(options[key], value)
            self.assertIn("/lv-saturation-followup-full-gpu/123/", options["--output"])
            outputs.append(options["--output"])
            with patch("sys.argv", ["train.py"] + command[2:]):
                self.assertEqual(train.parse_args().drift, arm)
        self.assertEqual(len(set(outputs)), 2)
        for changes in [{"SLURM_ARRAY_TASK_ID": "2"}, {"SLURM_ARRAY_TASK_ID": "-1"},
                        {"SLURM_ARRAY_TASK_ID": "0", "SNAPMMD_EPOCHS": "8"},
                        {"SLURM_ARRAY_TASK_ID": "0", "SNAPMMD_STAGE": "../other"},
                        {"SLURM_ARRAY_TASK_ID": "0", "SNAPMMD_EPOCHS": "0"}]:
            result = subprocess.run(["bash", str(launcher)], env=dict(env, **changes), capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        smoke = subprocess.run(["bash", str(launcher)],
                               env=dict(env, SLURM_ARRAY_TASK_ID="0", SNAPMMD_EPOCHS="8", SNAPMMD_STAGE="lv-saturation-smoke"),
                               text=True, capture_output=True, check=True)
        self.assertIn("--epochs 8", smoke.stdout)
        mismatch = subprocess.run(["bash", str(launcher)],
                                  env=dict(env, SLURM_ARRAY_TASK_ID="0", SNAPMMD_DRY_RUN="0", SNAPMMD_EXPECT_COMMIT="0"*40),
                                  text=True, capture_output=True)
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("Checkout changed", mismatch.stderr)


if __name__ == "__main__":
    unittest.main()
