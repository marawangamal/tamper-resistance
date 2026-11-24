"""Gradient-difference loss terms for unlearning."""

import torch

from axolotl.core.trainers.base import AxolotlTrainer
from trainers.utils import dict_dot, model_to_params, BalancedSampler

INP_FIELDS = ("input_ids", "attention_mask", "labels")


class SEAMAttackBaseTrainer(AxolotlTrainer):
    def __init__(
        self,
        *args,
        source_id_cosim_a,
        source_id_cosim_b,
        eps_finite_diff=1e-3,
        eps_norm_div=1e-8,
        **kwargs,
    ):
        if source_id_cosim_a == source_id_cosim_b:
            raise ValueError("Cosine tracking requires two distinct sources")
        super().__init__(*args, **kwargs)
        self.source_id_cosim_a = source_id_cosim_a
        self.source_id_cosim_b = source_id_cosim_b
        self.eps_norm_div = eps_norm_div
        self.eps_finite_diff = eps_finite_diff

    def _get_train_sampler(self, train_dataset=None):
        """Draw equally from the configured forget and retain sources."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_batch_size < 2 or self._train_batch_size % 2:
            raise ValueError("Cosine sampling requires an even batch size")

        source_ids = dataset["source_id"]
        source_id_cosim_a = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_cosim_a
        ]
        source_id_cosim_b = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_cosim_b
        ]
        if not source_id_cosim_a or not source_id_cosim_b:
            raise ValueError("Cosim sampling requires both forget and retain examples")
        return BalancedSampler(source_id_cosim_a, source_id_cosim_b)

    def compute_loss_avg_of_cosims(self, model, inputs, compute_loss_only=False):
        """Average normalized per-example CE gradient dot products across sources."""
        if not compute_loss_only:
            raise NotImplementedError("Cross-source cosine supports tracking only")
        B = len(inputs["labels"])
        if B < 2:
            raise ValueError("Cross-source cosine requires at least two examples")
        theta = model_to_params(model, only_trainable=True)

        # compute m (i.e., the mean of normalized gradients)
        source_ids = inputs["source_id"]
        mask_a = source_ids == self.source_id_cosim_a
        mask_b = source_ids == self.source_id_cosim_b
        n_a = mask_a.sum().item()
        n_b = mask_b.sum().item()

        if n_a == 0 or n_b == 0:
            raise ValueError("Cosine tracking requires both sources")

        # Keep cosine statistics in FP32 for low-precision model gradients.
        dtype = next(iter(theta.values())).dtype
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        m_a = {name: torch.zeros_like(p, dtype=dtype) for name, p in theta.items()}
        m_b = {name: torch.zeros_like(p, dtype=dtype) for name, p in theta.items()}

        for i in range(B):
            if mask_a[i]:
                m, n = m_a, n_a
            elif mask_b[i]:
                m, n = m_b, n_b
            else:
                continue
            batch = {name: inputs[name][i : i + 1] for name in INP_FIELDS}
            g_theta_i = dict(
                zip(
                    theta,
                    torch.autograd.grad(model(**batch).loss, tuple(theta.values())),
                )
            )
            norm_i = dict_dot(g_theta_i, g_theta_i, dtype=dtype).sqrt()
            denom_i = norm_i + self.eps_norm_div
            for name, p in theta.items():
                m[name] += g_theta_i[name].to(dtype) / (n * denom_i.to(p.device))

        return dict_dot(m_a, m_b)

    def compute_loss_cosim_of_batch_grads(
        self, model, inputs, compute_loss_only=False, return_layerwise=False
    ):
        """Cosine between the batch CE gradients of two sources."""
        if not compute_loss_only:
            raise NotImplementedError("Cross-source cosine supports tracking only")
        B = len(inputs["labels"])
        if B < 2:
            raise ValueError("Cross-source cosine requires at least two examples")
        theta = model_to_params(model, only_trainable=True)

        source_ids = inputs["source_id"]
        mask_a = source_ids == self.source_id_cosim_a
        mask_b = source_ids == self.source_id_cosim_b
        n_a = mask_a.sum().item()
        n_b = mask_b.sum().item()

        if n_a == 0 or n_b == 0:
            raise ValueError("Cosine tracking requires both sources")

        # Keep cosine statistics in FP32 for low-precision model gradients.
        dtype = next(iter(theta.values())).dtype
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        batch_a = {name: inputs[name][mask_a] for name in INP_FIELDS}
        batch_b = {name: inputs[name][mask_b] for name in INP_FIELDS}
        grad_a = dict(
            zip(
                theta,
                torch.autograd.grad(model(**batch_a).loss, tuple(theta.values())),
            )
        )
        grad_b = dict(
            zip(
                theta,
                torch.autograd.grad(model(**batch_b).loss, tuple(theta.values())),
            )
        )

        norm_a = dict_dot(grad_a, grad_a, dtype=dtype).sqrt()
        norm_b = dict_dot(grad_b, grad_b, dtype=dtype).sqrt()
        cosim = dict_dot(grad_a, grad_b, dtype=dtype) / (
            norm_a * norm_b + self.eps_norm_div
        )
        if not return_layerwise:
            return cosim

        cosims_layerwise = self._compute_loss_cosim_of_batch_grads_layerwise(
            grad_a, grad_b, norm_a, norm_b, dtype
        )
        return cosim, cosims_layerwise

    def _compute_loss_cosim_of_batch_grads_layerwise(
        self, grad_a, grad_b, norm_a, norm_b, dtype, suffix=""
    ):
        global_denom = norm_a * norm_b + self.eps_norm_div
        layer_stats = {}
        for name in grad_a:
            parts = name.split(".")
            if "layers" in parts:
                layer = f"layer_{int(parts[parts.index('layers') + 1]):02d}"
            elif "embed_tokens" in parts:
                layer = "embed_tokens"
            elif "lm_head" in parts:
                layer = "lm_head"
            elif "norm" in parts:
                layer = "final_norm"
            else:
                layer = parts[0]

            a = grad_a[name].to(dtype)
            b = grad_b[name].to(dtype)
            values = [(a * b).sum(), a.square().sum(), b.square().sum()]
            if layer not in layer_stats:
                layer_stats[layer] = values
            else:
                layer_stats[layer] = [
                    total + value.to(total.device)
                    for total, value in zip(layer_stats[layer], values)
                ]

        cosims_layerwise = {}
        for layer, (dot, norm_a_sq, norm_b_sq) in layer_stats.items():
            cosims_layerwise[f"cosim_of_batch_grads{suffix}/{layer}"] = dot / (
                norm_a_sq.sqrt() * norm_b_sq.sqrt() + self.eps_norm_div
            )
            cosims_layerwise[f"cosim_contribution{suffix}/{layer}"] = (
                dot / global_denom.to(dot.device)
            )
        return cosims_layerwise

    def compute_loss_cosim_of_batch_grads_signed(
        self, model, inputs, compute_loss_only=False, return_layerwise=False
    ):
        """Cosine between the signed forget batch gradient and retain batch gradient."""
        if not compute_loss_only:
            raise NotImplementedError("Cross-source cosine supports tracking only")
        B = len(inputs["labels"])
        if B < 2:
            raise ValueError("Cross-source cosine requires at least two examples")
        theta = model_to_params(model, only_trainable=True)

        source_ids = inputs["source_id"]
        mask_a = source_ids == self.source_id_cosim_a
        mask_b = source_ids == self.source_id_cosim_b
        n_a = mask_a.sum().item()
        n_b = mask_b.sum().item()

        if n_a == 0 or n_b == 0:
            raise ValueError("Cosine tracking requires both sources")

        dtype = next(iter(theta.values())).dtype
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        batch_a = {name: inputs[name][mask_a] for name in INP_FIELDS}
        batch_b = {name: inputs[name][mask_b] for name in INP_FIELDS}
        grad_a = dict(
            zip(
                theta,
                torch.autograd.grad(model(**batch_a).loss, tuple(theta.values())),
            )
        )
        grad_b = dict(
            zip(
                theta,
                torch.autograd.grad(model(**batch_b).loss, tuple(theta.values())),
            )
        )
        grad_a = {name: value.to(dtype).sign() for name, value in grad_a.items()}

        norm_a = dict_dot(grad_a, grad_a).sqrt()
        norm_b = dict_dot(grad_b, grad_b, dtype=dtype).sqrt()
        cosim = dict_dot(grad_a, grad_b, dtype=dtype) / (
            norm_a * norm_b + self.eps_norm_div
        )
        if not return_layerwise:
            return cosim

        cosims_layerwise = self._compute_loss_cosim_of_batch_grads_layerwise(
            grad_a, grad_b, norm_a, norm_b, dtype, suffix="_signed"
        )
        return cosim, cosims_layerwise

    def compute_loss_cosim_of_avgs(self, model, inputs, compute_loss_only=False):
        """Cosine between the mean per-example CE gradients of two sources."""
        if not compute_loss_only:
            raise NotImplementedError("Cross-source cosine supports tracking only")
        B = len(inputs["labels"])
        if B < 2:
            raise ValueError("Cross-source cosine requires at least two examples")
        theta = model_to_params(model, only_trainable=True)

        source_ids = inputs["source_id"]
        mask_a = source_ids == self.source_id_cosim_a
        mask_b = source_ids == self.source_id_cosim_b
        n_a = mask_a.sum().item()
        n_b = mask_b.sum().item()

        if n_a == 0 or n_b == 0:
            raise ValueError("Cosine tracking requires both sources")

        dtype = next(iter(theta.values())).dtype
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        m_a = {name: torch.zeros_like(p, dtype=dtype) for name, p in theta.items()}
        m_b = {name: torch.zeros_like(p, dtype=dtype) for name, p in theta.items()}

        for i in range(B):
            if mask_a[i]:
                m, n = m_a, n_a
            elif mask_b[i]:
                m, n = m_b, n_b
            else:
                continue
            batch = {name: inputs[name][i : i + 1] for name in INP_FIELDS}
            g_theta_i = dict(
                zip(
                    theta,
                    torch.autograd.grad(model(**batch).loss, tuple(theta.values())),
                )
            )
            for name in theta:
                m[name] += g_theta_i[name].to(dtype) / n

        norm_a = dict_dot(m_a, m_a).sqrt()
        norm_b = dict_dot(m_b, m_b).sqrt()
        return dict_dot(m_a, m_b) / (norm_a * norm_b + self.eps_norm_div)

    def compute_loss_cosim_of_avgs_signed(self, model, inputs, compute_loss_only=False):
        """Cosine between the signed mean forget gradient and mean retain gradient."""
        if not compute_loss_only:
            raise NotImplementedError("Cross-source cosine supports tracking only")
        B = len(inputs["labels"])
        if B < 2:
            raise ValueError("Cross-source cosine requires at least two examples")
        theta = model_to_params(model, only_trainable=True)

        # Compute the mean raw gradient for each source.
        source_ids = inputs["source_id"]
        mask_a = source_ids == self.source_id_cosim_a
        mask_b = source_ids == self.source_id_cosim_b
        n_a = mask_a.sum().item()
        n_b = mask_b.sum().item()

        if n_a == 0 or n_b == 0:
            raise ValueError("Cosine tracking requires both sources")

        # Keep cosine statistics in FP32 for low-precision model gradients.
        dtype = next(iter(theta.values())).dtype
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        m_a = {name: torch.zeros_like(p, dtype=dtype) for name, p in theta.items()}
        m_b = {name: torch.zeros_like(p, dtype=dtype) for name, p in theta.items()}

        for i in range(B):
            if mask_a[i]:
                m, n = m_a, n_a
            elif mask_b[i]:
                m, n = m_b, n_b
            else:
                continue
            batch = {name: inputs[name][i : i + 1] for name in INP_FIELDS}
            g_theta_i = dict(
                zip(
                    theta,
                    torch.autograd.grad(model(**batch).loss, tuple(theta.values())),
                )
            )
            for name in theta:
                m[name] += g_theta_i[name].to(dtype) / n

        m_a = {name: value.sign() for name, value in m_a.items()}
        norm_a = dict_dot(m_a, m_a).sqrt()
        norm_b = dict_dot(m_b, m_b).sqrt()
        return dict_dot(m_a, m_b) / (norm_a * norm_b + self.eps_norm_div)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        batch = {field: inputs[field] for field in INP_FIELDS}
        if model.training and len(batch["labels"]) < 2:
            raise ValueError("GMACrossBaseTrackTrainer requires at least two samples")

        result = super().compute_loss(
            model, batch, return_outputs=return_outputs, **kwargs
        )
        avg_of_cosims = self.compute_loss_avg_of_cosims(
            model, inputs, compute_loss_only=True
        )
        self.log({"avg_of_cosines": avg_of_cosims.item()})

        return result


class SEAMAttackTrainer(SEAMAttackBaseTrainer):
    def __init__(
        self,
        *args,
        source_id_forget,
        source_id_retain,
        coeff_retain=1.0,
        coeff_forget=1.0,
        coeff_gma=1.0,
        eps_finite_diff=1e-3,
        eps_norm_div=1e-8,
        **kwargs,
    ):
        super().__init__(
            *args,
            source_id_cosim_a=source_id_forget,
            source_id_cosim_b=source_id_retain,
            eps_finite_diff=eps_finite_diff,
            eps_norm_div=eps_norm_div,
            **kwargs,
        )
        self.source_id_forget = source_id_forget
        self.source_id_retain = source_id_retain
        self.coeff_retain = coeff_retain
        self.coeff_forget = coeff_forget
        self.coeff_gma = coeff_gma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        forget_mask = inputs["source_id"] == self.source_id_forget
        retain_mask = inputs["source_id"] == self.source_id_retain

        x_forget = {field: inputs[field][forget_mask] for field in INP_FIELDS}
        x_retain = {field: inputs[field][retain_mask] for field in INP_FIELDS}

        # Build the surrogate before retaining the ordinary loss graphs.
        avg_of_cosims = self.compute_loss_avg_of_cosims(
            model, inputs, compute_loss_only=True
        )
        cosim_of_batch_grads, layerwise_cosims = self.compute_loss_cosim_of_batch_grads(
            model, inputs, compute_loss_only=True, return_layerwise=True
        )
        (
            cosim_of_batch_grads_signed,
            layerwise_cosims_signed,
        ) = self.compute_loss_cosim_of_batch_grads_signed(
            model, inputs, compute_loss_only=True, return_layerwise=True
        )
        # cosim_of_avgs = self.compute_loss_cosim_of_avgs(
        #     model, inputs, compute_loss_only=True
        # )
        # cosim_of_avgs_signed = self.compute_loss_cosim_of_avgs_signed(
        #     model, inputs, compute_loss_only=True
        # )
        loss_forget = model(**x_forget).loss
        loss_retain = model(**x_retain).loss

        self.log(
            {
                "loss_retain": loss_retain.detach().item(),
                "loss_forget": loss_forget.detach().item(),
                "avg_of_cosims": avg_of_cosims.detach().item(),
                "cosim_of_batch_grads": cosim_of_batch_grads.detach().item(),
                "cosim_of_batch_grads_signed": cosim_of_batch_grads_signed.detach().item(),
                **{
                    name: value.detach().item()
                    for name, value in layerwise_cosims.items()
                },
                **{
                    name: value.detach().item()
                    for name, value in layerwise_cosims_signed.items()
                },
                # "cosim_of_avgs": cosim_of_avgs.detach().item(),
                # "cosim_of_avgs_signed": cosim_of_avgs_signed.detach().item(),
            }
        )

        loss = self.coeff_forget * loss_forget
        return (loss, None) if return_outputs else loss
