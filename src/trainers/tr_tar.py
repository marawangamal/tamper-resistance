"""Sharpness-aware minimization (SAM) applied to unlearning objectives."""

import torch
from torch.func import functional_call

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.utils import BalancedSampler, dict_dot

INP_FIELDS = ("input_ids", "attention_mask", "labels")


class TARBaseTrainer(AxolotlTrainer):
    """Apply a small SAM perturbation to gradient-difference unlearning."""

    def __init__(
        self,
        *args,
        source_id_pre,
        source_id_post,
        source_id_att,
        source_id_ref,
        attack_kwargs,
        coeff_post=1.0,
        coeff_ref=1.0,
        loss_pre="ce",
        loss_post="me",
        loss_ref="ce",
        **kwargs,
    ):
        self.source_id_pre = source_id_pre
        self.source_id_post = source_id_post
        self.source_id_att = source_id_att
        self.source_id_ref = source_id_ref
        self.attack_kwargs = attack_kwargs
        self.coeff_post = coeff_post
        self.coeff_ref = coeff_ref
        self.loss_pre = loss_pre
        self.loss_post = loss_post
        self.loss_ref = loss_ref

        self.loss_fns = {
            "ce": lambda x: self.model(x).loss
            # "me": lambda x: ... compute max entropy loss here.
        }

    @property
    def attack_fns(self):
        raise NotImplementedError  # must subclass this to get specific method, i.e., TAR, SAM, Etc.

    def _get_train_sampler(self, train_dataset=None):
        """Draw equally from the configured forget and retain sources."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_batch_size < 2 or self._train_batch_size % 2:
            raise ValueError("SAM sampling requires an even batch size")

        source_ids = dataset["source_id"]
        sample_ids_forget = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_forget
        ]
        sample_ids_retain = [
            i
            for i, source_id in enumerate(source_ids)
            if source_id == self.source_id_retain
        ]
        if not sample_ids_forget or not sample_ids_retain:
            raise ValueError("SAM sampling requires both forget and retain examples")
        return BalancedSampler(sample_ids_forget, sample_ids_retain)

    def compute_loss_pre(self, x):
        return self.loss_fns[self.loss_pre]

    def compute_loss_ref(self, x):
        return self.loss_fns[self.loss_ref]

    def compute_loss_post(self, x):
        # here we will need to make a surrogate loss that will lead to the correct gradient.
        pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        mask_pre = inputs["source_id"] == self.source_id_pre
        mask_post = inputs["source_id"] == self.source_id_post
        mask_att = inputs["source_id"] == self.source_id_att
        mask_ref = inputs["source_id"] == self.source_id_ref

        x_pre = {field: inputs[field][mask_pre] for field in INP_FIELDS}
        x_post = {field: inputs[field][mask_post] for field in INP_FIELDS}
        x_att = {field: inputs[field][mask_att] for field in INP_FIELDS}
        x_ref = {field: inputs[field][mask_ref] for field in INP_FIELDS}

        theta = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        loss_pre = self.compute_loss_pre(x_pre)
        loss_post = 0
        for attack_fn in self.attack_fns:
            loss_att_traj, delta = attack_fn(x_att, **self.attack_kwargs)
            theta_attacked = {
                name: param + delta.detach()
                for (name, param), delta in zip(theta.items(), delta)
            }
            loss_post += self.compute_loss_post(x_post, theta_attacked)

        loss_ref = self.compute_loss_ref(x_ref)

        # Compute a detached perturbation from the forget gradient.
        # loss_forget = self.compute_loss_forget(model, x_forget)
        # grad_forget = torch.autograd.grad(loss_forget, tuple(theta.values()))
        # grad_dict = dict(zip(theta, grad_forget))
        # grad_norm = dict_dot(grad_dict, grad_dict).sqrt() + 1e-6
        # theta_perturbed = {
        #     name: param + (self.rho * grad / grad_norm.to(param.device)).detach()
        #     for (name, param), grad in zip(theta.items(), grad_forget)
        # }
        # theta_perturbed = self.get_perturbed_theta(inputs)

        # # Evaluate at perturbed weights while keeping gradients connected to theta.
        # loss_forget = self.compute_loss_forget(
        #     lambda **x: functional_call(model, theta_perturbed, (), x), x_forget
        # )

        # # The model's original weights remain unchanged.
        # loss_retain = self.compute_loss_retain(model, x_retain)

        self.log(
            {
                "loss_pre": loss_pre.detach().item(),
                "loss_post": loss_post.detach().item(),
                "loss_ref": loss_ref.detach().item(),
            }
        )

        loss = loss_pre + self.coeff_post * loss_post + self.coeff_ref * loss_ref
        return (loss, None) if return_outputs else loss
