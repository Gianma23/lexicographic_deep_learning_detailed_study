import math
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


LossesFn = Callable[[], List[float]]


@dataclass
class _ParamUpdate:
    param: torch.nn.Parameter
    base: torch.Tensor
    delta: torch.Tensor


class LTEAdam(Optimizer):
    """Adam optimizer with Lexicographic Tolerance Enforcement (LTE) step filtering.

    This optimizer computes one Adam update direction per outer step (moments are updated
    exactly once), then performs backtracking over a scalar step multiplier to find an
    LTE-admissible parameter update against ordered losses.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps_adam: float = 1e-8,
        weight_decay: float = 0.0,
        lte_eps: Optional[Iterable[float]] = None,
        beta_backtrack: float = 0.5,
        max_backtracks: int = 5,
        reject_if_inadmissible: bool = True,
        fallback_small_step: bool = False,
        fallback_step_scale: Optional[float] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps_adam <= 0.0:
            raise ValueError(f"Invalid eps_adam: {eps_adam}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if len(betas) != 2:
            raise ValueError("betas must be a tuple/list of length 2.")
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if not (0.0 < beta_backtrack < 1.0):
            raise ValueError(f"beta_backtrack must be in (0, 1), got {beta_backtrack}.")
        if int(max_backtracks) < 0:
            raise ValueError(f"max_backtracks must be >= 0, got {max_backtracks}.")

        parsed_lte_eps: List[float] = []
        if lte_eps is not None:
            for idx, value in enumerate(lte_eps):
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError(f"lte_eps[{idx}] must be finite, got {parsed}.")
                if parsed < 0.0:
                    raise ValueError(f"lte_eps[{idx}] must be >= 0, got {parsed}.")
                parsed_lte_eps.append(parsed)

        parsed_fallback_step_scale: Optional[float] = None
        if fallback_step_scale is not None:
            parsed_fallback_step_scale = float(fallback_step_scale)
            if not math.isfinite(parsed_fallback_step_scale) or parsed_fallback_step_scale <= 0.0:
                raise ValueError(
                    "fallback_step_scale must be a finite float > 0 when provided, "
                    f"got {fallback_step_scale}."
                )

        defaults = dict(
            lr=float(lr),
            betas=(beta1, beta2),
            eps_adam=float(eps_adam),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)

        self.lte_eps: Tuple[float, ...] = tuple(parsed_lte_eps)
        self.beta_backtrack: float = float(beta_backtrack)
        self.max_backtracks: int = int(max_backtracks)
        self.reject_if_inadmissible: bool = bool(reject_if_inadmissible)
        self.fallback_small_step: bool = bool(fallback_small_step)
        self.fallback_step_scale: Optional[float] = parsed_fallback_step_scale

        self._validate_group_hparams()

    def _validate_group_hparams(self) -> None:
        for group_idx, group in enumerate(self.param_groups):
            lr = float(group.get("lr", 0.0))
            betas = group.get("betas", (0.9, 0.999))
            eps_adam = float(group.get("eps_adam", 1e-8))
            weight_decay = float(group.get("weight_decay", 0.0))
            if lr < 0.0:
                raise ValueError(f"Invalid learning rate in param group {group_idx}: {lr}")
            if len(betas) != 2:
                raise ValueError(f"Param group {group_idx}: betas must have length 2.")
            beta1, beta2 = float(betas[0]), float(betas[1])
            if not (0.0 <= beta1 < 1.0):
                raise ValueError(f"Param group {group_idx}: invalid beta1 {beta1}.")
            if not (0.0 <= beta2 < 1.0):
                raise ValueError(f"Param group {group_idx}: invalid beta2 {beta2}.")
            if eps_adam <= 0.0:
                raise ValueError(f"Param group {group_idx}: eps_adam must be > 0, got {eps_adam}.")
            if weight_decay < 0.0:
                raise ValueError(
                    f"Param group {group_idx}: weight_decay must be >= 0, got {weight_decay}."
                )

    @staticmethod
    def _to_float(value, name: str) -> float:
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"{name} must be scalar, got tensor with shape {tuple(value.shape)}.")
            return float(value.detach().item())
        return float(value)

    def _evaluate_losses(self, losses_fn: LossesFn) -> List[float]:
        raw_losses = losses_fn()
        if not isinstance(raw_losses, (list, tuple)):
            raise TypeError(
                "losses_fn() must return a list/tuple of ordered scalar losses, "
                f"got {type(raw_losses).__name__}."
            )
        losses: List[float] = []
        for idx, value in enumerate(raw_losses):
            parsed = self._to_float(value, name=f"losses_fn[{idx}]")
            losses.append(parsed)
        if not losses:
            raise ValueError("losses_fn() returned an empty list; expected at least one loss.")
        return losses

    def _validate_lte_shape(self, losses_len: int) -> None:
        expected = max(losses_len - 1, 0)
        if len(self.lte_eps) != expected:
            raise ValueError(
                f"Invalid LTE tolerance shape: losses_fn returned L={losses_len}, so "
                f"lte_eps must have length L-1={expected}, got {len(self.lte_eps)}."
            )

    @staticmethod
    def _all_finite(losses: List[float]) -> bool:
        return all(math.isfinite(value) for value in losses)

    def _is_lte_admissible(
        self,
        old_losses: List[float],
        new_losses: List[float],
    ) -> Tuple[bool, str]:
        if len(new_losses) != len(old_losses):
            return (
                False,
                f"loss_count_mismatch_old_{len(old_losses)}_new_{len(new_losses)}",
            )
        if not self._all_finite(new_losses):
            return False, "non_finite_candidate_losses"

        for level in range(len(old_losses) - 1):
            allowed = old_losses[level] + float(self.lte_eps[level])
            if new_losses[level] > allowed:
                return False, f"lte_violation_level_{level}"
        return True, ""

    def _build_param_updates(self) -> List[_ParamUpdate]:
        updates: List[_ParamUpdate] = []
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = float(group["betas"][0]), float(group["betas"][1])
            eps_adam = float(group["eps_adam"])
            weight_decay = float(group["weight_decay"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "LTEAdam does not support sparse gradients, matching Adam behavior."
                    )

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = int(state["step"])

                grad_data = grad.detach()
                if weight_decay != 0.0:
                    grad_data = grad_data.add(param.detach(), alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad_data, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_data, grad_data, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps_adam)
                delta = (-step_size) * (exp_avg / denom)
                updates.append(
                    _ParamUpdate(
                        param=param,
                        base=param.detach().clone(memory_format=torch.preserve_format),
                        delta=delta.detach(),
                    )
                )
        return updates

    def _snapshot_state_for_grads(self) -> List[Tuple[torch.nn.Parameter, Dict[str, Any]]]:
        snapshots: List[Tuple[torch.nn.Parameter, Dict[str, Any]]] = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                state_snapshot: Dict[str, Any] = {}
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state_snapshot[key] = value.detach().clone(memory_format=torch.preserve_format)
                    else:
                        state_snapshot[key] = copy.deepcopy(value)
                snapshots.append((param, state_snapshot))
        return snapshots

    def _restore_state_snapshot(
        self,
        snapshots: List[Tuple[torch.nn.Parameter, Dict[str, Any]]],
    ) -> None:
        for param, snapshot in snapshots:
            state = self.state[param]
            state.clear()
            for key, value in snapshot.items():
                if torch.is_tensor(value):
                    state[key] = value.detach().clone(memory_format=torch.preserve_format)
                else:
                    state[key] = copy.deepcopy(value)

    @staticmethod
    def _restore_params(updates: List[_ParamUpdate]) -> None:
        for update in updates:
            update.param.copy_(update.base)

    @staticmethod
    def _apply_scaled_step(updates: List[_ParamUpdate], scale: float) -> None:
        for update in updates:
            update.param.copy_(update.base + (scale * update.delta))

    @torch.no_grad()
    def step(
        self,
        closure=None,
        losses_fn: Optional[LossesFn] = None,
        train_loss=None,
    ) -> Dict[str, object]:
        if losses_fn is None:
            raise ValueError("LTEAdam.step requires losses_fn to evaluate LTE admissibility.")

        old_losses = self._evaluate_losses(losses_fn)
        self._validate_lte_shape(len(old_losses))

        closure_loss = None
        if closure is not None:
            with torch.enable_grad():
                closure_loss = closure()

        parsed_train_loss: Optional[float] = None
        if train_loss is not None:
            parsed_train_loss = self._to_float(train_loss, name="train_loss")
        elif closure_loss is not None:
            parsed_train_loss = self._to_float(closure_loss, name="closure_loss")

        state_snapshot = self._snapshot_state_for_grads()
        updates = self._build_param_updates()
        if not updates:
            return {
                "accepted": True,
                "num_backtracks": 0,
                "step_scale": 0.0,
                "old_losses": list(old_losses),
                "new_losses": list(old_losses),
                "train_loss": parsed_train_loss,
                "inadmissible_reason": "no_gradients",
            }

        accepted = False
        accepted_scale = 0.0
        accepted_backtracks = 0
        final_losses = list(old_losses)
        last_reason = "no_admissible_step"

        for backtrack_idx in range(self.max_backtracks + 1):
            step_scale = self.beta_backtrack ** backtrack_idx
            self._apply_scaled_step(updates, step_scale)
            candidate_losses = self._evaluate_losses(losses_fn)
            is_admissible, reason = self._is_lte_admissible(old_losses, candidate_losses)
            if is_admissible:
                accepted = True
                accepted_scale = float(step_scale)
                accepted_backtracks = int(backtrack_idx)
                final_losses = list(candidate_losses)
                last_reason = ""
                break
            self._restore_params(updates)
            last_reason = reason or "no_admissible_step"

        if not accepted:
            self._restore_params(updates)
            if self.fallback_small_step:
                fallback_scale = self.fallback_step_scale
                if fallback_scale is None:
                    fallback_scale = self.beta_backtrack ** (self.max_backtracks + 1)
                self._apply_scaled_step(updates, fallback_scale)
                final_losses = self._evaluate_losses(losses_fn)
                accepted_scale = float(fallback_scale)
                accepted_backtracks = int(self.max_backtracks + 1)
                last_reason = f"fallback_step_applied_after_{last_reason}"
            elif self.reject_if_inadmissible:
                final_losses = list(old_losses)
                accepted_scale = 0.0
                accepted_backtracks = int(self.max_backtracks)
                last_reason = last_reason or "rejected_no_admissible_step"
                self._restore_state_snapshot(state_snapshot)
            else:
                # Conservative default when explicit rejection is disabled but no fallback is enabled:
                # restore old parameters rather than silently applying an inadmissible candidate.
                final_losses = list(old_losses)
                accepted_scale = 0.0
                accepted_backtracks = int(self.max_backtracks)
                last_reason = f"restored_no_fallback_after_{last_reason}"
                self._restore_state_snapshot(state_snapshot)

        return {
            "accepted": bool(accepted),
            "num_backtracks": int(accepted_backtracks),
            "step_scale": float(accepted_scale),
            "old_losses": list(old_losses),
            "new_losses": list(final_losses),
            "train_loss": parsed_train_loss,
            "inadmissible_reason": str(last_reason),
        }


class LTEAdamW(LTEAdam):
    """LTE step filtering over candidate steps generated by torch.optim.AdamW."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps_adam: float = 1e-8,
        weight_decay: float = 0.0,
        lte_eps: Optional[Iterable[float]] = None,
        beta_backtrack: float = 0.5,
        max_backtracks: int = 5,
        reject_if_inadmissible: bool = True,
        fallback_small_step: bool = False,
        fallback_step_scale: Optional[float] = None,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps_adam=eps_adam,
            weight_decay=weight_decay,
            lte_eps=lte_eps,
            beta_backtrack=beta_backtrack,
            max_backtracks=max_backtracks,
            reject_if_inadmissible=reject_if_inadmissible,
            fallback_small_step=fallback_small_step,
            fallback_step_scale=fallback_step_scale,
        )

        # Use PyTorch AdamW internals for moment/state updates and base candidate step.
        self._adamw = torch.optim.AdamW(
            self.param_groups,
            lr=float(lr),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps_adam),
            weight_decay=float(weight_decay),
        )
        # Keep one shared source of truth for state/param groups.
        self.param_groups = self._adamw.param_groups
        self.state = self._adamw.state

    @torch.no_grad()
    def _build_param_updates(self) -> List[_ParamUpdate]:
        updates: List[_ParamUpdate] = []
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "LTEAdamW does not support sparse gradients, matching AdamW behavior."
                    )
                updates.append(
                    _ParamUpdate(
                        param=param,
                        base=param.detach().clone(memory_format=torch.preserve_format),
                        delta=torch.zeros_like(param, memory_format=torch.preserve_format),
                    )
                )

        if not updates:
            return updates

        self._adamw.step()

        for update in updates:
            update.delta = (
                update.param.detach() - update.base
            ).clone(memory_format=torch.preserve_format)

        self._restore_params(updates)
        return updates

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        # Re-sync wrapped AdamW references after checkpoint restore.
        self._adamw.param_groups = self.param_groups
        self._adamw.state = self.state
