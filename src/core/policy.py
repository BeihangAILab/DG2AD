from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .configuration import cfg_get, configure_huggingface_cache
from .contracts import CandidateResult, PipelineSample, PolicyUpdateStats
from .structure import DirectedOperatorGraph, END, START


@dataclass
class _Beam:
    nodes: list[str]
    token_ids: list[int]
    score: float
    finished: bool = False


class PipelinePolicy:
    """LoRA causal-LM policy with graph-constrained action-token generation."""

    def __init__(self, config, graph: DirectedOperatorGraph, project_root: str):
        cache_root = configure_huggingface_cache(
            project_root,
            cfg_get(config, "policy.cache_dir", ".model_cache/huggingface"),
        )

        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

        self.torch = torch
        self.config = config
        self.model_name = str(cfg_get(config, "policy.model_name", "Qwen/Qwen2.5-0.5B-Instruct"))
        self.revision = str(cfg_get(config, "policy.revision", "main"))
        requested_device = str(cfg_get(config, "policy.device", "auto"))
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("policy.device=cuda but CUDA is unavailable")
            memory_fraction = float(cfg_get(config, "policy.cuda_memory_fraction", 0.70))
            if not 0.0 < memory_fraction <= 1.0:
                raise ValueError("policy.cuda_memory_fraction must lie in (0, 1]")
            device_index = (
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            )
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device_index)

        if self.device.type == "cpu":
            cpu_threads = max(1, int(cfg_get(config, "policy.cpu_threads", 4)))
            cpu_interop_threads = max(1, int(cfg_get(config, "policy.cpu_interop_threads", 1)))
            torch.set_num_threads(cpu_threads)
            try:
                torch.set_num_interop_threads(cpu_interop_threads)
            except RuntimeError as exc:
                raise RuntimeError(
                    "policy.cpu_interop_threads must be configured before CPU parallel work starts"
                ) from exc
            self.cpu_threads = cpu_threads
            self.cpu_interop_threads = cpu_interop_threads
        else:
            self.cpu_threads = None
            self.cpu_interop_threads = None

        requested_dtype = str(cfg_get(config, "policy.dtype", "auto"))
        if requested_dtype == "auto":
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = getattr(torch, requested_dtype)

        load_kwargs = {
            "revision": self.revision,
            "cache_dir": str(cache_root),
            "local_files_only": bool(cfg_get(config, "policy.local_files_only", False)),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
        dtype_argument = (
            {"dtype": dtype}
            if int(str(__version__).split(".", 1)[0]) >= 5
            else {"torch_dtype": dtype}
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **dtype_argument, **load_kwargs
        )
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        if self.resolved_revision is None and len(self.revision) == 40:
            self.resolved_revision = self.revision
        self.model.config.use_cache = False
        self.model.to(self.device)
        lora = LoraConfig(
            r=int(cfg_get(config, "policy.lora.rank", 8)),
            lora_alpha=int(cfg_get(config, "policy.lora.alpha", 16)),
            lora_dropout=float(cfg_get(config, "policy.lora.dropout", 0.05)),
            target_modules=list(
                cfg_get(config, "policy.lora.target_modules", ["q_proj", "v_proj"])
            ),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora)
        if bool(cfg_get(config, "policy.gradient_checkpointing", False)):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        self.model.train()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(cfg_get(config, "policy.learning_rate", 1.0e-4)),
            weight_decay=float(cfg_get(config, "policy.weight_decay", 0.01)),
        )
        self.max_grad_norm = float(cfg_get(config, "policy.max_grad_norm", 1.0))
        self.set_graph(graph)

    def set_graph(self, graph: DirectedOperatorGraph) -> None:
        self.graph = graph
        self.alias_by_action = {
            action: f"<A{index:03d}>" for index, action in enumerate(sorted(graph.nodes))
        }
        self.alias_by_action[END] = "<END>"

    def _format_prompt(self, task_context: str) -> str:
        mapping = "\n".join(
            f"{alias} = {action}"
            for action, alias in sorted(self.alias_by_action.items(), key=lambda item: item[1])
        )
        user_text = (
            "Generate an operator pipeline for the following optimisation task.\n"
            f"Task context:\n{task_context}\n"
            f"Available action aliases:\n{mapping}\n"
            "Return actions only; graph constraints are enforced by the decoder."
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return user_text + "\nPipeline:"

    def _encode_prompt(self, task_context: str) -> list[int]:
        return list(
            self.tokenizer(self._format_prompt(task_context), add_special_tokens=True)["input_ids"]
        )

    def _alias_tokens(self, action: str) -> tuple[int, ...]:
        tokens = self.tokenizer.encode(" " + self.alias_by_action[action], add_special_tokens=False)
        if not tokens:
            raise RuntimeError(f"Tokenizer produced no tokens for action {action}")
        return tuple(int(token) for token in tokens)

    def _next_logits(self, token_ids: list[int]):
        tensor = self.torch.tensor([token_ids], dtype=self.torch.long, device=self.device)
        return self.model(input_ids=tensor, attention_mask=self.torch.ones_like(tensor)).logits[
            0, -1
        ]

    def _sample_constrained_action(
        self,
        prefix_ids: list[int],
        actions: list[str],
        temperature: float,
        greedy: bool,
    ) -> tuple[str, list[int], Any]:
        sequences = {action: self._alias_tokens(action) for action in actions}
        selected_prefix: tuple[int, ...] = ()
        log_terms = []
        while True:
            matches = {
                action: sequence
                for action, sequence in sequences.items()
                if sequence[: len(selected_prefix)] == selected_prefix
            }
            completed = [
                action
                for action, sequence in matches.items()
                if len(sequence) == len(selected_prefix)
            ]
            if completed:
                if len(matches) != len(completed):
                    raise RuntimeError("Action aliases must be prefix-free after tokenization")
                return completed[0], list(selected_prefix), self.torch.stack(log_terms).sum()
            next_tokens = sorted({sequence[len(selected_prefix)] for sequence in matches.values()})
            if not next_tokens:
                raise RuntimeError("Constrained action trie reached a dead end")
            logits = self._next_logits(prefix_ids + list(selected_prefix)) / temperature
            allowed_logits = logits[next_tokens]
            distribution = self.torch.distributions.Categorical(logits=allowed_logits)
            local_index = (
                int(self.torch.argmax(allowed_logits).item())
                if greedy
                else int(distribution.sample().item())
            )
            chosen_token = next_tokens[local_index]
            log_terms.append(
                distribution.log_prob(self.torch.tensor(local_index, device=self.device))
            )
            selected_prefix += (chosen_token,)

    def sample(
        self,
        task_context: str,
        graph: DirectedOperatorGraph,
        max_pipeline_length: int,
        temperature: float,
        greedy: bool = False,
    ) -> PipelineSample:
        self.set_graph(graph)
        prompt_ids = self._encode_prompt(task_context)
        generated_ids: list[int] = []
        nodes: list[str] = []
        log_terms = []
        current = START
        while True:
            remaining = max_pipeline_length - len(nodes)
            actions = graph.valid_next_actions(current, remaining)
            if not actions:
                raise RuntimeError(
                    f"No valid action from {current} with {remaining} slots remaining"
                )
            with self.torch.no_grad():
                action, action_tokens, action_log_prob = self._sample_constrained_action(
                    prompt_ids + generated_ids, actions, temperature, greedy
                )
            generated_ids.extend(action_tokens)
            log_terms.append(action_log_prob)
            if action == END:
                break
            nodes.append(action)
            current = action
        return PipelineSample(
            nodes=nodes,
            log_prob=self.torch.stack(log_terms).sum().detach(),
            policy_context=task_context,
            sampling_temperature=float(temperature),
            max_pipeline_length=int(max_pipeline_length),
        )

    def _replay_action_log_prob(
        self,
        prefix_ids: list[int],
        actions: list[str],
        chosen_action: str,
        temperature: float,
    ):
        sequences = {action: self._alias_tokens(action) for action in actions}
        if chosen_action not in sequences:
            raise ValueError(f"Recorded action is no longer graph-valid: {chosen_action}")
        target = sequences[chosen_action]
        selected_prefix: tuple[int, ...] = ()
        log_terms = []
        while len(selected_prefix) < len(target):
            matches = {
                action: sequence
                for action, sequence in sequences.items()
                if sequence[: len(selected_prefix)] == selected_prefix
            }
            next_tokens = sorted({sequence[len(selected_prefix)] for sequence in matches.values()})
            chosen_token = target[len(selected_prefix)]
            if chosen_token not in next_tokens:
                raise RuntimeError("Recorded action left the constrained token trie")
            logits = self._next_logits(prefix_ids + list(selected_prefix)) / temperature
            allowed_logits = logits[next_tokens]
            local_index = next_tokens.index(chosen_token)
            log_terms.append(self.torch.log_softmax(allowed_logits, dim=-1)[local_index])
            selected_prefix += (chosen_token,)
        return list(target), self.torch.stack(log_terms).sum()

    def _replay_sample_log_prob(self, sample: PipelineSample):
        if sample.policy_context is None:
            return sample.log_prob
        temperature = float(sample.sampling_temperature)
        max_length = int(sample.max_pipeline_length)
        prompt_ids = self._encode_prompt(sample.policy_context)
        generated_ids: list[int] = []
        log_terms = []
        current = START
        for occurrence, action in enumerate(sample.nodes + [END]):
            # Reconstruct the graph mask by operator occurrence rather than by
            # token count; repeated nodes remain distinct occurrences.
            remaining = max_length - occurrence
            actions = self.graph.valid_next_actions(current, remaining)
            tokens, action_log_prob = self._replay_action_log_prob(
                prompt_ids + generated_ids, actions, action, temperature
            )
            generated_ids.extend(tokens)
            log_terms.append(action_log_prob)
            if action != END:
                current = action
        return self.torch.stack(log_terms).sum()

    def _iter_replay_sample_log_terms(self, sample: PipelineSample):
        """Yield independent token log-probs so each graph can be freed promptly."""
        temperature = float(sample.sampling_temperature)
        max_length = int(sample.max_pipeline_length)
        prompt_ids = self._encode_prompt(sample.policy_context)
        generated_ids: list[int] = []
        current = START
        for occurrence, action in enumerate(sample.nodes + [END]):
            remaining = max_length - occurrence
            actions = self.graph.valid_next_actions(current, remaining)
            sequences = {candidate: self._alias_tokens(candidate) for candidate in actions}
            if action not in sequences:
                raise ValueError(f"Recorded action is no longer graph-valid: {action}")
            target = sequences[action]
            selected_prefix: tuple[int, ...] = ()
            while len(selected_prefix) < len(target):
                matches = {
                    candidate: sequence
                    for candidate, sequence in sequences.items()
                    if sequence[: len(selected_prefix)] == selected_prefix
                }
                next_tokens = sorted(
                    {sequence[len(selected_prefix)] for sequence in matches.values()}
                )
                chosen_token = target[len(selected_prefix)]
                if chosen_token not in next_tokens:
                    raise RuntimeError("Recorded action left the constrained token trie")
                logits = (
                    self._next_logits(prompt_ids + generated_ids + list(selected_prefix))
                    / temperature
                )
                yield self.torch.log_softmax(logits[next_tokens], dim=-1)[
                    next_tokens.index(chosen_token)
                ]
                selected_prefix += (chosen_token,)
            generated_ids.extend(target)
            if action != END:
                current = action

    def _action_log_scores(
        self, prefix_ids: list[int], actions: list[str], temperature: float
    ) -> dict[str, float]:
        raw_scores = []
        with self.torch.no_grad():
            for action in actions:
                working = list(prefix_ids)
                score = 0.0
                for token in self._alias_tokens(action):
                    logits = self._next_logits(working) / temperature
                    score += float(self.torch.log_softmax(logits, dim=-1)[token].item())
                    working.append(token)
                raw_scores.append(score)
        maximum = max(raw_scores)
        log_total = maximum + math.log(sum(math.exp(score - maximum) for score in raw_scores))
        return {action: score - log_total for action, score in zip(actions, raw_scores)}

    def beam_search(
        self,
        task_context: str,
        graph: DirectedOperatorGraph,
        max_pipeline_length: int,
        temperature: float,
        beam_size: int,
    ) -> list[PipelineSample]:
        self.set_graph(graph)
        prompt_ids = self._encode_prompt(task_context)
        beams = [_Beam([], [], 0.0)]
        completed: dict[tuple[str, ...], _Beam] = {}
        for _ in range(max_pipeline_length + 1):
            expanded = []
            for beam in beams:
                if beam.finished:
                    expanded.append(beam)
                    continue
                current = beam.nodes[-1] if beam.nodes else START
                remaining = max_pipeline_length - len(beam.nodes)
                actions = graph.valid_next_actions(current, remaining)
                for action, score in self._action_log_scores(
                    prompt_ids + beam.token_ids, actions, temperature
                ).items():
                    next_beam = _Beam(
                        nodes=list(beam.nodes),
                        token_ids=beam.token_ids + list(self._alias_tokens(action)),
                        score=beam.score + score,
                        finished=action == END,
                    )
                    if action == END:
                        completed.setdefault(tuple(next_beam.nodes), next_beam)
                    else:
                        next_beam.nodes.append(action)
                    expanded.append(next_beam)
            beams = sorted(expanded, key=lambda item: item.score, reverse=True)[:beam_size]
            if beams and all(beam.finished for beam in beams):
                break
        selected = sorted(completed.values(), key=lambda item: item.score, reverse=True)
        return [PipelineSample(nodes=beam.nodes) for beam in selected[:beam_size]]

    def update(self, candidates: list[CandidateResult]) -> PolicyUpdateStats:
        usable = [
            candidate
            for candidate in candidates
            if candidate.sample.policy_context is not None or candidate.sample.log_prob is not None
        ]
        if not usable:
            return PolicyUpdateStats(loss=0.0, grad_norm=0.0)
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for candidate in usable:
            advantage = self.torch.tensor(
                candidate.normalized_reward,
                dtype=self.torch.float32,
                device=self.device,
            ).detach()
            if candidate.sample.policy_context is not None:
                # Each trie token comes from an independent full-model forward.
                # Backpropagating it immediately is algebraically identical to
                # summing all token log-probs first, while avoiding retention of
                # every forward graph until the end of the pipeline.
                for log_prob in self._iter_replay_sample_log_terms(candidate.sample):
                    loss = -advantage * log_prob / len(usable)
                    loss_value += float(loss.detach().cpu().item())
                    loss.backward()
            else:
                log_prob = candidate.sample.log_prob
                if log_prob is None:
                    continue
                loss = -advantage * log_prob / len(usable)
                loss_value += float(loss.detach().cpu().item())
                loss.backward()
        grad_norm = self.torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )
        self.optimizer.step()
        return PolicyUpdateStats(
            loss=loss_value,
            grad_norm=float(grad_norm.detach().cpu().item())
            if hasattr(grad_norm, "detach")
            else float(grad_norm),
        )

    def training_state(self) -> dict[str, Any]:
        from peft import get_peft_model_state_dict

        return {
            "adapter": {
                key: value.detach().cpu()
                for key, value in get_peft_model_state_dict(self.model).items()
            },
            "optimizer": self.optimizer.state_dict(),
            "model_name": self.model_name,
            "revision": self.revision,
            "aliases": self.alias_by_action,
            "resolved_revision": self.resolved_revision,
        }

    def load_training_state(self, state: dict[str, Any], load_optimizer=True):
        from peft import set_peft_model_state_dict

        if state.get("model_name", self.model_name) != self.model_name:
            raise ValueError("Checkpoint policy model does not match configured model_name")
        saved_resolved = state.get("resolved_revision")
        current_resolved = self.resolved_revision
        if saved_resolved and current_resolved and saved_resolved != current_resolved:
            raise ValueError("Checkpoint base-model commit does not match the loaded model")
        if state.get("revision", self.revision) != self.revision and not (
            saved_resolved and saved_resolved == current_resolved
        ):
            raise ValueError("Checkpoint policy revision does not match configured revision")
        set_peft_model_state_dict(self.model, state["adapter"])
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def save_adapter(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory / "adapter", safe_serialization=True)
