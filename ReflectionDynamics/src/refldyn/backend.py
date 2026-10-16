import math


class InferenceBackend:
    def __init__(self, model_id, adapter=None, max_memory_gib=34, temperature=0.7):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for model inference")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.temperature = temperature
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=False, low_cpu_mem_usage=True,
            device_map="auto", attn_implementation="sdpa",
            max_memory={i: f"{max_memory_gib}GiB" for i in range(torch.cuda.device_count())},
        )
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        if self.input_device.type != "cuda":
            raise RuntimeError("Embeddings offloaded to CPU; reduce per-replica model size")

    def generate(self, prompt, seed, max_new_tokens, sample=True):
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        torch.manual_seed(int(seed))
        kwargs = {"max_new_tokens": int(max_new_tokens), "pad_token_id": self.tokenizer.eos_token_id,
                  "do_sample": bool(sample)}
        if sample:
            kwargs.update({"temperature": self.temperature, "top_p": 0.95})
        with torch.inference_mode():
            result = self.model.generate(**inputs, **kwargs)
        tokens = result[0, inputs["input_ids"].size(1):]
        return self.tokenizer.decode(tokens, skip_special_tokens=True), int(tokens.numel())

    def action_probability(self, prompt):
        torch = self.torch
        prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prefix:
            raise ValueError("Empty decision prompt")
        scores = []
        for action in ("COMMIT", "REFLECT"):
            continuation = self.tokenizer.encode(action, add_special_tokens=False)
            if not continuation:
                raise ValueError("Missing candidate action tokens")
            tokens = torch.tensor([prefix + continuation], dtype=torch.long, device=self.input_device)
            with torch.inference_mode():
                logits = self.model(input_ids=tokens).logits[0, len(prefix) - 1:-1]
                next_ids = tokens[0, len(prefix):].to(logits.device)
                score = torch.log_softmax(logits.float(), dim=-1).gather(1, next_ids[:, None]).sum()
                scores.append(float(score.cpu()))
        peak = max(scores)
        weights = [math.exp(score - peak) for score in scores]
        return float(weights[1] / sum(weights)), scores
