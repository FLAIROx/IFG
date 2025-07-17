from multiprocessing import pool
from typing import Optional
import copy

import transformers
import vllm
from hendrycks_math.utils import math_types

import gllm


def generate_from_model(
    model: vllm.LLM | transformers.PreTrainedModel | gllm.GLLM,
    prompts: list[str] | list[math_types.JsonMessages],
    sampling_params: math_types.GenericSamplingParams,
) -> list[str]:
    """Generate continutions from an LLM for given prompts and sampling params.

    Provides an abstraction over the different interfaces of gllm.GLLM, vllm.LLM and transformers.PreTrainedModel. This function also hides the differences between sampling from chat and base
    models.
    """
    assert prompts, "Prompts must be non-empty"
    if isinstance(model, vllm.LLM):
        assert prompts and isinstance(prompts[0], str)
        assert isinstance(prompts[0], str)
        assert isinstance(sampling_params, vllm.SamplingParams)
        results = model.generate(prompts, sampling_params=sampling_params) # type: ignore
        responses = [[x.text for x in prompt.outputs] for prompt in results]
        responses = sum(responses, [])
        return responses

    elif isinstance(model, transformers.PreTrainedModel):
        assert isinstance(prompts[0], str)
        assert isinstance(sampling_params, transformers.GenerationConfig) # type: ignore
        tokenizer = model.get_tokenizer() # type: ignore
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        length = inputs["input_ids"].shape[1]

        results = model.generate( # type: ignore
            **inputs, generation_config=sampling_params, tokenizer=tokenizer
        )
        results = results[:, length:]
        responses = tokenizer.batch_decode(results, skip_special_tokens=True)
        return responses

    elif isinstance(model, gllm.GLLM):

        def get_completion(prompt):
            assert isinstance(sampling_params, dict) # type: ignore
            local_sampling_params = copy.deepcopy(sampling_params)
            model_name = local_sampling_params.pop("model")

            if isinstance(prompt[0], dict):  # chat mode
                response: list[str] = model.get_chat_completion( # type: ignore
                    model_name,
                    prompt,
                    **local_sampling_params,
                    return_mode="primitives",
                )
                response = [r["content"] for r in response] # type: ignore
            else:  # direct completion
                assert isinstance(prompt, str)
                response: list[str] = model.get_completions( # type: ignore
                    model_name,
                    prompt,
                    **local_sampling_params,
                    return_mode="primitives",
                )
            
            # We need to add the stop string to the responses as OpenAI compatibile
            # servers omit the stop string from the response.
            response = [r + local_sampling_params.get("stop", "") for r in response]
            return response

        with pool.ThreadPool(len(prompts)) as p:
            responses = p.map(get_completion, prompts)
        responses = sum(responses, [])
        return responses

    else:
        raise ValueError(f"Invalid model type {type(model)=}")


def get_sampling_params_for_model(
    model: vllm.LLM | transformers.PreTrainedModel | gllm.GLLM,
    temperature: float,
    max_tokens: int,
    n: int,
    stop_str: str,
    include_stop_str_in_output: bool,
    stop_token: Optional[int] = None,
    model_name: Optional[str] = None,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
    gllm_mode: str = "completions",
) -> math_types.GenericSamplingParams:
    """Create a SamplingParams object for the model.

    Handles the different interfaces of vllm.LLM and transformers.PreTrainedModel.
    """

    if isinstance(model, vllm.LLM):
        assert num_beam_groups == 1, "vllm.LLM does not support num_beam_groups > 1"
        assert (
            diversity_penalty < 1e-5
        ), "vllm.LLM does not support diversity_penalty > 0.0"
        assert stop_token is None, "vllm.LLM does not support stop token."
        return vllm.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop_str,
            include_stop_str_in_output=include_stop_str_in_output,
        )

    elif isinstance(model, transformers.PreTrainedModel):
        num_beams = n if num_beam_groups > 1 else 1
        do_sample = False if num_beam_groups > 1 else True

        return transformers.GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            num_beams=num_beams,
            stop_strings=[stop_str],
            eos_token_id=stop_token,
            include_stop_str_in_output=include_stop_str_in_output,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )
    elif isinstance(model, gllm.GLLM):
        if gllm_mode == "chat":
            raise NotImplementedError("Chat mode is not supported for gllm.GLLM, further testing needed.")
        assert model_name is not None, "model_name is required for gllm.GLLM"
        assert num_beam_groups == 1, "gllm.GLLM does not support num_beam_groups > 1"
        assert (
            diversity_penalty < 1e-5
        ), "gllm.GLLM does not support diversity_penalty > 0.0"
        assert stop_token is None, "gllm.GLLM does not support stop token."
        args = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
            "stop": stop_str,
            "model": model_name,
        }
        return math_types.GllmGenerationArgs(args)
    else:
        raise ValueError(f"Invalid model type {type(model)}")
