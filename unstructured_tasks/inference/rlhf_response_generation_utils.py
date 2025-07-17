from typing import Any, Optional, List, Dict
import json
from multiprocessing import dummy as threading_tools
import functools
import tqdm
import dataclasses
import vllm
# import gllm
import transformers


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return all records as a list.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict[str, Any]]: List of JSON objects from the file
        
    Raises:
        JSONDecodeError: If a line contains invalid JSON
        FileNotFoundError: If the file doesn't exist
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                record = json.loads(line)
                records.append(record)
    return records


class FixedQwen(transformers.Qwen2ForCausalLM):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        model.tokenizer = tokenizer
        return model

    def get_tokenizer(self):
        return self.tokenizer


def get_sampling_params_for_model(
    model: vllm.LLM | transformers.PreTrainedModel,
    temperature: float,
    max_tokens: int,
    n: int,
    stop_str: str,
    include_stop_str_in_output: bool,
    stop_token: Optional[int] = None,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
    gllm_mode: str = "completions",
) -> vllm.SamplingParams | transformers.GenerationConfig:
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
        low_memory = ~do_sample

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
            low_memory=low_memory
        )
    else:
        raise ValueError(f"Invalid model type {type(model)}")

class VLLMLookingOutput:
    # Looks like a VLLM LLM.generate output.
    # Duck typing says Quack, Quack!
    @dataclasses.dataclass
    class VLLMSingleOutput:
        text: str

    outputs: list[VLLMSingleOutput]
    prompt: str

    def __init__(self, outputs: list[str], prompt: str):
        self.outputs = [self.VLLMSingleOutput(output) for output in outputs]
        self.prompt = prompt

def generate_from_model(
    model: vllm.LLM | transformers.PreTrainedModel,
    prompts: list[str],
    sampling_params: (
        vllm.SamplingParams | transformers.GenerationConfig
    ),
) -> list[str]:
    """Generate continutions from an LLM for given prompts and sampling params.

    Provides an abstraction over the different interfaces of vllm.LLM and transformers.PreTrainedModel.
    """
    if isinstance(model, vllm.LLM):
        assert isinstance(sampling_params, vllm.SamplingParams)
        results = model.generate(prompts, sampling_params=sampling_params)
        responses = [[x.text for x in prompt.outputs] for prompt in results]
        responses = sum(responses, [])
        return responses

    elif isinstance(model, transformers.PreTrainedModel):
        assert isinstance(sampling_params, transformers.GenerationConfig)
        tokenizer = model.get_tokenizer()
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        length = inputs["input_ids"].shape[1]

        results = model.generate(
            **inputs, generation_config=sampling_params, tokenizer=tokenizer
        )
        results = results[:, length:]
        responses = tokenizer.batch_decode(results, skip_special_tokens=True)
        return responses


def generate_comments_through_kw(
    articles: list[str],
    prompt_keywords: str,
    model: vllm.LLM,
    comment_stop_str: str = "\n",
    comment_stop_token: int = 14374,
    kw_stop_str: str = "###",
    keyword_temp: float = 1.1,
    comment_temp: float = 0.5,
    n_comments: int = 10,
    max_kw_tokens: int = 10,
    max_comment_tokens: int = 300,
) -> list[list[str]]:

    kw_sampling_params = vllm.SamplingParams(
        temperature=keyword_temp,
        max_tokens=max_kw_tokens,
        n=n_comments,
        stop=kw_stop_str,
        include_stop_str_in_output=True,
    )

    prompts = [prompt_keywords.format(article=article) for article in articles]
    
    print("**************************************prompts", prompts)

    
    def enforce_stop_str(text: str, stop_str: str= kw_stop_str) -> str:
        text = text.rstrip()
        while text.endswith(","):
            text = text[:-1]
        if not stop_str in text:
            text += " " + stop_str
        return text
    
    keywords = model.generate(prompts, sampling_params=kw_sampling_params)
    keywords = [([enforce_stop_str(x.text) for x in prompt.outputs], prompt.prompt) for prompt in keywords]
    keywords = [VLLMLookingOutput(*outputs) for outputs in keywords]
    
    keywords_generated = [[x.text for x in prompt.outputs] for prompt in keywords]
    keywords_generated = sum(keywords_generated, [])

    comment_sampling_params = vllm.SamplingParams(
        temperature=comment_temp,
        max_tokens=max_comment_tokens,
        n=1,
        stop=comment_stop_str,
        include_stop_str_in_output=True,
    )

    prompts = [[prompt.prompt + x.text for x in prompt.outputs] for prompt in keywords]  # type: ignore
    prompts = sum(prompts, [])

    comments = generate_from_model(
        model, prompts, sampling_params=comment_sampling_params
    )

    return (
        _post_process_comments(comments, n_comments, len(articles)),
        keywords_generated,
    )


def generate_comments_directly(
    articles: list[str],
    prompt: str,
    model: vllm.LLM,
    comment_stop_str: str = "###",
    comment_stop_token: int = 14374,
    temperature: float = 0.5,
    n_comments: int = 10,
    max_comment_tokens: int = 300,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
) -> list[list[str]]:

    sampling_params = get_sampling_params_for_model(
        model=model,
        temperature=temperature,
        max_tokens=max_comment_tokens,
        n=n_comments,
        stop_str=comment_stop_str,
        stop_token=comment_stop_token,
        include_stop_str_in_output=False,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
    )
    prompts = [prompt.format(article=article) for article in articles]
    
    
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^prompts", prompts)

    comments = generate_from_model(model, prompts, sampling_params)

    return _post_process_comments(comments, n_comments, len(articles))


def _post_process_comments(
    comments: Any, n_comments: int, n_articles: int
) -> list[list[str]]:
    """Extract comments from nested object return by vllm.LLM.generate().

    Args:
        comments: List of vllm responses objects.
        n_comments: Number of comments per article.
    Returns:
        comments: list of list of strings: The constituent list i contains the comments
        generated for the i-th article.
    """

    assert len(comments) == n_articles * n_comments

    comments = [
        comments[i : i + n_comments] for i in range(0, len(comments), n_comments)
    ]

    return comments