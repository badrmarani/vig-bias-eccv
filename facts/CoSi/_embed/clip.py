from typing import Dict, Union

from .encoder import Encoder
from facts.vig import ViG

def clip(
    variant: str = "ViT-B/32",
    device: Union[int, str] = "cpu",
    model_path: str = None,
    dataset_name: str = "waterbirds",
    mask_threshold: float = 0.0,
    apply_masks: bool = False,
) -> Dict[str, Encoder]:
    """Contrastive Language-Image Pre-training (CLIP) encoders [radford_2021]_. Includes
    encoders for the following modalities:

    - "text"
    - "image"

    Encoders will map these different modalities to the same embedding space.

    Args:
        variant (str, optional): A model name listed by `clip.available_models()`, or
            the path to a model checkpoint containing the state_dict. Defaults to
            "ViT-B/32".
        device (Union[int, str], optional): The device on which the encoders will be
            loaded. Defaults to "cpu".


    .. [radford_2021]

        Radford, A. et al. Learning Transferable Visual Models From Natural Language
        Supervision. arXiv [cs.CV] (2021)
    """
    try:
        from clip import load, tokenize
    except ImportError:
        raise ImportError(
            "To embed with CLIP run pip install git+https://github.com/openai/CLIP.git"
            "and install domino with the `clip` submodule. For example, "
            "`pip install domino[clip]`"
        )

    model, preprocess = load(variant, device=device)
    
    if apply_masks:
        preprocess = ViG(
            model=model_path,
            device_or_id=device,
            dataset_name=dataset_name,
            mask_threshold=mask_threshold,
        )
    
    return {
        "image": Encoder(encode=model.encode_image, preprocess=preprocess),
        "text": Encoder(
            # need to squeeze out the batch dimension for compatibility with collate
            encode=model.encode_text,
            preprocess=lambda x: tokenize(x, truncate=True).squeeze(0),
        ),
    }
