from typing import Union, List

from transformers import pipeline


class ResponderFromHF:
    """Loading text classifier from huggingface(hf)
    Args:
        model_id (str): model id from hf
        tokenizer (str): tokenizer id from hf
        task(str): hf task name (https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task). 
            Allowed task: "text2text-generation", "text-generation", "question-answering"
        device (int, optional): GPU device id starting from 0, if using cpu set -1. Defaults to -1.
    """

    def __init__(
        self,
        model_id: str,
        tokenizer: str,
        task="text-generation",
        device: int = -1,
    ):
        assert task in ["text-generation", "text2text-generation", "question-answering"]
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.hf_pipeline = pipeline(
            task=task,
            model=self.model_id,
            tokenizer=self.tokenizer,
            framework="pt",
            device=device
        )

    def __call__(self, text: Union[List[str], str], **kwargs) -> List[str]:
        """ Make prediction

        Args:
            text (Union[List[str], str]): input text

        Returns:
            List[str]: output text of model
            
        Examples:
        >>> input = ["text 1", "text 2"]
        >>> model(input)
        >>> ['output 1', 'output 2']
        
        """

        return self.hf_pipeline(text, return_full_text=False)
