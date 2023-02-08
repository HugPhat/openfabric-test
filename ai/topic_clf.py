from typing import Union, List

from transformers import pipeline

class TopicClassifierFromHF:
    """Loading text classifier from huggingface(hf)
    Args:
        model_id (str): model id from hf
        tokenizer (str): tokenizer id from hf
        device (int, optional): GPU device id starting from 0, if using cpu set -1. Defaults to -1.
    """
    def __init__(
        self, 
        model_id:str,
        tokenizer:str,
        device:int=-1,
    ):
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.hf_pipeline = pipeline(
            task="text-classification",
            model=self.model_id,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            device=device
        )
        
    def __call__(self, text:Union[List[str], str]) -> List:
        """ Make prediction

        Args:
            text (Union[List[str], str]): input text

        Returns:
            List: list of dict, each dict contains keys "label": str, "score": float
            
        Examples:
        >>> input = ["text 1", "text 2"]
        >>> model(input)
        >>> [[{"label": "class A", "score": 0.9}, {"label": "class B", "score": 0.1}], [{"label": "class A", "score": 0.2}, {"label": "class B", "score": 0.8}]]
        
        """
                    
        return self.hf_pipeline(text)