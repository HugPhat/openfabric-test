from typing import Any, Union, List

import ai.qa as qa_module
import ai.topic_clf as topic_module

class _AIBase:
    
    def __init__(self, **kwargs) -> None:
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()

class QAfromHF(_AIBase):
    """Question answering science topic using hf model
    Args:
        clf_model_id (str): topic classifier hf id
        resp_model_id (str): text generator hf id
        resp_task (str): task of text generator. Defeault to "text-generation"
        device (int, optional): index of gpu device start from 0, set -1 to use cpu. Defaults to -1.
        topic_threshold (float, optional): threshold of topic classification. Defaults to 0.8.
        interested_clf_class (str, optional): interested topic/label of topic classifier. Defaults to "science".
    """        
    def __init__(
        self, 
        clf_model_id:str,
        resp_model_id:str,
        resp_task:str = "text-generation",
        device:int = -1,
        topic_threshold:float = 0.8,
        interested_clf_class:str = "science",
    ) -> None:
        self.clf_model_id = clf_model_id
        self.resp_model_id = resp_model_id
        self.resp_task = resp_task
        self.device = device
        self.topic_threshold = topic_threshold
        self.interested_clf_class = interested_clf_class
        
        print(f"- Loading topic classifier model id: {self.clf_model_id}")
        self.clf_model = topic_module.TopicClassifierFromHF(
            model_id=self.clf_model_id,
            tokenizer=self.clf_model_id,
            device=self.device
        )
        
        print(f"- Loading text generator model id: {self.resp_model_id}")
        self.responder_model = qa_module.ResponderFromHF(
            model_id=self.resp_model_id,
            tokenizer=self.resp_model_id,
            task=self.resp_task,
            device=self.device
        )
        
    @property
    def INCORRECT_TOPIC_RESPONSE(self) -> str:
        _msg = f"Irrelevant topic, please input text in {self.interested_clf_class}"
        return _msg
    
    def __call__(self, text: Union[str, List[str]]) -> List[str]:
        """Make prediction

        Args:
            text (Union[str, List[str]]): input text

        Returns:
            List[str]: list of answer
        """
        
        topics_and_scores = self.clf_model(text)
        output_text = []
        # store index, input for responder modle
        for i, tp_n_sc in enumerate(topics_and_scores):
            _ans = ""
            for each in tp_n_sc:
                if self.interested_clf_class.lower() in each["label"].lower() \
                and self.topic_threshold < each["score"]:
                    _ans = self.responder_model(text[i])[0]["generated_text"]
                    break
            if not _ans:
                output_text.append(self.INCORRECT_TOPIC_RESPONSE)
            else:
                output_text.append(_ans)
                
        return output_text
