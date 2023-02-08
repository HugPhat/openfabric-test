import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time

###########################################################
# Global model
###########################################################
from ai import QAfromHF
model = QAfromHF(
    clf_model_id="jonaskoenig/topic_classification_04",
    resp_model_id="gpt2-large",
    topic_threshold=0.8,
    device=-1,
)

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        response = model([text])[0]
        output.append(response)

    return SimpleText(dict(text=output))
