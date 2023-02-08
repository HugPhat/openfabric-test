# Summary
Using public Huggingface models. Basic workflow:

`Topic classification => Text generation/Text2text generation`

Guide:
+ Using cpu, set GPU device by setting device params in QAfromHF
+ Support all text classification and text generation model of hf, params `clf_model_id` for text classification, `resp_model_id` for text generation.
+ To get better answer, consider to larger text generation model such as bloom, cosmo-xl,... If you have strong hardware.

# Possible approaches:
1. Larger model
2. Information retriever + QA model

# Some note:
1. NLP is not my expertise, I'm happy to hear any feedback.
2. Due to current work, total time I spent for it just ~5-6h.

