# Last Updated Date: 2024-07-17
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Purpose: Trying out PT ver of model initailzation: Self-Training (Not to be used for actual app)
# Status: Done


from transformers import AutoModel, BertConfig, BertModel

model = BertModel.from_pretrained("bert-base-uncased")

model.save_pretrained("self_training/transformers/3_model_initiazation_pt/dictionary_on_my_computer")