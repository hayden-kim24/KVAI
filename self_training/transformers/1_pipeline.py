# Last Updated Date: 2024-07-16
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Status: Training
# Based on: Hugging Face NLP Course Chapter 1, https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt


from transformers import pipeline

#Zero-Shot-Classification
classifier = pipeline("zero-shot-classification")
a = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
print(a)

classifier = pipeline("zero-shot-classification")
a = classifier(
    "Can you give me the next hearing date for the case? I really miss my son",
    candidate_labels=["next hearing date", "judge", "name"],
)
print(a)

classifier = pipeline("zero-shot-classification")
a = classifier(
    "what even is this, I don't know what I'm supposed to say",
    candidate_labels=["judge", "case name", "confusion", "greeting"],
)
print(a)


classifier = pipeline("zero-shot-classification",model = "facebook/bart-large-mnli")
a = classifier(
    "I don't know the FIR number",
    candidate_labels=["confusion", "fir number", "name"],
)
print(a)

classifier = pipeline("zero-shot-classification")
a = classifier(
    "hey could you give me the case date and also the laywer name",
    candidate_labels=["confusion", "fir number", "greeting", "date", "name"],
)
print(a)