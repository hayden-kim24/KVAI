# Last Updated Date: 2024-07-16
# Last Updated By: Hayden Kim (hayden [dot] kim [at] stanford [dot] edu)
# Purpose: Self-Training (Not to be used for actual app)
# Status: DONE
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
# print(a)

# classifier = pipeline("zero-shot-classification")
# a = classifier(
#     "what even is this, I don't know what I'm supposed to say",
#     candidate_labels=["judge", "case name", "confusion", "greeting"],
# )
# print(a)

# classifier = pipeline("zero-shot-classification",model = "facebook/bart-large-mnli")
# a = classifier(
#     "I don't know the FIR number",
#     candidate_labels=["confusion", "fir number", "name"],
# )
# print(a)

# classifier = pipeline("zero-shot-classification")
# a = classifier(
#     "hey could you give me the case date and also the laywer name",
#     candidate_labels=["confusion", "fir number", "greeting", "date", "name"],
# )
# print(a)

# #named entity recognition
# ner = pipeline("ner", grouped_entities=True)
# a = ner("Hey could you pull up some information about John Doe? And maybe about NYPD? Also, do you have any case for Ramgopalpet?")
# print(a)
# #it worked yippie

context_a = {
    "CNR Number": "NYFL252721821656",
    "Case Stage": "Pending",
    "Case Type": "Civil",
    "Court Number": 53,
    "Decision Date": "2021-08-06",
    "FIR Number": 916,
    "Filing Date": "2020-03-24",
    "Filing Number": "2391/7933",
    "First Hearing Date": "2020-11-28",
    "Judge": "Randy Aguilar",
    "Nature of Disposal": "Contested--CONVICTED",
    "Next Hearing Date": "2021-03-28",
    "Past Case History": {
        "business_on_date": "2021-05-26",
        "hearing_dates": [
            "2020-08-13",
            "2020-05-27",
            "2021-04-26",
            "2020-08-06"
        ],
        "judge": "Judge U",
        "order_dates": [
            "2021-04-13",
            "2020-09-18"
        ],
        "order_numbers": [
            1100,
            4186
        ],
        "purposes_of_hearing": [
            "Hard police.",
            "Role begin.",
            "Attack true miss know.",
            "Quite spring."
        ]
    },
    "Petitioner and Advocate": "Nathan Brown",
    "Police Station": "New Richardton",
    "Registration Date": "2024-06-25",
    "Registration Number": "8733/3947",
    "Respondent and Advocate": "Brett Lawrence IV",
    "Under Acts": "Act898",
    "Under Sections": 7402,
    "Year": 2010
}

str_ctxt = "CNR Number: NYFL252721821656. Case Stage: Pending. Case Type: Civil. Court Number: 53. Decision Date: 2021-08-06. FIR Number: 916. Filing Date: 2020-03-24. Filing Number: 2391/7933. First Hearing Date: 2020-11-28. Judge: Randy Aguilar. Nature of Disposal: Contested--CONVICTED. Next Hearing Date: 2021-03-28. Past Case History: business_on_date: 2021-05-26. hearing_dates: 2020-08-13, 2020-05-27, 2021-04-26, 2020-08-06. judge: Judge U. order_dates: 2021-04-13, 2020-09-18. order_numbers: 1100, 4186. purposes_of_hearing: Hard police., Role begin., Attack true miss know., Quite spring. Petitioner and Advocate: Nathan Brown. Police Station: New Richardton. Registration Date: 2024-06-25. Registration Number: 8733/3947. Respondent and Advocate: Brett Lawrence IV. Under Acts: Act898. Under Sections: 7402. Year: 2010."


#question answering
question_answerer = pipeline("question-answering")
a = question_answerer(
    question="what's the stage for the case right now?",
    context=str_ctxt,
)

print(a)


#summary

summarizer = pipeline("summarization", max_length = 40)
a = summarizer(str_ctxt,
)
print(a)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")