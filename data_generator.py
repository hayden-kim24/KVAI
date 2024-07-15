from faker import Faker
import random
from datetime import datetime, timedelta
import json

faker = Faker()

def generate_case_data():
    
    case_type = random.choice(['Civil', 'Criminal', 'Family', 'Commercial'])
    
    filing_number = faker.bothify(text='####/####')

    filing_date = faker.date_this_decade()

    registration_number = faker.bothify(text='####/####')

    registration_date = faker.date_this_decade()

    cnr_number = faker.bothify(text='????############', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    first_hearing_date = faker.date_between(start_date=filing_date, end_date=filing_date + timedelta(days=365))

    next_hearing_date = first_hearing_date + timedelta(days=random.randint(7, 365))

    decision_date = next_hearing_date + timedelta(days=random.randint(1, 365))

    case_stage = random.choice(['Pending', 'In Progress', 'Disposed'])

    nature_of_disposal = random.choice(['Contested--CONVICTED', 'Under Consideration', 'Adjourned'])

    court_number = random.randint(1, 99)

    judge = faker.name()

    petitioner_advocate = faker.name()
    
    respondent_advocate = faker.name()

    under_acts = faker.bothify(text='Act###')

    under_sections = faker.random_number(digits=4)

    police_station = faker.city()

    fir_number = faker.random_number(digits=3)

    year = random.randint(2000, 2023)

    past_case_history = {
        "judge": faker.bothify(text= "Judge ?", letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "business_on_date": faker.date_between(start_date=filing_date, end_date=decision_date),
        "hearing_dates": [faker.date_between(start_date=filing_date, end_date=decision_date) for _ in range(4)],
        "purposes_of_hearing": [faker.sentence(nb_words=3) for _ in range(4)],
        "order_numbers": [faker.random_number(digits=4) for _ in range(2)],
        "order_dates": [faker.date_between(start_date=filing_date, end_date=decision_date) for _ in range(2)]
    }

    case_data = {
        'Case Type': case_type,
        'Filing Number': filing_number,
        'Filing Date': filing_date,
        'Registration Number': registration_number,
        'Registration Date': registration_date,
        'CNR Number': cnr_number,
        'First Hearing Date': first_hearing_date,
        'Next Hearing Date': next_hearing_date,
        'Decision Date': decision_date,
        'Case Stage': case_stage,
        'Nature of Disposal': nature_of_disposal,
        'Court Number': court_number,
        'Judge': judge,
        'Petitioner and Advocate': petitioner_advocate,
        'Respondent and Advocate': respondent_advocate,
        'Under Acts': under_acts,
        'Under Sections': under_sections,
        'Police Station': police_station,
        'FIR Number': fir_number,
        'Year': year,
        'Past Case History': past_case_history
    }

    return case_data

# Generating 100000 sample cases and write them to a JSON file
sample_cases = []
for i in range(10000):
    case = generate_case_data()
    sample_cases.append(case)
    # print(f"Case {i + 1}:")
    # for key, value in case.items():
    #     if isinstance(value, list):
    #         print(f"{key}: {', '.join(map(str, value))}")
    #     else:
    #         print(f"{key}: {value}")
    # print("\n")

# Write to JSON file
output_file = 'sample_cases.json'

try:
    with open(output_file, 'w') as f:
        json.dump(sample_cases, f, indent=4, sort_keys=True, default=str)
    print(f"Sample data has been generated and saved to '{output_file}'.")
except Exception as e:
    print(f"Error occurred while writing to '{output_file}': {e}")


# with open(output_file, 'w') as f:
#     json.dumps(sample_cases, indent=4, sort_keys=True, default=str)
    

print(f"Sample data has been generated and saved to '{output_file}'.")


