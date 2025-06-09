import h5py
import os
import sys

# Data preprocessing dan encoding
def encode(value, mapping):
    return mapping.get(value, 0)

def prepare_input(row):
    gender = encode(row[1], {"Male": 1, "Female": 0})
    married = encode(row[2], {"Yes": 1, "No": 0})
    dependents = encode(row[3], {"0": 0, "1": 1, "2": 2, "3+": 3})
    education = encode(row[4], {"Graduate": 1, "Not Graduate": 0})
    self_emp = encode(row[5], {"Yes": 1, "No": 0})
    applicant_income = int(row[6]) if row[6] else 0
    coapplicant_income = float(row[7]) if row[7] else 0.0
    loan_amount = float(row[8]) if row[8] else 120.0
    loan_term = float(row[9]) if row[9] else 360.0
    credit_history = float(row[10]) if row[10] else 1.0
    property_area = encode(row[11], {"Urban": 2, "Semiurban": 1, "Rural": 0})
    
    return [
        gender, married, dependents, education, self_emp,
        applicant_income, coapplicant_income, loan_amount,
        loan_term, credit_history, property_area
    ]

def predict_with_details(x, model):
    root_idx = model.attrs["root_feature"]
    left_idx = model.attrs["left_feature"]
    right_idx = model.attrs["right_feature"]
    
    left_0 = model.attrs["left_prediction_if_0"]
    left_1 = model.attrs["left_prediction_if_1"]
    right_0 = model.attrs["right_prediction_if_0"]
    right_1 = model.attrs["right_prediction_if_1"]

    if x[root_idx] == 1:
        if x[right_idx] == 1:
            prediction = right_1
            path = f"Root[F{root_idx}=1] -> Right[F{right_idx}=1]"
        else:
            prediction = right_0
            path = f"Root[F{root_idx}=1] -> Right[F{right_idx}=0]"
    else:
        if x[left_idx] == 1:
            prediction = left_1
            path = f"Root[F{root_idx}=0] -> Left[F{left_idx}=1]"
        else:
            prediction = left_0
            path = f"Root[F{root_idx}=0] -> Left[F{left_idx}=0]"
    
    return prediction, path

def format_prediction(test_case, prediction, path):
    status = "APPROVED" if prediction == 1 else "REJECTED"
    emoji = "✅" if prediction == 1 else "❌"
    
    print("\n" + "="*50)
    print(f"LOAN APPLICATION PREDICTION: {status} {emoji}")
    print("="*50)
    print(f"Application ID: {test_case[0]}")
    print(f"Gender: {test_case[1]}")
    print(f"Married: {test_case[2]}")
    print(f"Prediction Result: {prediction} (1 = Approved, 0 = Rejected)")
    print("\nDecision Path:")
    print(path)
    print("\nKey Features:")
    print(f"- Credit History: {'Good' if test_case[10] == '1' else 'Bad'}")
    print(f"- Income: {test_case[6]} (Applicant) + {test_case[7]} (Co-applicant)")
    print(f"- Loan Amount: {test_case[8]}")
    print("="*50 + "\n")

def main():
    # Inisialisasi dengan default value
    h5_path = None
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    
    try:
        # Definisikan path file
        h5_path = os.path.join(script_dir, "tree_output.h5")
        
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Model file not found at: {h5_path}")

        test_cases = [
            {
                "data": [
                    "LP001", "Male", "Yes", "1", "Graduate", "No",
                    "5000", "0", "150", "360", "1", "Urban", "Y"
                ],
                "description": "Good credit history, high income"
            },
            {
                "data": [
                    "LP002", "Female", "No", "0", "Not Graduate", "Yes",
                    "1000", "500", "50", "180", "0", "Rural", "N"
                ],
                "description": "Bad credit history, low income"
            }
        ]

        with h5py.File(h5_path, "r") as model:
            for case in test_cases:
                x = prepare_input(case["data"])
                prediction, path = predict_with_details(x, model)
                format_prediction(case["data"], prediction, path)
                print(f"Case Description: {case['description']}\n")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nTROUBLESHOOTING:")
        print("1. Pastikan file 'tree_output.h5' ada di folder yang sama dengan script ini")
        print(f"2. Current directory: {script_dir}")
        print(f"3. Looking for file at: {h5_path if h5_path else 'N/A'}")
        print("4. Jalankan notebook training untuk generate file model")
        sys.exit(1)
        
    except KeyError as e:
        print(f"ERROR: Invalid model format - missing attribute {str(e)}")
        print("Pastikan file model adalah output dari notebook training yang benar")
        sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()