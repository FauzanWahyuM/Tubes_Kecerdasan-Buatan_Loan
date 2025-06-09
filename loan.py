# Fungsi untuk membaca file CSV dan memisahkan header dari data
def read_csv(file_path):
    file = open(file_path, "r")
    lines = file.readlines()
    file.close()
    data = [line.strip().split(",") for line in lines]
    print(f"read_csv: header = {data[0]}, number of rows = {len(data)-1}")
    return data[0], data[1:]

# Fungsi untuk mengisi nilai kosong dengan default value
def fill_missing(row, index, default):
    return row[index] if row[index] != "" else default

# Fungsi untuk mengubah nilai string (kategorikal) menjadi angka berdasarkan mapping
def encode(row, col_idx, mapping):
    return mapping.get(row[col_idx], 0)

# Fungsi utama untuk preprocessing data mentah menjadi data numerik siap proses
def preprocess(data, header):
    processed = []
    for row in data:
        try:
            loan_id = row[0]

            married = encode(row, 2, {"Yes": 1, "No": 0})
            dependents = encode(row, 3, {"0": 0, "1": 1, "2": 2, "3+": 3})
            education = encode(row, 4, {"Graduate": 1, "Not Graduate": 0})
            self_emp = encode(row, 5, {"Yes": 1, "No": 0})

            applicant_income = int(fill_missing(row, 6, "0"))
            coapplicant_income = float(fill_missing(row, 7, "0"))
            loan_amount = float(fill_missing(row, 8, "120"))
            loan_term = float(fill_missing(row, 9, "360"))
            credit_history = float(fill_missing(row, 10, "1"))
            property_area = encode(row, 11, {"Urban": 2, "Semiurban": 1, "Rural": 0})

            income_high = 1 if applicant_income >= 7000 else 0

            loan_status = 1 if row[12] == "Y" else 0

            features = [
                married, dependents, education, self_emp,
                income_high, coapplicant_income, loan_amount,
                loan_term, credit_history, property_area
            ]

            processed.append((loan_id, features, loan_status))
        except:
            continue
    print(f"preprocess: processed {len(processed)} rows")
    return processed

# Membagi data menjadi data latih dan uji
def split_data(dataset, split_ratio=0.8):
    train_size = int(len(dataset) * split_ratio)
    train, test = dataset[:train_size], dataset[train_size:]
    print(f"split_data: train size = {len(train)}, test size = {len(test)}")
    return train, test

# Menentukan label mayoritas
def majority_label(data):
    if not data:
        return 0
    total = len(data)
    ones = sum(1 for _, _, label in data if label == 1)
    return 1 if ones / total >= 0.5 else 0

# Evaluasi model dan tampilkan ID
def evaluate(dataset):
    correct = 0
    accepted_ids = []
    rejected_ids = []

    for id_, features, label in dataset:
        pred = predict(features)
        if pred == label:
            correct += 1
        if pred == 1:
            accepted_ids.append(id_)
        else:
            rejected_ids.append(id_)

    accuracy = correct / len(dataset)

    print("\n=== Prediction Summary ===")
    print(f"Jumlah ID yang layak diterima = {len(accepted_ids)}")
    print("ID yang diterima =", ", ".join(accepted_ids))
    print(f"Jumlah ID yang tidak layak diterima = {len(rejected_ids)}")
    print("ID yang ditolak =", ", ".join(rejected_ids))

    return accuracy

# Load dan preprocessing data
header, raw_data = read_csv("loan_data.csv")
data = preprocess(raw_data, header)
train_data, test_data = split_data(data)

# Gunakan fitur tetap: Credit_History (index 8), dan Property_Area (index 9)
root_feature = 8
second_feature = 4

# Pisahkan berdasarkan root
left_data, right_data = [], []
for id_, features, label in train_data:
    if features[root_feature] == 1:
        right_data.append((id_, features, label))
    else:
        left_data.append((id_, features, label))

# Ambil label mayoritas berdasarkan second_feature
def get_branch_labels(data_subset):
    left_branch, right_branch = [], []
    for id_, features, label in data_subset:
        if features[second_feature] == 1:
            right_branch.append((None, features, label))
        else:
            left_branch.append((None, features, label))
    label_1 = majority_label(right_branch)
    label_0 = majority_label(left_branch)
    return label_1, label_0

right_1, right_0 = get_branch_labels(right_data)
left_1, left_0 = get_branch_labels(left_data)

# Fungsi prediksi dengan dua level pohon
def predict(features):
    if features[root_feature] == 1:
        return right_1 if features[second_feature] == 1 else right_0
    else:
        return left_1 if features[second_feature] == 1 else left_0

# Evaluasi dan cetak
acc_test = evaluate(test_data)

print("\n=== Model Evaluation ===")
print("Decision Tree Depth-2 Accuracy:", round(acc_test * 100, 2), "%")

print("\n=== Decision Tree Structure ===")
print(f"Root Feature Index: {root_feature} -> 'Credit_History'")
print(f"  If Credit_History == 1:")
print(f"    -> Check 'income_high' (>=7000): 1 -> {right_1}, else -> {right_0}")  # Ubah deskripsi
print(f"  Else:")
print(f"    -> Check 'income_high' (>=7000): 1 -> {left_1}, else -> {left_0}")  # Ubah deskripsi

print("\nJumlah data tiap cabang pohon:")
print(f"right_data (CH==1): {len(right_data)}")
print(f"  right_branch (income_high==1): {sum(1 for _, f, _ in right_data if f[second_feature] == 1)}")  # Ubah label
print(f"  left_branch  (income_high==0): {sum(1 for _, f, _ in right_data if f[second_feature] != 1)}")  # Ubah label
print(f"left_data (CH!=1): {len(left_data)}")
print(f"  right_branch (income_high==1): {sum(1 for _, f, _ in left_data if f[second_feature] == 1)}")  # Ubah label
print(f"  left_branch  (income_high==0): {sum(1 for _, f, _ in left_data if f[second_feature] != 1)}")  # Ubah label
