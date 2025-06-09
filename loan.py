# Fungsi untuk membaca file CSV dan memisahkan header dari data
def read_csv(file_path):
    file = open(file_path, "r")
    lines = file.readlines()
    file.close()
    # Pisahkan tiap baris menjadi list berdasarkan koma
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
            loan_id = row[0]  # Simpan ID untuk identifikasi

            # Encode semua fitur kategorikal
            gender = encode(row, 1, {"Male": 1, "Female": 0})
            married = encode(row, 2, {"Yes": 1, "No": 0})
            dependents = encode(row, 3, {"0": 0, "1": 1, "2": 2, "3+": 3})
            education = encode(row, 4, {"Graduate": 1, "Not Graduate": 0})
            self_emp = encode(row, 5, {"Yes": 1, "No": 0})

            # Konversi nilai numerik dan isi nilai kosong jika ada
            applicant_income = int(fill_missing(row, 6, "0"))
            coapplicant_income = float(fill_missing(row, 7, "0"))
            loan_amount = float(fill_missing(row, 8, "120"))  # Default = median estimasi
            loan_term = float(fill_missing(row, 9, "360"))    # Default = 360 bulan (30 tahun)
            credit_history = float(fill_missing(row, 10, "1")) # Default = good credit
            property_area = encode(row, 11, {"Urban": 2, "Semiurban": 1, "Rural": 0})

            # Encode target label
            loan_status = 1 if row[12] == "Y" else 0

            # Gabungkan semua fitur ke dalam list
            features = [
                gender, married, dependents, education, self_emp,
                applicant_income, coapplicant_income, loan_amount,
                loan_term, credit_history, property_area
            ]

            # Tambahkan tuple (id, fitur, label) ke list hasil
            processed.append((loan_id, features, loan_status))
        except:
            continue
    print(f"preprocess: processed {len(processed)} rows")
    return processed

# Membagi data menjadi data latih dan uji dengan rasio tertentu (default: 80/20)
def split_data(dataset, split_ratio=0.8):
    train_size = int(len(dataset) * split_ratio)
    train, test = dataset[:train_size], dataset[train_size:]
    print(f"split_data: train size = {len(train)}, test size = {len(test)}")
    return train, test

# Menentukan label mayoritas dalam dataset untuk prediksi sederhana
def majority_label(data):
    count = {0: 0, 1: 0}
    for _, _, label in data:
        count[label] += 1
    return 1 if count[1] >= count[0] else 0

# Evaluasi pembagian data berdasarkan fitur tertentu
# Mengembalikan akurasi, dan label prediksi masing-masing sisi (kiri & kanan)
def evaluate_split(data, feature_index):
    left, right = [], []
    for _, features, label in data:
        if features[feature_index] == 1:
            left.append((features, label))
        else:
            right.append((features, label))

    # Jika salah satu sisi kosong, pembagian tidak valid
    if len(left) == 0 or len(right) == 0:
        return 0, 0, 0

    # Cari prediksi terbaik (mayoritas) untuk masing-masing sisi
    left_pred = majority_label([(None, f, l) for f, l in left])
    right_pred = majority_label([(None, f, l) for f, l in right])

    # Hitung akurasi
    correct = 0
    for _, features, label in data:
        pred = left_pred if features[feature_index] == 1 else right_pred
        if pred == label:
            correct += 1

    accuracy = correct / len(data)
    return accuracy, left_pred, right_pred

# Load dan preprocessing data
header, raw_data = read_csv("loan_data.csv")
data = preprocess(raw_data, header)
train_data, test_data = split_data(data)

# Menentukan fitur terbaik sebagai root (akar pohon)
best_accuracy = 0
root_feature = -1
for i in range(11):  # Ada 11 fitur
    acc, _, _ = evaluate_split(train_data, i)
    if acc > best_accuracy:
        best_accuracy = acc
        root_feature = i

# Pisahkan data latih berdasarkan root feature
left_data, right_data = [], []
for id_, features, label in train_data:
    if features[root_feature] == 1:
        right_data.append((id_, features, label))
    else:
        left_data.append((id_, features, label))

# Fungsi untuk mencari fitur terbaik untuk anak cabang (level ke-2)
def find_best_feature(data_subset):
    best_fi = -1
    best_acc = 0
    best_lp = 0
    best_rp = 0
    for i in range(11):
        acc, lp, rp = evaluate_split(data_subset, i)
        if acc > best_acc:
            best_acc = acc
            best_fi = i
            best_lp = lp
            best_rp = rp
    return best_fi, best_lp, best_rp

# Cari fitur terbaik untuk sisi kiri dan kanan dari root
left_feature, left_1, left_0 = find_best_feature(left_data)
right_feature, right_1, right_0 = find_best_feature(right_data)

# Fungsi prediksi berdasarkan pohon keputusan 2 tingkat
def predict(features):
    if features[root_feature] == 1:
        return right_1 if features[right_feature] == 1 else right_0
    else:
        return left_1 if features[left_feature] == 1 else left_0

# Fungsi evaluasi model, juga menampilkan ID yang diterima/ditolak
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

    # Ringkasan hasil prediksi
    print("\n=== Prediction Summary ===")
    print(f"Jumlah ID yang layak diterima = {len(accepted_ids)}")
    print("ID yang diterima =", ", ".join(accepted_ids))
    print(f"Jumlah ID yang tidak layak diterima = {len(rejected_ids)}")
    print("ID yang ditolak =", ", ".join(rejected_ids))

    return accuracy

# Jalankan evaluasi pada data uji
acc_test = evaluate(test_data)

# Tampilkan akurasi model
print("\n=== Model Evaluation ===")
print("Decision Tree Depth-2 Accuracy:", round(acc_test * 100, 2), "%")

# Tampilkan struktur pohon keputusan yang dibuat
print("\n=== Decision Tree Structure ===")
print(f"Root Feature Index: {root_feature} -> '{header[root_feature+1]}'")
print(f"  If {header[root_feature+1]} == 1:")
print(f"    -> Check '{header[right_feature+1]}': 1 -> {right_1}, 0 -> {right_0}")
print(f"  Else:")
print(f"    -> Check '{header[left_feature+1]}': 1 -> {left_1}, 0 -> {left_0}")
