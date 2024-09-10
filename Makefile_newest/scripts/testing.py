import pickle
# Asumsikan data baru sudah disiapkan dalam format yang sesuai
data_baru = [[5.5, 3.2, 1.7, 0.3]]  # Data baru dalam format array

# Muat model yang sudah di-deploy
with open('models/deployed_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prediksi data baru
prediksi = model.predict(data_baru)
print(f'Hasil prediksi untuk data baru: {prediksi}')
