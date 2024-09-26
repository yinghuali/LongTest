from tensorflow.keras.models import load_model


path_model = './save_models/20new_lstm_6.h5'
path_data =

model = load_model('path_to_model/model.h5')

# 查看模型结构
model.summary()

# 使用模型进行预测
predictions = model.predict(data)