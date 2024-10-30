import pickle
import numpy as np

y = list(pickle.load(open('./data/embedding_data/FakeNews_file_y.pkl', 'rb')))

new_y = []
for i in y:
    if int(i) >= 2:
        new_y.append(2)
    else:
        new_y.append(i)
new_y = np.array(new_y)
pickle.dump(new_y, open('./data/embedding_data/FakeNews_file_y.pkl', 'wb'))

