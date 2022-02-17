import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from keras.layers import Dense, LSTM, Dropout


output_size = 2         # sonraki 2 degeri tahmin etmeye calis
epochs = 3000            # islemi tekrar sayisi
features = 3           # geçmişe dönük kaç değeri kullanılacağımı belirttim..

def get_data(y):

    train_size = int(len(y)*0.8)                # Verinin %70 ini egitim icin kalaninin da test icin ayiracagiz, 180
# datayı train ve test diye ikiye ayırıyoruz.
    train = y[0:train_size]                                             #0'dan başlayıp train_size'a kadar
    test = y[train_size:len(y)]                                         # train_size'den datanın boyuna kadar

    ############## TRAIN DATA #################### Sistemi öğretmek için kullanacağız.

    train_x = []                                                     #Öncelikle nul alan belirliyoruz.
    train_y = []
    for i in range(0, train_size - features - output_size):         #0'dan başlıyoruz. baştan başlayıp sona doğru giderken ...
        tmp_x = y[i:(i+features)]                                   #input datasını aldı
        tmp_y = y[(i+features):(i+features+output_size)]            #geçmişteki 3 tane değer bir sonraki 2 değeri üretiyor kavramından yola çıkıyourz.
        train_x.append(np.reshape(tmp_x, (1, features)))            #tmp_x tek boyutlu alanı ters çevirerek 3 boyutlu alana çeviriyoruz.
        train_y.append(tmp_y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    ########### TEST DATA ######################## öğrenmedeki başarımı ölçmek için kullanacağım.
    test_x = []
    test_y = []
    last = len(y) - output_size - features                                           # son noktayı belirtiyor.
    for i in range(train_size, last):
        tmp_x = y[i:(i+features)]
        tmp_y = y[(i + features):(i + features + output_size)]
        test_x.append(np.reshape(tmp_x, (1, features)))
        test_y.append(tmp_y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    ######## Tahmin edilecek data #######################array
    data_x = []
    tmp_x = y[-features:len(y)]                                             # son noktadan geriye doğru gelmek istediğim için.
    data_x.append(np.reshape(tmp_x, (1, features)))
    data_x = np.array(data_x)

    return train_x, train_y, test_x, test_y, data_x


raw_data = pd.read_csv('./datasets/covid151.csv', header=None, names=["i", "t", "y"])
t = np.array(raw_data.t.values)
y = np.array(raw_data.y.values)

min = y.min()
max = y.max()
y = np.interp(y, (min, max), (-1, +1))

x_train, y_train, x_test, y_test, data_x = get_data(y)

model = Sequential()
model.add(LSTM(13, input_shape=(1,  features), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(1))
model.add(Dense(output_size))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=0)

score = model.evaluate(x_test, y_test)
print("%2s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.summary()

data_y = model.predict(data_x)

result = np.interp(data_y, (-1, +1), (min, max))

print("Gelecekteki Degerler (output_size) :", result)



#plt.plot(random_x,
#         lineer_regresyon.intercept_[0] +
  #       lineer_regresyon.coef_[0][0] * random_x,
   #      color='red',
    #     label='regresyon grafiği')