<a href="https://colab.research.google.com/github/ayrna/ap-ts-rnn/blob/main/redes_recurrentes/text_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## ¿Qué es esto?

Este cuaderno de Jupyter contiene código Python para construir una red recurrente LSTM  que proporciona alrededor de un 87-88% de precisión en el dataset *IMDB Movie Review Sentiment Analysis Dataset*.

Para más información podéis consultar este [enlace](https://www.bouvet.no/bouvet-deler/explaining-recurrent-neural-networks).

## Pensado para Google Collaboratory

El código está preparado para funcionar en Google Collab. Si queréis ejecutarlo de forma local, tendréis que configurar todos los elementos (Cuda, Tensorflow...).

En Google Collab, para conseguir que la red se entrene más rápido deberíamos usar la GPU. En el menú **Entorno de ejecución** elige **Cambiar tipo de entorno de ejecución** y selecciona "GPU".

No olvides hacer los cambios efectivos pulsando sobre **Reiniciar entorno de ejecución**.


## Preparándolo todo

Al ejecutar este código, puede ser que recibas un *warning* pidiéndote que reinicies el *Entorno de ejecución*. Puedes ignorarlo o reiniciarlo con "Entorno de ejecución -> Reiniciar entorno de ejecución" si encuentras algún tipo de problema.


```python
# Todos los import necesarios
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import sequence
from numpy import array

# Suprimir los warning de tensorflow
import logging
logging.getLogger('tensorflow').disabled = True

# Obtener los datos de "IMDB Movie Review", limitando las revisiones
# a las 10000 palabras más comunes
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

# Maping de las etiquetas
class_names = ["Negative", "Positive"]
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
    [1m17464789/17464789[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 0us/step


## Crear un mapeo que nos permita convertir el dataset IMDB a revisiones que podamos leer

Las revisiones en el dataset IMDB están codificadas como una secuencia de enteros. Afortunadamente, el dataset también contiene un índice que nos permite volver a una representación tipo texto.


```python
# Obtener el índice de palabras del dataset
word_index = tf.keras.datasets.imdb.get_word_index()

# Asegurarnos de que las palabras "especiales" pueden leerse correctamente
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNKNOWN>"] = 2
word_index["<UNUSED>"] = 3

# Buscar las palabras en el índice y hacer una función que decodifique cada review
# Si la palabra no está devolverá '?'
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
    [1m1641221/1641221[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1us/step


## Echemos un vistazo a los datos

Ahora vamos a ver más de cerca los datos. ¿Cuántas palabras contienen nuestras reviews?

¿Qué aspecto tiene una review codificada y decodificada?



```python
# Concatenar los datasets de entrenamiento y de test
allreviews = np.concatenate((x_train, x_test), axis=0)

# Longitud de las revisiones a largo de los dos conjuntos
result = [len(x) for x in allreviews]
print("Máxima longitud de una revisión: {}".format(np.max(result)))
print("Mínima longitud de una revisión: {}".format(np.min(result)))
print("Longitud media de las revisiones: {}".format(np.mean(result)))

# Imprimir una revisión concreta y su etiqueta.
# Reemplaza el número si quieres ver otra.
review_to_print=60
print("")
print("Revisión en modo máquina (codificada)")
print("  Código de la revisión: " + str(x_train[review_to_print]))
print("  Sentimiento: " + str(y_train[review_to_print]))
print("")
print("Revisión en modo texto")
print("  Texto de la revisión: " + decode_review(x_train[review_to_print]))
print("  Sentimiento: " + class_names[y_train[review_to_print]])

```

    Máxima longitud de una revisión: 2494
    Mínima longitud de una revisión: 7
    Longitud media de las revisiones: 234.75892
    
    Revisión en modo máquina (codificada)
      Código de la revisión: [1, 13, 219, 14, 33, 4, 2, 22, 1413, 12, 16, 373, 175, 2711, 1115, 1026, 430, 939, 16, 23, 2444, 25, 43, 697, 89, 12, 16, 170, 8, 130, 262, 19, 32, 4, 665, 7, 4, 2, 322, 5, 4, 1520, 7, 4, 86, 250, 10, 10, 4, 249, 173, 16, 4, 3891, 6, 19, 4, 167, 564, 5, 564, 1325, 36, 805, 8, 216, 638, 17, 2, 21, 25, 100, 376, 507, 4, 2110, 15, 79, 125, 23, 567, 13, 2134, 233, 36, 4852, 2, 5, 81, 1672, 10, 10, 92, 437, 129, 58, 13, 69, 8, 401, 61, 1432, 39, 1286, 46, 7, 12]
      Sentimiento: 0
    
    Revisión en modo texto
      Texto de la revisión: <START> i saw this at the <UNKNOWN> film festival it was awful every clichéd violent rich boy fantasy was on display you just knew how it was going to end especially with all the shots of the <UNKNOWN> wife and the rape of the first girl br br the worst part was the q a with the director writer and writer producer they tried to come across as <UNKNOWN> but you could tell they're the types that get off on violence i bet anything they frequent <UNKNOWN> and do drugs br br don't waste your time i had to keep my boyfriend from walking out of it
      Sentimiento: Negative


## Pre-procesando los datos

Tenemos que asegurarnos de que nuestras revisiones tienen siempre la misma
longitud, ya que se necesita para establecer los parámetros de la LSTM.

Para algunas revisiones tendremos que truncar algunas palabras, mientras que para otras habrá que establecer palabras de relleno (`padding`).


```python
# Longitud a la que vamos a dejar la ventana
review_length = 500

# Truncar o rellenar los conjuntos
x_train = sequence.pad_sequences(x_train, maxlen = review_length)
x_test = sequence.pad_sequences(x_test, maxlen = review_length)

# Comprobamos el tamaño de los conjuntos. Revisaremos los datos de entrenamiento
# y de test, comprobando que las 25000 revisiones tienen 500 enteros.
# Las etiquetas de clase deberían ser 25000, con valores de 0 o 1
print("Shape de los datos de entrenamiento: " + str(x_train.shape))
print("Shape de la etiqueta de entrenamiento " + str(y_train.shape))
print("Shape de los datos de test: " + str(x_test.shape))
print("Shape de la etiqueta de test: " + str(y_test.shape))

# Note padding is added to start of review, not the end
print("")
print("Texto de la revisión (post padding): " + decode_review(x_train[60]))
```

    Shape de los datos de entrenamiento: (25000, 500)
    Shape de la etiqueta de entrenamiento (25000,)
    Shape de los datos de test: (25000, 500)
    Shape de la etiqueta de test: (25000,)
    
    Texto de la revisión (post padding): <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <START> i saw this at the <UNKNOWN> film festival it was awful every clichéd violent rich boy fantasy was on display you just knew how it was going to end especially with all the shots of the <UNKNOWN> wife and the rape of the first girl br br the worst part was the q a with the director writer and writer producer they tried to come across as <UNKNOWN> but you could tell they're the types that get off on violence i bet anything they frequent <UNKNOWN> and do drugs br br don't waste your time i had to keep my boyfriend from walking out of it


## Crear y construir una red LSTM recurrente


```python
# Empezamos definiendo una pila vacía. Usaremos esta pila para ir construyendo
# la red, capa por capa
model = tf.keras.models.Sequential()

# La capa de tipo Embedding proporciona un mapeo (también llamado Word Embedding)
# para todas las palabras de nuestro conjunto de entrenamiento. En este embedding,
# las palabras que están cerca unas de otras comparten información de contexto
# y/o de significado. Esta transformación es aprendida durante el entrenamiento
model.add(
    tf.keras.layers.Embedding(
        input_dim = vocab_size, # Tamaño del vocabulario
        output_dim = 32, # Dimensionalidad del embedding
        input_length = review_length # Longitud de las secuencias de entrada
    )
)

# Las capas de tipo Dropout combaten el sobre aprendizaje y fuerzan a que el
# modelo aprenda múltiples representaciones de los mismos datos, ya que pone a
# cero de forma aleatoria algunas de las neuronas durante la fase de
# entrenamiento
model.add(
    tf.keras.layers.Dropout(
        rate=0.25 # Poner a cero aleatoriamente un 25% de las neuronas
    )
)

# Aquí es donde viene realmente la capa LSTM. Esta capa va a mirar cada
# secuencia de palabras de la revisión junto con sus embeddings y utilizará
# ambos elementos para determinar el sentimiento de la revisión
model.add(
    tf.keras.layers.LSTM(
        units=32 # La capa va a tener 32 neuronas de tipo LSTM
    )
)

# Añadir una segunda capa Dropout con el mismo objetivo que la primera
model.add(
    tf.keras.layers.Dropout(
        rate=0.25 # Poner a cero aleatoriamente un 25% de las neuronas
    )
)

# Todas las neuronas LSTM se conectan a un solo nodo en la capa densa. La
# función sigmoide determina la salida de este nodo, con un valor entre 0 y 1.
# Cuanto más cercano sea a 1, más positiva es la revisión
model.add(
    tf.keras.layers.Dense(
        units=1, # Una única salida
        activation='sigmoid' # Función de activación sigmoide
    )
)

# Compilar el modelo
model.compile(
    loss=tf.keras.losses.binary_crossentropy, # Entropía cruzada
    optimizer=tf.keras.optimizers.Adam(), # Optimizador Adam
    metrics=['accuracy']) # Métrica de los informes

# Mostrar un resumen de la estructura del modelo
model.summary()
```

    /usr/local/lib/python3.11/dist-packages/keras/src/layers/core/embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.
      warnings.warn(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)                │ ?                           │     <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ ?                           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                          │ ?                           │     <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ ?                           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ ?                           │     <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



## Entrenar la LSTM


```python
# Entrenar la LSTM en los datos de entrenamiento
history = model.fit(

    # Datos de entrenamiento : características (revisiones) y clases (positivas o negativas)
    x_train, y_train,

    # Número de ejemplos a examinar antes de actualizar los pesos en el
    # backpropagation. Cuanto más grande sea el tamaño de batch, más memoria
    # necesitaremos
    batch_size=256,

    # Una época hace tantos batches como sea necesario para agotar el conjunto
    # de entrenamiento
    epochs=3,

    # Esta fracción de los datos será usada como conjunto de validación, con
    # vistas a detener el algoritmo si se está produciendo sobre aprendizaje
    validation_split=0.2,

    verbose=1
)
```

    Epoch 1/3
    [1m79/79[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 29ms/step - accuracy: 0.5824 - loss: 0.6811 - val_accuracy: 0.7292 - val_loss: 0.5796
    Epoch 2/3
    [1m79/79[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 24ms/step - accuracy: 0.7970 - loss: 0.4778 - val_accuracy: 0.8560 - val_loss: 0.3378
    Epoch 3/3
    [1m79/79[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 30ms/step - accuracy: 0.9009 - loss: 0.2607 - val_accuracy: 0.8618 - val_loss: 0.3342


## Evaluar el modelo con los datos de test y ver el resultado


```python
# Obtener las predicciones para los datos de test
from sklearn.metrics import classification_report
predicted_probabilities = model.predict(x_test)
predicted_classes = predicted_probabilities  > 0.5
print(classification_report(y_test, predicted_classes, target_names=class_names))
tf.math.confusion_matrix(y_test, predicted_classes)
```

    [1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 7ms/step
                  precision    recall  f1-score   support
    
        Negative       0.80      0.94      0.87     12500
        Positive       0.93      0.76      0.84     12500
    
        accuracy                           0.85     25000
       macro avg       0.87      0.85      0.85     25000
    weighted avg       0.87      0.85      0.85     25000
    





    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[11802,   698],
           [ 2975,  9525]], dtype=int32)>



## Ver algunas predicciones incorrectas

Vamos a echar un vistazo a algunas de las revisiones incorrectamente clasificadas. Eliminaremos el `padding`.




```python
predicted_classes_reshaped = np.reshape(predicted_classes, 25000)

incorrect = np.nonzero(predicted_classes_reshaped!=y_test)[0]

# Nos vamos a centrar en las 20 primeras revisiones incorrectas
for j, incorrect in enumerate(incorrect[0:20]):

    predicted = class_names[predicted_classes_reshaped[incorrect].astype(int)]
    actual = class_names[y_test[incorrect]]
    human_readable_review = decode_review(x_test[incorrect])

    print("Revisión de test incorrectamente clasificada ["+ str(j+1) +"]")
    print("Revisión de test #" + str(incorrect)  + ": Predicho ["+ predicted + "] Objetivo ["+ actual + "]")
    print("Texto de la revisión: " + human_readable_review.replace("<PAD> ", ""))
    print("")
```

    Revisión de test incorrectamente clasificada [1]
    Revisión de test #8: Predicho [Positive] Objetivo [Negative]
    Texto de la revisión: <START> hollywood had a long love affair with bogus <UNKNOWN> nights tales but few of these products have stood the test of time the most memorable were the jon hall maria <UNKNOWN> films which have long since become camp this one is filled with dubbed songs <UNKNOWN> <UNKNOWN> and slapstick it's a truly crop of corn and pretty near <UNKNOWN> today it was nominated for its imaginative special effects which are almost <UNKNOWN> in this day and age <UNKNOWN> mainly of trick photography the only outstanding positive feature which survives is its beautiful color and clarity sad to say of the many films made in this genre few of them come up to alexander <UNKNOWN> original thief of <UNKNOWN> almost any other <UNKNOWN> nights film is superior to this one though it's a loser
    
    Revisión de test incorrectamente clasificada [2]
    Revisión de test #22: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> how managed to avoid attention remains a mystery a potent mix of comedy and crime this one takes chances where tarantino plays it safe with the hollywood formula the risks don't always pay off one character in one sequence comes off <UNKNOWN> silly and falls flat in the lead role thomas jane gives a wonderful and complex performance and two brief appearances by mickey rourke hint at the high potential of this much under and <UNKNOWN> used actor here's a director one should keep one's eye on
    
    Revisión de test incorrectamente clasificada [3]
    Revisión de test #32: Predicho [Positive] Objetivo [Negative]
    Texto de la revisión: <START> if you have never read the classic science fiction novel this mini series is based on it may actually be good unfortunately if you are a fan of the book you probably won't be able to watch more than the first hour or two all of the political intrigue has been taken out of the film the most important scenes from the book have been taken out characters motivations have been changed completely and words from the wrong characters mouths where in the novel paul was a teen age boy with incredible political skill and a great understanding of the way the world worked in this film he is hot headed and and frustrated avoid this movie at all costs
    
    Revisión de test incorrectamente clasificada [4]
    Revisión de test #43: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> i saw this 1997 movie because i am a fan of <UNKNOWN> lamas and of his father the late <UNKNOWN> lamas in my opinion <UNKNOWN> looked his best in this film mostly due to his <UNKNOWN> and the <UNKNOWN> wardrobe that were <UNKNOWN> to him br br as the plot progressed i realized the movie was more than just entertainment or a reason to see a favorite actor the story was about a ring of serial killers and the attempts of law <UNKNOWN> to investigate the ring and bring the members to justice there was adequate suspense and i believe the violence was necessary to relate the story to the viewer br br at the end of the film i was shocked to learn the film is the true account of horrendous murders that occurred in <UNKNOWN> furthermore <UNKNOWN> and his leading lady were portraying actual fbi agents who solved the <UNKNOWN> of many young women and contributed to the <UNKNOWN> of the ring i believe the film is worthwhile as it informs the public about the dangers and <UNKNOWN> of the criminal element
    
    Revisión de test incorrectamente clasificada [5]
    Revisión de test #45: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> the above profile was written by me when i used the nick of which is still my email address i still believe andy <UNKNOWN> character of <UNKNOWN> is the best twilight episodes ever and i watch this episode at least once a year as i consider to be a fortunate man as he has many friends who love him <UNKNOWN> br br in case many of you are too young to remember i'm <UNKNOWN> andy <UNKNOWN> <UNKNOWN> a children's entertainment show in the 50's i believe called <UNKNOWN> gang on it he had three <UNKNOWN> a cat named midnight who played the <UNKNOWN> a mouse named <UNKNOWN> who played an a hand <UNKNOWN> and a <UNKNOWN> <UNKNOWN> named who's could appear and disappear at will embarrassing many of <UNKNOWN> funny guest stars like billy gilbert
    
    Revisión de test incorrectamente clasificada [6]
    Revisión de test #55: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> this is a special film if you know the context antonioni in his eighties had been crippled by a stroke mute and half <UNKNOWN> his friends who incidentally are the best the film world has arranged for him to a last significant film the idea is that he can <UNKNOWN> a story into being by just looking at it so we have a film about a director who <UNKNOWN> stories by simple observation and the matter of the four stories is about how the visual imagination <UNKNOWN> love br br the film emerges by giving us the tools to bring it into being through our own imagination the result is pure movie world every person except the director is lovely in aspect or movement some of these women are and they exist in a dreamy misty world of sensual encounter there is no nuance no hint that anything exists but what we see no desire is at work other than what we create br br i know of no other film that so successfully <UNKNOWN> our own visual yearning to have us create the world we see he understands something about not touching no one understands van <UNKNOWN> visually like he does <UNKNOWN> <UNKNOWN> space music is <UNKNOWN> on precisely the same notion the sensual touch that implies but doesn't physically touch br br <UNKNOWN> <UNKNOWN> wife appears appropriately as the <UNKNOWN> and she also directs a lackluster <UNKNOWN> <UNKNOWN> film that is on the dvd br br ted's <UNKNOWN> 3 of 4 worth watching
    
    Revisión de test incorrectamente clasificada [7]
    Revisión de test #56: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> i was very disappointed when this show was canceled although i can not vote i live on the island of i sat down to see the show on <UNKNOWN> and was very surprised that it didn't aired the next day i read on the internet that it was canceled br br it's true not every one was as much talented as the other but there were very talented people singing br br i find it very sad for them br br that they worked so hard and there dreams came <UNKNOWN> down br br its a pity br br
    
    Revisión de test incorrectamente clasificada [8]
    Revisión de test #72: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> alice is the kind of movie they made in the 30's and 40's never attempts to be an event just wants to entertain and it does i was surprised by <UNKNOWN> sutherland in a role that could be a cliche he made it real the plot does make <UNKNOWN> to alice in wonderland a guy dressed in white does go through a hole and <UNKNOWN> does fall down one like alice the plot does twist and turn but with a <UNKNOWN> you don't see in small movies i loved the direction sutherland just a very fast paced and interesting movie
    
    Revisión de test incorrectamente clasificada [9]
    Revisión de test #80: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> with the obvious exception of fools horses this was in my opinion david <UNKNOWN> finest series br br coming straight after his tv debut on <UNKNOWN> not <UNKNOWN> your set ' these 13 episodes revealed a <UNKNOWN> of comic timing not seen since the old silent movie days by comparison <UNKNOWN> open all hours and that awful series <UNKNOWN> man' did not come close br br i believe jason banned the series being repeated because it showed him at his <UNKNOWN> shame on him a new generation deserves to enjoy this the series actually <UNKNOWN> in the ratings but that is most likely because it was shown against 'the <UNKNOWN> which aired on bbc at the same time before <UNKNOWN> were <UNKNOWN> br br btw i have only just noticed that his long suffering assistant spencer was played by mark <UNKNOWN> alan <UNKNOWN> off <UNKNOWN> street i am amazed he didn't try to murder edgar <UNKNOWN>
    
    Revisión de test incorrectamente clasificada [10]
    Revisión de test #83: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> i'm not into reality shows that much but this one is exceptional because at the end the viewer gets something useful out of it besides entertainment i don't have children but lessons will be a great help when and if i ever do my only complaint is that the show has been watered down since it has been in the us i prefer the british version with the <UNKNOWN> <UNKNOWN> approach is it just me or does anyone else find the us version to be a bit soft now it's the <UNKNOWN> <UNKNOWN> <UNKNOWN> i guess we <UNKNOWN> need a <UNKNOWN> of sugar to help the medicine go down after all still nothing wrong with a sappy happy ending is there
    
    Revisión de test incorrectamente clasificada [11]
    Revisión de test #91: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> i think this is a great classic monster film for the family the mole what a machine the tall creature with the <UNKNOWN> the flying green <UNKNOWN> or whatever they are and the ape men things the speak <UNKNOWN> with them the battle of the men in rubber suits fighting for a doll for breakfast <UNKNOWN> <UNKNOWN> class what else can i say how would they make a 2002 remake of this one
    
    Revisión de test incorrectamente clasificada [12]
    Revisión de test #100: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> a quick glance at the premise of this film would seem to indicate just another dumb <UNKNOWN> <UNKNOWN> <UNKNOWN> slash fest the type where sex equals death and the actors are all annoying stereotypes you actually want to die however delivers considerably more br br rather than focus on bare flesh and gore though there is a little of each no sex however the flick focuses on delivering impending dread <UNKNOWN> tension amidst a lovely <UNKNOWN> backdrop these feelings are further <UNKNOWN> by a cast of realistically likable characters and <UNKNOWN> that are more amoral than cardboard <UNKNOWN> of evil oh yeah george kennedy is here too and when is that not a good thing br br if you liked wrong turn then watch this to see where much of its' <UNKNOWN> came from
    
    Revisión de test incorrectamente clasificada [13]
    Revisión de test #101: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> <UNKNOWN> is the first of its kind in turkish cinema and it's way better than i expected those people who say it's neither scary nor funny have a point it's not all that great indeed but it must be kept in mind that everyone involved with the movie is rather amateur so it's basically a maiden voyage and comparing this one to other films such as the 1st class garbage propaganda this movie is pretty damn good br br one thing that must be said it deals with the <UNKNOWN> <UNKNOWN> life in turkey very realistically that's exactly how it goes the scenes that are meant to scare are somewhat cheap and <UNKNOWN> most of them even if not all but that religion lesson scene made me laugh in tears and performs the best acting of this flick as a religion teacher br br it's not a waste of your time go and watch it you'll find it rather amusing especially if you know turkey enough to relate to turkish school lives
    
    Revisión de test incorrectamente clasificada [14]
    Revisión de test #125: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> ok so in any <UNKNOWN> e <UNKNOWN> road runner cartoons we know that is going to set up all sorts of traps for <UNKNOWN> but always himself in various ways that certainly happens in <UNKNOWN> <UNKNOWN> predictable i guess that it is but when you think about it these cartoons show how the more you try to harm someone else the more you get <UNKNOWN> sort of like how <UNKNOWN> duck always tries to <UNKNOWN> bugs <UNKNOWN> integrity but bugs sees around it br br overall this is another classic from the <UNKNOWN> <UNKNOWN> crowd sometimes i think that if we really had wanted to ease cold war tensions we could have just let the soviet union see looney tunes cartoons i'm sure that they would have loved them another great one br br ps i learned on <UNKNOWN> that <UNKNOWN> e middle name is
    
    Revisión de test incorrectamente clasificada [15]
    Revisión de test #126: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> this was answer to fatal attraction and this is a classic film in its own right <UNKNOWN> <UNKNOWN> was good and so was sunny <UNKNOWN> but it was shah rukh khan who shot to fame as the stalker since then he has become a favourite of the <UNKNOWN> <UNKNOWN> <UNKNOWN> le <UNKNOWN> <UNKNOWN> to <UNKNOWN> <UNKNOWN> <UNKNOWN> <UNKNOWN> and de india shah rukh at first appears to be a villain but then towards the end you start to sympathize with him the scripting was superb and the songs were my favourites are too mere and <UNKNOWN> br br after the dismal failure of the underrated <UNKNOWN> <UNKNOWN> fought back with <UNKNOWN> the dialogues were memorable the k k <UNKNOWN> dialogue is often repeated since <UNKNOWN> <UNKNOWN> <UNKNOWN> has slipped bit <UNKNOWN> to <UNKNOWN> <UNKNOWN> was bad but he redeemed himself slightly with <UNKNOWN> <UNKNOWN> which was far far better this was <UNKNOWN> <UNKNOWN> last masterpiece
    
    Revisión de test incorrectamente clasificada [16]
    Revisión de test #130: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> i just saw this movie at the berlin film <UNKNOWN> children's program and it just killed me and pretty much everyone else in the audience and make no mistake about it this film belongs into the all time 250 let me tell you that i'm in no way associated with the creators of this film if that's what you come to believe reading this no but this actually is it <UNKNOWN> the kid's film label on it <UNKNOWN> girl is on almost every account a classic as in biblical the story concerns 12 year old <UNKNOWN> <UNKNOWN> julie who is <UNKNOWN> to learn of her <UNKNOWN> <UNKNOWN> illness special surgery in the us would cost 1 5 million and of course nobody could afford that so <UNKNOWN> and her friends <UNKNOWN> and sebastian do what every good kid would and <UNKNOWN> a bank sounds corny don't forget this is not america and is by no means the tear <UNKNOWN> robin williams <UNKNOWN> <UNKNOWN> du <UNKNOWN> nobody takes seriously anyway director <UNKNOWN> set out to make a big budget action comedy for kids and boy did he succeed let me put it this way this film rocks like no kid film and few others did before and there's a whole lot more to it than just the action after about 20 minutes of by the numbers exposition well granted it into a monster that br br effortlessly puts mission impossible to shame the numerous action sequences are masterfully staged and look real expensive take that mummy br br <UNKNOWN> almost every other movie suspense wise no easy they're only kids antics here br br easily <UNKNOWN> a dense story with enough laughs to make jim carrey look for career <UNKNOWN> br br <UNKNOWN> to both damon <UNKNOWN> and karate kid within the same seconds br br comes up with so much wicked humor that side of p c that i can hear the american ratings board wet their pants from over here br br manages to actually be tender and serious and sexy at the same time what am i saying they're kids they're kids <UNKNOWN> watch that last scene br br stars <UNKNOWN> anderson who since last years is everybody's favourite kid actor br br what a ride
    
    Revisión de test incorrectamente clasificada [17]
    Revisión de test #134: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> the <UNKNOWN> is a very worthwhile addition to recent re releases from the soviet union director 1977 film <UNKNOWN> the moral <UNKNOWN> of <UNKNOWN> and honor in german occupied russia during the second world war on foot and in a <UNKNOWN> two members of a <UNKNOWN> soviet group leave to locate supplies and are captured by nazi soldiers the focus of the movie is on how each man handles or <UNKNOWN> his moral <UNKNOWN> one chooses dignity and integrity while the other <UNKNOWN> for collaboration with the enemy however in the end he cannot <UNKNOWN> by his selfish decision the film makes much use of slow wide angle pans which shift to extreme closeups and highlight the spiritual <UNKNOWN> within the souls of each man this is not a great film but it does effectively portray an intense moral dilemma against the backdrop of a harsh and <UNKNOWN> soviet wilderness
    
    Revisión de test incorrectamente clasificada [18]
    Revisión de test #145: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> rugged david <UNKNOWN> solid doug <UNKNOWN> and dr <UNKNOWN> perry a delightfully <UNKNOWN> peter cushing <UNKNOWN> their way into the <UNKNOWN> core in their <UNKNOWN> mole machine the duo discover an ancient <UNKNOWN> world populated by dangerous gigantic <UNKNOWN> and human beings who are used as both food and slaves by evil <UNKNOWN> <UNKNOWN> men director kevin <UNKNOWN> working from a <UNKNOWN> silly script by <UNKNOWN> <UNKNOWN> maintains a constant <UNKNOWN> pace throughout and treats the exceptionally foolish premise with astonishing seriousness thereby giving this picture a certain <UNKNOWN> earnest quality that's amusing and endearing in equal measure the <UNKNOWN> hokey not so special effects are quite unintentionally funny the cheesy array of cut price creatures in particular are positively <UNKNOWN> <UNKNOWN> guys in obvious shoddy rubber suits <UNKNOWN> men equally <UNKNOWN> <UNKNOWN> savage ape man <UNKNOWN> and a hilariously ludicrous fire <UNKNOWN> frog thing who blows up real good one gut <UNKNOWN> highlight occurs when <UNKNOWN> mixes it up with a fat and clumsy giant <UNKNOWN> another priceless scene depicts a dinosaur <UNKNOWN> a doll in its <UNKNOWN> jaws moreover we also get some rousing <UNKNOWN> <UNKNOWN> and an exciting climactic slave revolt it's a total treat to see cushing <UNKNOWN> ham it up in a rare broad comedic part and become an unlikely but enthusiastic arrow <UNKNOWN> action hero in the last third of the flick the ever <UNKNOWN> caroline <UNKNOWN> looks positively <UNKNOWN> as the <UNKNOWN> princess <UNKNOWN> plus there are nice supporting turns by grant as <UNKNOWN> warrior <UNKNOWN> and sean lynch as <UNKNOWN> coward mike neatly varied score <UNKNOWN> between <UNKNOWN> <UNKNOWN> music and <UNKNOWN> <UNKNOWN> <UNKNOWN> stuff alan <UNKNOWN> crisp cinematography adds a glossy sheen to the <UNKNOWN> inane proceedings a complete campy riot
    
    Revisión de test incorrectamente clasificada [19]
    Revisión de test #146: Predicho [Negative] Objetivo [Positive]
    Texto de la revisión: <START> i was shocked to learn that jimmy <UNKNOWN> has left this show does anyone know why i regard james as one of the all time <UNKNOWN> and wasn't surprised he ended up on tv which can be better than the crap you see on the big screen the stories are slick and the camera faster than a <UNKNOWN> bullet <UNKNOWN> forget the rest of the cast james vanessa <UNKNOWN> <UNKNOWN> molly josh mitch also can anyone tell me why on earth there's a crap theme tune on the dvd sets but <UNKNOWN> <UNKNOWN> of a little less conversation is used on the initial nbc <UNKNOWN> does it not make sense to use a tune that you would associate with the gambling <UNKNOWN> of america for dvd releases
    
    Revisión de test incorrectamente clasificada [20]
    Revisión de test #147: Predicho [Positive] Objetivo [Negative]
    Texto de la revisión: <START> having enjoyed neil <UNKNOWN> writing especially his collaboration with <UNKNOWN> in the dream hunters in the past i figured <UNKNOWN> to be a sure thing and was very disappointed the beginning live action section of the movie was intriguing enough the relationships between the characters was believable and easy to empathize with and i loved the sets the <UNKNOWN> and <UNKNOWN> artwork the subsequent computer generated scenes however were excruciating the dialogue was awkward and pretentious the interaction between the live actors and the cgi horrifying events occurred for the <UNKNOWN> reasons and most events seemed superfluous to whatever plot may have existed i only watched the first twenty or thirty minutes of the movie so i'm not exactly an authority but i strongly recommend that you don't watch any of it at all and stick with <UNKNOWN> strong written work
    


## Prueba tu propio texto como conjunto de test

Esta es una forma divertida de comprobar los límites del modelo que hemos entrenado. Debes teclear todo en minúscula y no usar signos de puntuación.

Podrás comprobar la predicción del modelo, un valor entre 0 y 1



```python
# Escribe tu propia revisión (EN INGLÉS)
#review = "this was a terrible film with too much sex and violence i walked out halfway through"
review = "this is the best film i have ever seen it is great and fantastic and i loved it"
#review = "this was an awful film that i will never see again"
#review = "absolutely wonderful movie i am sure i will repeat it really worth the money i am delighted with the actors and i think it is epic the best movie from this director it is really fantastic and super"

# Codificamos la revisión (reemplazamos las palabras por los enteros)
tmp = []
for word in review.split(" "):
    tmp.append(word_index[word])

# Nos aseguramos que la longitud de secuencia es 500
tmp_padded = sequence.pad_sequences([tmp], maxlen=review_length)

# Introducimos la revisión ya procesada en el modelo
rawprediction = model.predict(array([tmp_padded][0]))[0][0]
prediction = int(round(rawprediction))

# Probamos el modelo y vemos los resultados
print("Revisión: " + review)
print("Predicción numérica: " + str(rawprediction))
print("Clase predicha: " + class_names[prediction])
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step
    Revisión: this is the best film i have ever seen it is great and fantastic and i loved it
    Predicción numérica: 0.94914544
    Clase predicha: Positive


## Redes GRU

Ahora vamos a repetir el entrenamiento pero con las redes GRU. Recuerda que las redes Gated Recurrent Unit (GRU) implementan una simplificación de la neurona LSTM basada en reducir el número de puertas, parámetros y estaddos de la misma.


```python
# Pila vacía
model = tf.keras.models.Sequential()

# Embedding
model.add(
    tf.keras.layers.Embedding(
        input_dim = vocab_size, # Tamaño del vocabulario
        output_dim = 32, # Dimensionalidad del embedding
        input_length = review_length # Longitud de las secuencias de entrada
    )
)

# Primer Dropout
model.add(
    tf.keras.layers.Dropout(
        rate=0.25 # Poner a cero aleatoriamente un 25% de las neuronas
    )
)

# Capa GRU
model.add(
    tf.keras.layers.GRU(
        units=32 # La capa va a tener 32 neuronas de tipo LSTM
    )
)

# Segundo Dropout
model.add(
    tf.keras.layers.Dropout(
        rate=0.25 # Poner a cero aleatoriamente un 25% de las neuronas
    )
)

# Capa densa final
model.add(
    tf.keras.layers.Dense(
        units=1, # Una única salida
        activation='sigmoid' # Función de activación sigmoide
    )
)

# Compilar el modelo
model.compile(
    loss=tf.keras.losses.binary_crossentropy, # Entropía cruzada
    optimizer=tf.keras.optimizers.Adam(), # Optimizador Adam
    metrics=['accuracy']) # Métrica de los informes

# Mostrar un resumen de la estructura del modelo
model.summary()
```

    /usr/local/lib/python3.11/dist-packages/keras/src/layers/core/embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.
      warnings.warn(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)              │ ?                           │     <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ ?                           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ gru (<span style="color: #0087ff; text-decoration-color: #0087ff">GRU</span>)                            │ ?                           │     <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ ?                           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ ?                           │     <span style="color: #00af00; text-decoration-color: #00af00">0</span> (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
# Entrenar la GRU en los datos
history = model.fit(

    # Datos de entrenamiento
    x_train, y_train,

    # Tamaño de batch
    batch_size=256,

    # Número de épocas
    epochs=3,

    # Porcentaje de validación
    validation_split=0.2,

    verbose=1
)
```

    Epoch 1/3
    [1m79/79[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 29ms/step - accuracy: 0.5612 - loss: 0.6818 - val_accuracy: 0.7482 - val_loss: 0.5156
    Epoch 2/3
    [1m79/79[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 23ms/step - accuracy: 0.8211 - loss: 0.4108 - val_accuracy: 0.8396 - val_loss: 0.3655
    Epoch 3/3
    [1m79/79[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 23ms/step - accuracy: 0.8969 - loss: 0.2641 - val_accuracy: 0.8600 - val_loss: 0.3267



```python
# Obtener las predicciones para los datos de test
from sklearn.metrics import classification_report

predicted_probabilities = model.predict(x_test)
predicted_classes = predicted_probabilities  > 0.5
print(classification_report(y_test, predicted_classes, target_names=class_names))
tf.math.confusion_matrix(y_test, predicted_classes)
```

    [1m782/782[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 7ms/step
                  precision    recall  f1-score   support
    
        Negative       0.87      0.85      0.86     12500
        Positive       0.85      0.87      0.86     12500
    
        accuracy                           0.86     25000
       macro avg       0.86      0.86      0.86     25000
    weighted avg       0.86      0.86      0.86     25000
    





    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[10619,  1881],
           [ 1607, 10893]], dtype=int32)>



## Referencia

Este material ha sido elaborado a partir del [cuaderno](https://github.com/markwest1972/LSTM-Example-Google-Colaboratory) de Mark West
