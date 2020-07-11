import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sympy.solvers import solve
from sympy import Symbol
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

features, _ = Image.open('/content/vel_merged/trainingangvel/trainangvel/a1/a1_s1_t1_skeleton.jpg').size

traindirang = '/content/metalearner_merged/trainangle/trainangle'
valdirang = '/content/metalearner_merged/valangle'

traindirdist = '/content/metalearner_merged/traindist/traindist'
valdirdist = '/content/metalearner_merged/valdist'

traindirangvel = '/content/vel_merged/trainingangvel/trainangvel'
valdirangvel = '/content/vel_merged/valangvel'

traindirdistvel = '/content/vel_merged/trainingdistvel/traindistvel'
valdirdistvel = '/content/vel_merged/valdistvel'

train_datagen_dist = ImageDataGenerator()
val_datagen_dist = ImageDataGenerator()

training_set_dist= train_datagen_dist.flow_from_directory(
    traindirdist,
    target_size=(70, 190),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    seed = 42
   )

val_set_dist= val_datagen_dist.flow_from_directory(
    valdirdist,
    target_size=(70, 190),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

train_datagen_distvel = ImageDataGenerator()
val_datagen_distvel = ImageDataGenerator()

training_set_distvel= train_datagen_distvel.flow_from_directory(
    traindirdistvel,
    target_size=(65, 190),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    seed = 42
   )

val_set_distvel= val_datagen_distvel.flow_from_directory(
    valdirdistvel,
    target_size=(65, 190),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

train_datagen_ang = ImageDataGenerator()
val_datagen_ang = ImageDataGenerator()

training_set_ang= train_datagen_ang.flow_from_directory(
    traindirang,
    target_size=(70, features),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    seed = 42
   )

val_set_ang= val_datagen_ang.flow_from_directory(
    valdirang,
    target_size=(70, features),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

train_datagen_angvel = ImageDataGenerator()
val_datagen_angvel = ImageDataGenerator()

training_set_angvel= train_datagen_angvel.flow_from_directory(
    traindirangvel,
    target_size=(65, features),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    seed = 42
   )

val_set_angvel= val_datagen_angvel.flow_from_directory(
    valdirangvel,
    target_size=(65, features),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

distmodel = load_model('modeldist.h5')
distvelmodel = load_model('modeldistvel.h5')
angmodel = load_model('modelang.h5')
angvelmodel = load_model('modelangvel.h5')

Ypred_val_dist = distmodel.predict_generator(val_set_dist, verbose=1)
Y_val_dist = val_set_dist.labels

Ypred_val_distvel = distvelmodel.predict_generator(val_set_distvel, verbose=1)
Y_val_distvel = val_set_distvel.labels

Ypred_val_ang = angmodel.predict_generator(val_set_ang, verbose=1)
Y_val_ang = val_set_ang.labels

Ypred_val_angvel = angvelmodel.predict_generator(val_set_angvel, verbose=1)
Y_val_angvel = val_set_angvel.labels

fuzzyMeasures = np.array([ 0.04 , 0.03, 0.06, 0.02 ])

l = Symbol('l', real = True)
lam = solve(  ( 1 + l* fuzzyMeasures[0]) * ( 1 + l* fuzzyMeasures[1]) *( 1 + l* fuzzyMeasures[2]) *( 1 + l* fuzzyMeasures[3]) - (l+1), l )
lam = lam[1]

Ypred_fuzzy = np.zeros(shape = Ypred_val_ang.shape, dtype = float)

for sample in range(0,Ypred_val_angvel.shape[0]):
  for classes in range(0,27):
    
    scores = np.array([ Ypred_val_dist[sample][classes],Ypred_val_distvel[sample][classes],Ypred_val_ang[sample][classes],Ypred_val_angvel[sample][classes] ])
    permutedidx = np.flip(np.argsort(scores))
    scoreslambda = scores[permutedidx]
    fmlambda = fuzzyMeasures[permutedidx]

    ge_prev = fmlambda[0]
    fuzzyprediction = scoreslambda[0] * fmlambda[0]

    for i in range(1,3):
      ge_curr = ge_prev + fmlambda[i] + lam * fmlambda[i] * ge_prev
      fuzzyprediction = fuzzyprediction + scoreslambda[i] *(ge_curr - ge_prev)
      ge_prev = ge_curr

    fuzzyprediction = fuzzyprediction + scoreslambda[3] * ( 1 - ge_prev)

    Ypred_fuzzy[sample][classes] = fuzzyprediction

ypred_fuzzy = np.argmax(Ypred_fuzzy, axis=1)

mylabels = []

actions = ['a1', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19'\
           ,'a2', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a3', 'a4', 'a5', 
           'a6', 'a7', 'a8', 'a9']

mapping = {
    'a1': 'Swipe\n Left',
    'a2': 'Swipe\n Right',
    'a3': 'Wave',
    'a4': 'Clap',
    'a5': 'Throw',
    'a6': 'Arm\n Cross',
    'a7': 'Basketball\n Shoot',
    'a8': 'Draw X',
    'a9': 'Draw Circle\n CW',
    'a10': 'Draw Circle\n ACW',
    'a11': 'Draw triangvelle',
    'a12': 'Bowling',
    'a13': 'Boxing',
    'a14': 'Baseball\n Swing',
    'a15': 'Tennis Swing',
    'a16': 'Arm Curl',
    'a17': 'Tennis Serve',
    'a18': 'Push',
    'a19': 'Knock',
    'a20': 'Catch',
    'a21': 'Pickup &\n Throw',
    'a22': 'Jog',
    'a23': 'Walk',
    'a24': 'Sit to\n Stand',
    'a25': 'Stand to\n Sit',
    'a26': 'Lunge',
    'a27': 'Squat'
}


for l in actions:
  mylabels.append( mapping[l])


matrix = confusion_matrix(val_set_angvel.classes, ypred_fuzzy,labels=None)

fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                cmap = 'viridis',
                                show_absolute=True,
                                show_normed=False,
                                figsize = (12,12),
                                class_names = mylabels
                                )
plt.savefig('fuzzyfusion.png')

print(classification_report(val_set_angvel.classes, ypred_fuzzy, digits=5))