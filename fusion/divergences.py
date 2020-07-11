import numpy as np
import tensorflow as tf
import itertools
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

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

streams = [Ypred_val_dist,Ypred_val_distvel, Ypred_val_ang, Ypred_val_angvel]

for p,q in itertools.combinations(streams,2):

  kl_pq = rel_entr(p, q)

  kl_pq_av = np.ndarray(shape=(430,))

  for i in range(0,430):
    kl_pq_av[i] = np.sum(kl_pq[i], axis = 0)

  kl_qp = rel_entr(q,p)

  kl_qp_av = np.ndarray(shape=(430,))

  for i in range(0,430):
    kl_qp_av[i] = np.sum(kl_qp[i], axis = 0)

  print(np.mean(kl_pq_av), namestr(p, globals()), namestr(q, globals()) )
  print(np.mean(kl_qp_av), namestr(q, globals()), namestr(p, globals()) )

for p,q in itertools.combinations(streams,2):

  js_pq = np.ndarray(shape=(430,))
  for i in range(430):
    js_pq[i] = jensenshannon(p[i],q[i], base=10)


  print(np.mean(js_pq), namestr(p, globals()), namestr(q, globals()) )