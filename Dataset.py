# uzyc os walk zamiast dwa razy        for item:
                              #            for itemname:
import os
import face_alignment
from face_alignment import FaceAlignment
from skimage import io, data
import numpy as np
import os.path
import math 
import skimage.transform
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import pandas as pd
import shutil



class InstagramDataset:

  def __init__(self):

    '''Tworzy liste kont, wczytuje aktualna sciezke oraz wczytuje konta do listy '''

    self.accounts = [] 
    self.current_path = os.getcwd()+"/InstagramDataset/" 
    self.read_account()


  def create_labels(self, delete_no_pair=False):

    '''
    Jesli istnieja txt dla ktorych nie ma zdjec delete_no_pair=True usunie je
    Jesli  nie istnieja dir dla kont w Images create_main_label_dir=True utworzy je

    Sprawdza czy dla pliku jpg jest plik txt
    jesli nie
    Tworzy label dla kazdego pliku jpg znajdujacego sie w Images

    !!!!!!!!!!!!!!!!!!!!!                         TRZEBA NAJPIERW utworzyc foldery dla labels           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   

    '''

    ## Zoptymalizowac to bo niepotrzebnie sie zawsze cale wykonuje 


    jpg_files = set()
    txt_files_paired = set() 
    txt_files_unpaired = set() 
    img_pth = self.current_path+"Image/"
    label_pth = self.current_path+"label/"

  #Create set with pair of files (txt with jpg)
    for name in os.listdir(img_pth):
      for file in os.listdir(img_pth+name):
        jpg_files.add(img_pth+name+"/"+file[:-3]+"txt") # zmieniamy w secie jpg na txt zeby otem odjac i latwo dorobic pliki
        lbl_pth = img_pth+name+"/"+file[:-3]+"txt"
        lbl_pth = lbl_pth.replace("Image", "label")
        if os.path.isfile(lbl_pth):
          txt_files_paired.add(lbl_pth)
    

    #Create set with all txt files (unpaired also)
    for name in os.listdir(label_pth):
      for file in os.listdir(label_pth+name):
          txt_files_unpaired.add(label_pth+name+"/"+file)

    #Find files to delete (subtract sets) and delete them
    if delete_no_pair == True:
      txt_to_delete = txt_files_unpaired - txt_files_paired
      for file in txt_to_delete:
        print(file)
        os.remove(file)
        txt_files_unpaired.remove(file)
      
      print(len(txt_files_paired))
      print(len(txt_files_unpaired))
    else:
      pass

    ################### TUTAJ NAPISAC TWORZENIE LABELI DLA PLIKOW JPG BEZ LABELA I DIR ORAZ 20  SPACJI MA WRZUCAC
    #txt_to_create = jpg_files - txt_files_paired
    #for file in txt_to_create:
      #f = open(file, "x")


  def read_account(self): # potem zmienic na samo ig_users.txt

    ''' wczytuje osoby podane w pliku ig_users.txt'''

    pth_ig_users = self.current_path+"ig_users.txt"
    with open(pth_ig_users) as f:
      lines = [line.rstrip('\n') for line in f]
    for line in lines:
      self.accounts.append(line)
    print(self.accounts)
    print("W bazie znajduje sie: ",len(self.accounts)," kont")
    return self.accounts
    
  def count_images_labels(self, ig_user, all=False): # zamienic na try except

    '''liczy zdjecia i labele dla podanego ig_user'''

    pth_to_data = self.current_path
    if self.accounts.count(ig_user) == True:
      countimage = 0
      countlabel = 0
      for file in os.listdir(pth_to_data+"Image/"+ig_user):
        countimage +=1
      for file in os.listdir(pth_to_data+"label/"+ig_user):
        countlabel +=1
      print("{} posiada {} zdjec oraz {} adnotacji do nich".format(ig_user,countimage,countlabel))
    else:
      print("ERROR {} NOT EXIST".format(ig_user))

  def sortuj(self): # to z decoratorem bedzie
    #print("DZIALA")
    pass

  def rename(self): # to z decoratorem bedzie
    pass

  def create_main_dir(self, dirname, all_ig_users=False):
    try:
        os.mkdir(self.current_path+dirname)
    except OSError:
        print("Creation of the directory %s failed" % self.current_path+dirname)
    else:
        print("Successfully created the directory %s " % self.current_path+dirname)

    if all_ig_users == True:
      for ig_user in self.accounts:
        try:
          os.mkdir(self.current_path+dirname+"/"+ig_user)
        except OSError:
          print("Creation of the directory %s failed" % self.current_path+dirname+"/"+ig_user)
        else:
          print("Successfully created the directory %s " % self.current_path+dirname+"/"+ig_user)
    else:
      pass

  def create_faceapp_data(self, treshold=0.93,copy_files=False,dest_path="/content/drive/MyDrive/projects/Instagram_dataset_version_1.0/To_faceapp/"):
    '''
    W plik.csv mam zapisane zdjecia juz
    Jak bedzie ich duzo wiecej to mozna ustawic jakis treshold
    Dla kazdego zdjecia w face(image) przenosi zdjecie z image do folderu w gdrive
    zwraca liste z  gotowym juz do os.shutil move czy tam kopiuj
    '''
    
    df = pd.read_csv('/content/InstagramDataset/plik.csv')  
    # Usuwa jesli podobienstwo jest wieksze niz treshold
    df = df[df['podobienstwo'] < treshold]

    #df['Unnamed: 0'] = self.current_path+'Image/' + df['Unnamed: 0'].astype(str)
    # przerobic kolumne z nazwami na liste
    df['Unnamed: 0'] = df['Unnamed: 0'].str.replace("npy","jpg")    
    files = df['Unnamed: 0'].to_numpy()
    #print(files)
    # Tworzy kolejna liste zeby dodac glowny folder dla kazdego konta
    files_upper = []
    for osoba in files:
      osoba = osoba[:-9]
      osoba = osoba+"/"
      files_upper.append(osoba)

    #laczy w calosc
    zipped_lists = ["{}{}".format(files_upper_, files_) for files_upper_, files_ in zip(files_upper, files)]
    for idx, item in enumerate(zipped_lists):
      zipped_lists[idx] = self.current_path+"Image/"+item
    print(zipped_lists)
    if copy_files == True:
      for item in zipped_lists:
        shutil.copy2(item, dest_path)
        print("Przekopiowano {}".format(item))



class Facenet:

  def __init__(self, show=False):
    self.vector_path = os.getcwd()+"/InstagramDataset/face(vector)/"
    self.model = tf.keras.models.load_model(
    "/content/facenet_keras.h5", custom_objects=None, compile=True, options=None)
    if show ==True:
      print("FaceNet Podsumowanie wej/wyj:")
      print(self.model.inputs)
      print(self.model.outputs)
    print(type(self.model))
    print(self.model)

  def get_feature_vector(self, filename, required_size=(160, 160)):
    '''
    Bierze zdjecie wycina z niego twarz i zwraca je jako nparray
    '''
    
	# load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    print(type(results))
    print(results)
    try:
      # extract the bounding box from the first face
      x1, y1, width, height = results[0]['box']
    except IndexError:
      print("{} nie wykryto twarzy".format(filename))
      # zatrzymac fnkcje cala
    # bug fix
    if not results:
      print("nie wykryto twarzy")
      return
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    '''
    Bierze twarz jako nparray i zwraca jako pukt w 128 wymiarowej przestzreni
    jesli save_embedding jest True to zapisuje do dir face(vector)
    '''
    
    # scale pixel values
    face_array = face_array.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_array, axis=0)
    # make prediction to get embedding
    yhat = self.model.predict(samples)
    return yhat[0]

  def add_data_face_embeddings(self):
    '''
    All_data
    robi wektory ze wszytskich dla kazdego
    '''
    for name in os.listdir(os.getcwd()+"/InstagramDataset/face(image)/"):
      for file in os.listdir(os.getcwd()+"/InstagramDataset/face(image)/"+name):
        vector_pth = self.vector_path+name+"/"+file
        feature_vector = self.get_feature_vector(os.getcwd()+"/InstagramDataset/face(image)/"+name+"/"+file)
        print(type(feature_vector))
        try:
          if feature_vector.size == 0:
            print(file,"PUSTY POMIJA")
          else:
            np.save(vector_pth.replace(".jpg", ""), feature_vector)
        except AttributeError:
          pass


  def findCosineDistance(self, source_representation, test_representation):
    """"
    Funckja do porownywania dwuch punktow w przestrzeni niezaleznie od ilosci wymiarow
    """
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


  def compare_face(self, face_embedding="/content/emixxly.npy",save_df=False):
    '''
    porownuje twarze aktualnie posiadane w bazie danych do wybranej

    zwraca pd dataframe
    '''
    startpath="/content/InstagramDataset/face(vector)/"
    df = pd.DataFrame()
    emixxly_vector = np.load(face_embedding, allow_pickle=True) # TU ZMIENIC NA SCIEZKE TAM GDZIE BEDZIE EMIXXLY
    for name in os.listdir(startpath):
      for file in os.listdir(startpath+name):
        vector = np.load(startpath+name+"/"+file,allow_pickle=True)
        result = self.findCosineDistance(emixxly_vector,vector)
        df_npy = pd.DataFrame(result, columns = [file], index = ["podobienstwo"])
        df_npy = df_npy.T
        df = df.append(df_npy)
        
    df = df.sort_values(by=['podobienstwo'])
    if save_df == True:
      df.to_csv('Suki.csv') 
    return df

facenet = Facenet() # kiedys nazywalo sie generator wiec jesli uzywam starych komend to pozmieniac

        
data = InstagramDataset()


#data.iterate(data.sortuj, path_to_dir="/content/InstagramDataset/Image/", main_dir=True)
#data.iterate(data.count_images_labels, path_to_dir="/content/InstagramDataset/Image/", main_dir=False)
#face_landmarks = img.get_landmarks('/content/InstagramDataset/Image/toochi0025.jpg')
#print(face_landmarks)
#face = lbl.get_landmarks('/content/InstagramDataset/Image/toochi_kash/toochi0025.jpg')
#img.get_face_angle("/content/InstagramDataset/label/asdf_0014.txt")
#data.create_labels(delete_no_pair=True) # TO MUSI ZOSTAC USUWA LABELE BEZ PARY
#data.count_images_labels("realmarycarey")
#lbl.label_add_data_batch_angle("face_angle", save_to_label=True) 
#lbl.label_add_data_batch_face_image()
#gen.compare_face(save_df=True)

#gen.add_data_face_embeddings() # Msui zostac przed wprowadzeniem 1.5
#feature_vector = gen.get_feature_vector("/content/drive/MyDrive/projects/Instagram_dataset_version_1.0/MODEL/zwykla.jpg")
#np.save("emixxly", feature_vector) 
#data.create_main_dir("segmentation/osoba",all_ig_users=True) 

#data.create_faceapp_data(copy_files=True) # Przekopiowywanie do face app






