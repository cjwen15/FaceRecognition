
import pickle

f = open('knn_classifier.pkl', 'rb')
info = pickle.load(f)
print(info)