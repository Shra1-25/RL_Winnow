import gym
import random
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
import tensorflow as tf
import numpy as np
import random
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                  loop controls style="height: 400px;">
                  <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                  </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
    

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env

#function to calculate accuracy
def accuracy_score(y, y_pred):
    
    acc = np.mean(y == y_pred)
    return acc


#function to calculate netsum and prediction for one instance
def predictOne(W, X, thres):
    
    netsum = np.sum(W * X) #net sum
    
    #threshold check
    if netsum >= thres:
        y_hat = 1
    else:
        y_hat = 0
    
    return y_hat


#function to calculate netsums and predictions for all instances
def predictAll(W, X, thres):
    
    NetSum = np.dot(X, W)
    Idx = np.where(NetSum >= thres)
    y_pred = np.zeros(X.shape[0])
    y_pred[Idx] = 1
    
    return y_pred


#function to compute and print the classification summary
def ComputePerf(W, X, y, thres, print_flag = False):
    
    y_pred = predictAll(W, X, thres) #compute the prediction
    acc = accuracy_score(y, y_pred) #compute accuracy

    #print the summary if printflag is True
    if print_flag == True:
        print('Accuracy:',acc)
        #print(confusion_matrix(y, y_pred))
        #print(classification_report(y, y_pred))
    
    return acc
def VGG16_features(imgs, model):
    binary_img_features = []
    for img in imgs:
        print(img.shape)
        resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)#(240, 160),img_rgb
        #resized_img = np.expand_dims(resized_img, axis=0)
        resized_img = preprocess_input(resized_img)
        #print("resized image size: "+str(resized_img.shape))

        vgg_features = np.squeeze(Flatten()(model.predict(np.array([resized_img]))))
        #print("vgg_feature size: "+str(vgg_features.shape))
        mean_arr = np.mean(vgg_features,axis=0)
        #print("mean_arr size: "+str(mean_arr.shape))
        temp_feat = np.where(vgg_features<mean_arr.squeeze(), 0, 1)
        #print("temp_feat size: "+str(temp_feat.shape))
        binary_img_features.append(temp_feat)
      
    binary_img_features = np.array(binary_img_features)
    #print("binary_img_features size: "+str(binary_img_features.shape))


    return binary_img_features

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('-s', '--steps', default=200, type=int, help='max number of steps per episode.')
    parser.add_argument('-e', '--episodes', default=10, type=int, help='number of episodes.')
    args = parser.parse_args()
print(" >> Training... ")
env = wrap_env(gym.make("CartPole-v1"))
model = VGG16(weights='imagenet', include_top=False)
img_feat=[]
W = np.ones((3,25088)) #winnow2 initialisation
alphas = [2,3,4]
best_model_weights = np.ones((3,25088))
best_scores=[0,0,0]
thres = [25088, 25088 // 2]
for ep in range(args.episodes): 
    env.reset()
    print("\nEpisode: "+str(ep))
    for i in range(args.steps):
        img=env.render(mode='rgb_array')
        #cv2_imshow(img)
        img_feat = (np.squeeze(VGG16_features(np.array([img]),model)))
        print("img_feat size: "+str(np.array(img_feat).shape))
        X_train = img_feat
        #Winnow loop (computation and update) starts
        y_hat = predictOne(W[0], X_train, thres[0])
        action = np.squeeze(y_hat)
        next_state, reward, done, info = env.step(action)
        #Winnow prediction is a mismatch
        #if y_train[i] != y_hat:
        if done:

            #active attribute indices
            Idx = np.where(X_train == 1)

            if y_hat == 0: 
                W[0,Idx] *= alphas[0] #netsum was too small (promotion step)

            else:
                W[0,Idx] /= alphas[0] #netsum was too high (demotion step)
            if i>best_scores[0]:
                best_scores[0] = i
                best_model_weights[0] = W[0]
	
            print("state: ",i, next_state, reward, done, info)
            print("action: ",action)
            print("score for alpha=" +str(alphas[0])+": ", i, "best scire: ", best_scores)
            break
    
        print("state: ",i, next_state, reward, done, info)
        print("action: ",action)
        print("score for alpha="+str(alphas[0])+": ", i, "best score: ",best_scores)
np.savetxt('winnow_weights.csv',best_model_weights, delimiter=', ')


print("\n >> Testing...")
test_W = np.loadtxt('winnow_weights.csv', delimiter=', ')
env = wrap_env(gym.make("CartPole-v1"))
model = VGG16(weights='imagenet', include_top=False)
img_feat=[]
env.reset()

for i in range(200):
    img=env.render(mode='rgb_array')
    #cv2_imshow(img)
    img_feat = (np.squeeze(VGG16_features(np.array([img]),model)))
    print("img_feat size: "+str(np.array(img_feat).shape))
    X_test = img_feat
    #Winnow loop (computation and update) starts
    y_hat = predictOne(test_W[0], X_test, thres[0])
    action = np.squeeze(y_hat)
    next_state, reward, done, info = env.step(action)
    #Winnow prediction is a mismatch
    #if y_train[i] != y_hat:
    if done:
      print("state: ",i, next_state, reward, done, info)
      print("action: ",action)
      print("Score: ", i)
      break
    
    print("state: ",i, next_state, reward, done, info)
    print("action: ",action)
