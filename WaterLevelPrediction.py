
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from genetic_selection import GeneticSelectionCV
import webbrowser
from sklearn.metrics import mean_squared_error

main = tkinter.Tk()
main.title("Groundwater Level Prediction Using Hybrid Artificial Neural Network with Genetic Algorithm")
main.geometry("1300x1200")

global filename, dataset
global X, Y, X_train, X_test, y_train, y_test, Y1
global mse, text, pathlabel

'''
gray wolf optimization algortihm which consists of different wolf such as alpha, beta, delta and omega and all this wold will hunt in group
alpha wolf is the commander and help in taking optimal decision
omega wolf will separate prey from group
delta and beta will attack the prey

in features selection also we will apply alfa technique to select optimize features
using delta and beta will calculate fitness of each features and the best fitness features will be selected
omega will help to remove irrelevant features
'''
def grayWolf(X, Y):
    X_selected_features = None
    #take X as Random population and shuffle it randomly
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    features = X.T
    solution = np.cov(features.astype(float))
    iterations, vectors = np.linalg.eig(solution)#get solution for each features (to allow alpha wolf to take decision)
    fitness = []
    for i in range(len(iterations)):
        fitness.append(round(iterations[i] / np.sum(iterations), 8)) #calculate fitness values
    optimal_features = np.sort(fitness)[::-1] #list of best optimal features
    selected_features = np.zeros(len(fitness)) #wolf or selected features population
    for i in range(0,X.shape[1]):
        for j in range(len(fitness)):
            if optimal_features[i] > fitness[j]:
                selected_features[j] = 1
    return selected_features
      
#function to optimize features with crow search
def crowSearch(X, Y):
    selected = []
    fitness = 0
    for i in range(2,12): #loop each features
        features = X[:,0:i]
        X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.2)
        classifier = linear_model.LogisticRegression(max_iter=1000) #train the classifier
        classifier.fit(X_train, y_train)
        acc = accuracy_score(classifier.predict(X_test), y_test)#calculate accuracy as fitness
        print(str(acc)+" "+str(i))
        if acc > fitness: #if fitness high then select features else ignore it
            fitness = acc
            selected = i
    return selected    

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def processDataset():
    global filename, dataset, X, Y, Y1
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head()))
    dataset['Situation'] = pd.Series(le.fit_transform(dataset['Situation'].astype(str)))
    dataset = dataset.values
    X = dataset[:,1:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    Y1 = dataset[:,dataset.shape[1]-2]
    Y = Y.astype('int')

def ANNwithCrow():
    global filename, X, Y, Y1, mse
    mse = []
    text.delete('1.0', END)
    text.insert(END,"Total features found in dataset before applying Crow Search GA : "+str(X.shape[1])+"\n")
    #call crow search and get selected features
    crow_search_features = crowSearch(X, Y)
    X1 = X[:,0:crow_search_features]
    #now define genetic algorithm object
    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr") #
    selector = GeneticSelectionCV(estimator, cv=5, verbose=1, scoring="accuracy", max_features=5, n_population=5, crossover_proba=0.5, mutation_proba=0.2,
                                  n_generations=5, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=2,
                                  caching=True, n_jobs=-1)
    selector = selector.fit(X1, Y)#OPTIMIZING CRow FEATURES WITH GENETIC ALGORITHM and then select tnem
    print(selector.support_)
    X_selected_features = X1[:,selector.support_==True] #take selected features
    print(X_selected_features.shape)
    text.insert(END,"Total features found in dataset after applying Crow Search GA : "+str(X_selected_features.shape[1])+"\n")
    #now split selected features into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_selected_features, Y1, test_size=0.2)
    #now build ann model with different layers
    ann_model = Sequential()
    ann_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(512))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer="adam", loss='mse', metrics=['mae']) #compile the model
    hist = ann_model.fit(X_train, y_train, batch_size=16,epochs=100, validation_data=(X_test, y_test))#train the model on train data and test on test data
    predict = ann_model.predict(X_test)#perform prediction on test data
    error = mean_squared_error(predict, y_test)#calculatee MSE
    mse.append(error)
    text.insert(END,"ANN with Crow Search MSE : "+str(error)+"\n\n")

    output = '<table border=1 align=center>'
    output+= '<tr><th>Algorithm Name</th><th>Test Data Water Level</th><th>Predicted Water Level</th></tr>'
    for i in range(len(predict)):
        output+='<tr><td>ANN with Crow Search GA</td><td>'+str(y_test[i])+'</td><td>'+str(predict[i])+"</td></tr>"
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)   
    plt.plot(y_test, color = 'red', label = 'Available Test Data Water Level')
    plt.plot(predict, color = 'green', label = 'Predicted Water Level')
    plt.title('ANN with Crow Search Water Level Prediction')
    plt.xlabel('Test Data Values')
    plt.ylabel('Water Level Prediction')
    plt.legend()
    plt.show()

def ANNwithWolf():
    global filename, X, Y, Y1, mse
    text.insert(END,"Total features found in dataset before applying Gray Wolf GA : "+str(X.shape[1])+"\n")
    gray_wolf_features = grayWolf(X, Y)
    X1 = X[:,gray_wolf_features==1]

    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr") #
    selector = GeneticSelectionCV(estimator, cv=5, verbose=1, scoring="accuracy", max_features=5, n_population=5, crossover_proba=0.5, mutation_proba=0.2,
                                  n_generations=5, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=2,
                                  caching=True, n_jobs=-1)
    selector = selector.fit(X1, Y)#OPTIMIZING FEATURES WITH GENETIC ALGORITHM OBJECT SELECTOR
    print(selector.support_)
    X_selected_features = X1[:,selector.support_==True]
    print(X_selected_features.shape)
    text.insert(END,"Total features found in dataset after applying Gray Wolf GA : "+str(X_selected_features.shape[1])+"\n")
    X_train, X_test, y_train, y_test = train_test_split(X_selected_features, Y1, test_size=0.2)
    ann_model = Sequential()
    ann_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(512))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer="adam", loss='mse', metrics=['mae'])
    hist = ann_model.fit(X_train, y_train, batch_size=16,epochs=100, validation_data=(X_test, y_test))
    predict = ann_model.predict(X_test)

    error = mean_squared_error(predict, y_test)
    mse.append(error)
    text.insert(END,"ANN with Gray Wolf MSE : "+str(error)+"\n\n")

    output = '<table border=1 align=center>'
    output+= '<tr><th>Algorithm Name</th><th>Test Data Water Level</th><th>Predicted Water Level</th></tr>'
    for i in range(len(predict)):
        output+='<tr><td>ANN with Grey Wolf GA</td><td>'+str(y_test[i])+'</td><td>'+str(predict[i])+"</td></tr>"
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)  

    plt.plot(y_test, color = 'red', label = 'Available Test Data Water Level')
    plt.plot(predict, color = 'green', label = 'Predicted Water Level')
    plt.title('ANN with Gray Wolf GA Water Level Prediction')
    plt.xlabel('Test Data Values')
    plt.ylabel('Water Level Prediction')
    plt.legend()
    plt.show()

def graph():
    height = mse
    bars = ('ANN with Crow Search GA', 'ANN with Gray Wolf GA')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("ANN MSE Comparison Between Crow Search & Gray Wolf")
    plt.xlabel("Algorithm Names")
    plt.title("MSE ERROR")
    plt.show()

def close():
    main.destroy()

def GUI():
    global main, text, pathlabel
    font = ('times', 16, 'bold')
    title = Label(main, text='Groundwater Level Prediction Using Hybrid Artificial Neural Network with Genetic Algorithm')
    title.config(bg='brown', fg='white')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 13, 'bold')
    upload = Button(main, text="Upload Ground Water Level Dataset", command=uploadDataset)
    upload.place(x=50,y=100)
    upload.config(font=font1)  

    pathlabel = Label(main)
    pathlabel.config(bg='brown', fg='white')  
    pathlabel.config(font=font1)           
    pathlabel.place(x=400,y=100)

    preprocess = Button(main, text="Preprocess Dataset", command=processDataset)
    preprocess.place(x=50,y=150)
    preprocess.config(font=font1) 

    anncrow = Button(main, text="Run ANN with Crow Search GA", command=ANNwithCrow)
    anncrow.place(x=300,y=150)
    anncrow.config(font=font1) 

    annwolf = Button(main, text="Run ANN with Gray Wolf GA", command=ANNwithWolf)
    annwolf.place(x=600,y=150)
    annwolf.config(font=font1) 

    graphButton = Button(main, text="MSE Comparison Graph", command=graph)
    graphButton.place(x=50,y=200)
    graphButton.config(font=font1) 

    exitButton = Button(main, text="Exit", command=close)
    exitButton.place(x=300,y=200)
    exitButton.config(font=font1) 

    font1 = ('times', 12, 'bold')
    text=Text(main,height=30,width=150)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=250)
    text.config(font=font1)

    main.config(bg='brown')
    main.mainloop()

if __name__ == "__main__":
    GUI()
