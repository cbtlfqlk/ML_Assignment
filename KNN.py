from pandas import read_csv
import math

if __name__ == "__main__":
    file = r"class_imbalance\dataset_38_sick.csv"
    df = read_csv(file)

class KNN_Classifier():
    def __init__(self,k=3,*,distance_function=None):
        self.k = k
        self.df = None
        self.df_target = None
        self.distF = distance_function or KNN_Classifier.distance
    def fit(self,data_frame,data_frame_target):
        self.df = data_frame
        self.df_target = data_frame_target
    def distance(row1,row2): #Está a considerar a distância com nan sempre como 1
        Sum = 0 #É necessário normalizar os valores
        for i,j in zip(row1,row2):
            if (isinstance(i,(float,int)) and not(math.isnan(i) or math.isnan(j))): Sum+=abs(i-j)
            else: Sum += i!=j
        return Sum
    def predict_row(self,row):
        dists = [(i,self.distF(row,row0)) for i,row0 in self.df.iterrows()]
        knn = [self.df_target.iloc[i[0]] for i in sorted(dists,key=lambda x:x[1])[:self.k]]
        counter = {}
        for i in knn:
            if not i in counter: counter[i] = 0
            counter[i]+=1
        pred = max(counter,key=lambda x:counter[x])
        return pred
            
    def predict(self,df):
        return [self.predict_row(row) for _,row in df.iterrows()]


