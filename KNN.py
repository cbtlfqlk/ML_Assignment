from pandas import read_csv
import math
import sys

sys.setrecursionlimit(100000)

if __name__ == "__main__":
    file = r"class_imbalance\dataset_38_sick.csv"
    df = read_csv(file)




class kdTree():
    def __init__(self,df,distance_function = None):
        self.tree = []      #Nodes are composed of 4 elements: the feature used to separate, the row used as the node, lesser or equal, and greater
        self.attributes = df.columns
        self.build(self.tree,df.reset_index(drop=True).reset_index(),0)
    def build(self, tree, df, attribute,*,fails=0):
        if len(df) == 0: return
        if len(df) == 1: tree.extend([None,list(df.iloc[0][1:]),df.iloc[0]["index"]]); return
        col = self.attributes[attribute]
        df = df.sort_values(by=col)
        t=len(df)
        nextAtt=(attribute+1)%len(self.attributes)
        nextfails = 0
        i = t//2
        if fails != len(self.attributes):   #If all elements are the same
            while i and df.iloc[i-1][col]==df.iloc[i][col]:
                i-=1
            if not i:
                i = t//2
                t0 = t-1
                while i<t0 and df.iloc[i][col] == df.iloc[i+1][col]:
                    i+=1
                if i==t0:
                    self.build(tree,df,nextAtt,fails=fails+1)
                    return
            else: i-=1
        else:
            nextfails = fails
        tree.extend([attribute,list(df.iloc[i][1:]),[],[],df.iloc[i]["index"],df.iloc[i][col] if i==t-1 else df.iloc[i+1][col]])
        self.build(tree[2],df.head(i),nextAtt,fails = nextfails)
        self.build(tree[3],df.tail(t-i-1),nextAtt,fails = nextfails)
    def insert(ind,dist,array):
        i = 0
        while i<len(array) and dist < array[i][1]:
            array[i] = array[i+1]
            i+=1
        if i==0: return
        array[i-1]=(ind,dist)
    def explore(self,row,tree,array,distF,ref):
        if not tree: return
        if tree[0] is None: kdTree.insert(tree[2],distF(row,tree[1]),array);return
        col = tree[0]
        node = tree[1]
        if (not (isinstance(node[col],(float)) and math.isnan(node[col]))) and ((isinstance(row[col],(float))and math.isnan(row[col])) or row[col] > node[col]):
            self.explore(row,tree[3],array,distF,ref)
            t = ref[col]
            ref[col] = node[col]
            if distF(row,ref) >= array[0][1]: ref[col] = t;return
            kdTree.insert(tree[4],distF(row,node),array)
            self.explore(row,tree[2],array,distF,ref)
        else:
            self.explore(row,tree[2],array,distF,ref)
            t = ref[col]
            ref[col] = tree[5]
            kdTree.insert(tree[4],distF(row,node),array)
            if distF(row,ref) >= array[0][1]: ref[col] = t;return
            self.explore(row,tree[3],array,distF,ref)
        ref[col] = t
    def knn(self,row,k,distF)-> [(int,int)]:
        array = [*((0,float("inf")),)*k,(0,0)]
        ref = list(row)
        self.explore(list(row),self.tree,array,distF,ref)
        return map(lambda x:(int(x[0]),float(x[1])),array[:-1])
        
class KNN_Classifier():
    def __init__(self,k=7,*,distance_function=None):
        self.k = k
        self.df = None
        self.df_target = None
        self.distF = distance_function or KNN_Classifier.distance
    def fit(self,data_frame,data_frame_target,*, KDtree = True):
        if KDtree:
            self.df = kdTree(data_frame)
            self.knnF = self.df.knn
        else:
            self.df = data_frame.reset_index(drop=True)
            self.knnF = self.df
        self.df_target = data_frame_target.reset_index(drop=True)
    def distance(row1,row2): #Está a considerar a distância com nan sempre como 1
        Sum = 0 #É necessário normalizar os valores
        for i,j in zip(row1,row2):
            if (isinstance(i,(float,int)) and not(math.isnan(i) or math.isnan(j))): Sum+=abs(i-j)
            else: Sum += i!=j
        return Sum
    def knn(self,row,*_):
        dists = [(i,self.distF(row,row0)) for i,row0 in self.df.iterrows()]
        knn = sorted(dists,key=lambda x:x[1])[:self.k]
        return knn
    def predict_row(self,row):
        knn = [self.df_target.iloc[i[0]] for i in self.knnF(row,self.k,self.distF)]
        counter = {}
        for i in knn:
            if not i in counter: counter[i] = 0
            counter[i]+=1
        pred = max(counter,key=lambda x:counter[x])
        return pred
            
    def predict(self,df):
        return [self.predict_row(row) for _,row in df.iterrows()]
