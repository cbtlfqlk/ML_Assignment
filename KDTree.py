import math
import sys

sys.setrecursionlimit(100000)

class kdTree():
    def __init__(self,df):
        self.tree = []                                                                             # Initializes an empty tree.
        self.attributes = df.columns                                                               # Defines the list of attributes to consider.
        self.build(self.tree,df.reset_index(drop=True).reset_index(),0)                            # Builds the tree based on our dataframe, starting with the 1st attribute.

    def build(self, tree, df, attribute,*,fails=0):                                                # Function to build our KDtree recursively.
        if len(df) == 0: return                                                                    # The recursive calls end when our dataframe has size 0,
        if len(df) == 1: tree.extend([None,list(df.iloc[0][1:]),df.iloc[0]["index"]]); return      # Or when it has size 1, in which case it adds that element to the tree.
        col = self.attributes[attribute]                                                           # Gets the current attribute of our list of attributes.
        df = df.sort_values(by=col)                                                                # Sorts the dataframe using our selected attribute as the criteria.
        t=len(df)
        nextAtt=(attribute+1)%len(self.attributes)                                                 # Gets the next attribute in the list.
        nextfails = 0
        i = t//2
        if fails != len(self.attributes):                                                          # If all elements are the same
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

    
    def exploreD(self,row,tree,array,eps,distF,ref):
        if not tree: return
        if tree[0] is None:
            if distF(row,tree[1])<=eps:array.append(tree[2])
            return
        col = tree[0]
        node = tree[1]
        if (not (isinstance(node[col],(float)) and math.isnan(node[col]))) and ((isinstance(row[col],(float))and math.isnan(row[col])) or row[col] > node[col]):
            self.exploreD(row,tree[3],array,eps,distF,ref)
            t = ref[col]
            ref[col] = node[col]
            if distF(row,ref) > eps: ref[col] = t;return
            if distF(row,node)<=eps:array.append(tree[4])
            self.exploreD(row,tree[2],array,eps,distF,ref)
        else:
            self.exploreD(row,tree[2],array,eps,distF,ref)
            t = ref[col]
            ref[col] = tree[5]
            if distF(row,node)<=eps:array.append(tree[4])
            if distF(row,ref) > eps: ref[col] = t;return
            self.exploreD(row,tree[3],array,eps,distF,ref)
        ref[col] = t
    def epsNeighbors(self,row,eps,distF)-> [int]:
        array = []
        ref = list(row)
        self.exploreD(list(row),self.tree,array,eps,distF,ref)
        return array
