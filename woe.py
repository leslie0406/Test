# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:23:49 2021

@author: yunhaoyu
"""
import pandas as pd
import numpy as np
from sklearn import tree
import itertools
import xlwt

def con(w, w_set, var,woe = True):
    if var == TARGET : return(np.nan)
    N_B = sum(w_set[TARGET])
    N_G = len(w_set) - N_B
    nt = w_set[w_set[var] == w]
    nb = sum(nt[TARGET]) / N_B
    if nb == 0:return(0)
    ng = (len(nt) - nb) / N_G
    w = w if woe else np.log(ng / nb)
    return((ng - nb) * w)
def cal_IV(var, woe, iswoe = True):
    var_set = woe[[var, TARGET]]
    woe_set = set(woe[var])
    iV = sum([con(w, var_set, var, iswoe) for w in woe_set])
    return(iV)
    
def list2str(list1):
    con = ''
    for i in list1: 
        con = con + ',' + i
    return con
def check_con(var,i,j):
    return (var[i][j]=='<') or (var[i][j]=='>')
def creat_ca_class(val, var_1, var_2):
    return [val[i]for i in range(len(var_1))if var_1[i] == var_2]
def find_con(lis, key):
    maxima = min([np.inf] + [float((v.split('<'))[1])for v in lis if v[0] == '<'])
    minima = max([-np.inf] + [float((v.split('>'))[1])for v in lis if v[0] == '>'])
    l = (maxima == np.inf) and (minima != -np.inf)
    u_l = (maxima != np.inf) and (minima != -np.inf)
    minima = str(minima)
    maxima = str(maxima)
    con = minima + '<' + key + '<' + maxima if u_l else minima + '<' + key if l else key + '<' + maxima
    return con
def con_list(class_of_tree_key):
    var_0 = class_of_tree_key
    var_1 = var_0.split(',')
    var_2 = [var_1[i][:j]for i in range(1,len(var_1))for j in range(len(var_1[i]))if check_con(var_1,i,j)]
    value_0 = [var_1[i][j:]for i in range (1,len(var_1))for j in range (len(var_1[i]))if check_con(var_1,i,j)]     
    CA_class = {set_var: creat_ca_class(value_0, var_2, set_var)for set_var in set(var_2)}                   
    return [find_con(CA_class[key], key)for key in CA_class.keys()]

def clf(data, v, TARGET):
    max_depth = 3
    dt1 = tree.DecisionTreeClassifier(max_depth=max_depth,#最多切幾層調這裡
                                      min_samples_leaf=0.1)#每群至少要幾個
    train_x = data.loc[: , [v[0], v[1]]]
    train_x = train_x.fillna(-9999999999)
    train_y = data[TARGET].values
    dt1_result = dt1.fit(train_x , train_y)
    
    cut_temp = 0
    #feature_temp = 0
    cut_temp2 = 0
    class_of_tree={}
    condition=[]
    temp_point=[] #現在這層的point
    temp_max_depth_point=[] #換算成最底層的point
    current_depth=[] #算現在第幾層location_neg2
    temp_last_neg2=[] #算最後一個-2
    k1=0    
    for i in range(len(dt1_result.tree_.feature)):
        

           if len(dt1_result.tree_.feature)==1:
              break #決策樹不切組直接跳過

           
           if i < len(dt1_result.tree_.feature)-1:
               cut_right = dt1_result.tree_.threshold[i+1]              
           elif i == len(dt1_result.tree_.feature)-1:
               cut_right = -99             
           if i > 0:              
               cut_temp = dt1_result.tree_.threshold[i-1]
           if i > 1:
               cut_temp2 = dt1_result.tree_.threshold[i-2]              
           cut = dt1_result.tree_.threshold[i]  
           point = 0 #紀錄每次的分數
           depth=0 #紀錄現在第幾層
           depth_point=0 #紀錄換算成max_depth的分數
           last_neg2=0 #季每次最後一個2的位子
           if (cut_temp != -2) & (cut == -2) & (cut_right != -2) :
              k1 = k1+1 
              if  k1 == 1 :
                  temp_last_neg2.append (-1)   
              
              for aa in range(temp_last_neg2[-1]+1,i):
                  cut_left = dt1_result.tree_.threshold[aa]
                  feature_left = dt1_result.tree_.feature[aa]     
                  seq = v[feature_left]+ '<'+str(cut_left) 
                  condition.append(seq)
              class_of_tree[list2str(condition)]=[dt1_result.tree_.value[i][0,0],dt1_result.tree_.value[i][0,1]] 
              depth = len(condition)
              current_depth.append(depth)              
              point = 1
              temp_point.append(point)              
              depth_point = 2**(max_depth-depth)
              temp_max_depth_point.append(depth_point)              
              last_neg2 = i
              temp_last_neg2.append (last_neg2)   
              if  sum(temp_max_depth_point)<2**(max_depth):
                  
                  re_seq = condition[-1].replace('<','>')
                  condition = condition[:-1]
                  condition.append(re_seq)
           if  (cut_temp2 != -2) & (cut_temp == -2) & (cut == -2) :
                 k1 = k1+1
                 if  k1 == 1:
                     temp_last_neg2.append (-1)                                       
                 for aa in range(temp_last_neg2[-1]+1,i-1):
                     cut_left = dt1_result.tree_.threshold[aa]
                     feature_left = dt1_result.tree_.feature[aa]     
                     seq = v[feature_left]+ '<'+str(cut_left) 
                     condition.append(seq)
                 class_of_tree[list2str(condition)]=[dt1_result.tree_.value[i-1][0,0],dt1_result.tree_.value[i-1][0,1]]                   
                 re_seq = condition[-1].replace('<','>')
                 condition = condition[:-1]
                 condition.append(re_seq)
                 class_of_tree[list2str(condition)]=[dt1_result.tree_.value[i][0,0],dt1_result.tree_.value[i][0,1]]                   
                 depth=len(condition)
                 current_depth.append(depth)                  
                 point = 2
                 temp_point.append(point)
                 depth_point = point*2**(max_depth-depth)
                 temp_max_depth_point.append(depth_point)                  
                 last_neg2 = i
                 temp_last_neg2.append (last_neg2)
                 neg2_count=0
                 for j in range(i+1,len(dt1_result.tree_.feature)):
                     if dt1_result.tree_.feature[j] == -2:
                        neg2_count=neg2_count+1
        
                     else:
                         break
                 if  sum(temp_max_depth_point)<2**(max_depth) :                    
                     for jj in range(i+1,i+neg2_count+1):                                                                               
                         re_point = sum(temp_max_depth_point)/2**(max_depth-depth)   
                         location = re_point
                         quo = location/2
                         power2=0  
                         if (location%2==0) & (quo%2!=0):
                             power2=2 #1+1
                         if (location%2==0) & (quo%2==0):
                             for ii in range(0, len(dt1_result.tree_.feature)):
                                 quo = quo/2
                                 power2 = power2+1
                                 if quo%2!=0:
                                    break
                             power2 = power2+2
                          
                         re_seq = condition[-power2].replace('<','>')
                         condition = condition[:-power2]
                         condition.append(re_seq)                     
                         class_of_tree[list2str(condition)]=[dt1_result.tree_.value[jj][0,0],dt1_result.tree_.value[jj][0,1]]                   
                         depth=len(condition)
                         current_depth.append(depth)                  
                         point = 1
                         temp_point.append(point)                 
                         depth_point = 2**(max_depth-depth)
                         temp_max_depth_point.append(depth_point)                  
                         last_neg2 = jj
                         temp_last_neg2.append (last_neg2)
                     if  sum(temp_max_depth_point)<2**(max_depth) :    
                         re_point = sum(temp_max_depth_point)/2**(max_depth-depth)   
                         #print(re_point)
                         location = re_point
                         quo = location/2
                         power2=0  
                         if (location%2==0) & (quo%2!=0):
                             power2=2                                    
                         if (location%2==0) & (quo%2==0):
                             for ii in range(0, len(dt1_result.tree_.feature)):
                                 quo = quo/2
                                 power2 = power2+1
                                 if quo%2!=0:
                                     break
                             power2 = power2+2                            
                         re_seq = condition[-power2].replace('<','>')
                         condition = condition[:-power2]
                         condition.append(re_seq)        
                                         
           if  sum(temp_max_depth_point)==2**(max_depth):
               break

    class_of_tree={list2str(con_list(key)): class_of_tree[key]for key in class_of_tree.keys()}
    return(class_of_tree)
def cot2exc(class_of_tree, table, startrow):
        startrow=startrow  
        table.write(startrow,3,'Sum')
        table.write(startrow,4,'%Goog')
        table.write(startrow,5,'%Bad')
        table.write(startrow,6,'Odds')
        table.write(startrow,7,'Woe')
        table.write(startrow,8,'Contribution')
        table.write(startrow,9,'Bad(% of Sum)')
        table.write(startrow,10,'%')
                                      
        if len(v) ==2 :
           table.write(startrow,0,v[0]+'&'+v[1])
           #print (v[0]+'&'+v[1])
        else:
           table.write(startrow,0,v[0])
        table.write(startrow,1,'Good')
        table.write(startrow,2,'Bad')

        row=startrow + 1         
        for key, value in class_of_tree.items():
            table.write(row,0,key)
            table.write(row,1,value[0])
            table.write(row,2,value[1])

            row=row+1

        #print(row)
        for rows in range(startrow + 1,row) :
            table.write(rows,3,xlwt.Formula('B'+str(rows+1)+'+C'+str(rows+1)))
            table.write(rows,4,xlwt.Formula('B'+str(rows+1)+'/B'+str(row+1)))
            table.write(rows,5,xlwt.Formula('C'+str(rows+1)+'/C'+str(row+1)))
            table.write(rows,6,xlwt.Formula('E'+str(rows+1)+'/F'+str(rows+1)))
            table.write(rows,7,xlwt.Formula('LN(G'+str(rows+1)+')'))
            table.write(rows,8,xlwt.Formula('(E'+str(rows+1)+'-F'+str(rows+1)+')*H'+str(rows+1)))
            table.write(rows,9,xlwt.Formula('C'+str(rows+1)+'/D'+str(rows+1)))
            table.write(rows,10,xlwt.Formula('D'+str(rows+1)+'/D'+str(row+1)))
            
        table.write(row,0,'total')
        table.write(row,1,xlwt.Formula('SUM(B'+str(startrow + 2)+':B'+str(row)+')'))
        table.write(row,2,xlwt.Formula('SUM(C'+str(startrow + 2)+':C'+str(row)+')'))
        table.write(row,3,xlwt.Formula('SUM(D'+str(startrow + 2)+':D'+str(row)+')'))
        table.write(row,8,xlwt.Formula('SUM(I'+str(startrow + 2)+':I'+str(row)+')'))
        table.write(row,9,xlwt.Formula('C'+str(row+1)+'/D'+str(row+1)))
        
        return(table, row)
    
def cot2exc2(class_of_tree, table, startrow, test_data):
        startrow=startrow  
        table.write(startrow,3,'Sum')
        table.write(startrow,4,'%Goog')
        table.write(startrow,5,'%Bad')
        table.write(startrow,6,'Odds')
        table.write(startrow,7,'Woe')
        table.write(startrow,8,'Contribution')
        table.write(startrow,9,'Bad(% of Sum)')
        table.write(startrow,10,'%')
        
        table.write(startrow,13,'Sum')
        table.write(startrow,14,'%Goog')
        table.write(startrow,15,'%Bad')
        table.write(startrow,16,'Odds')
        table.write(startrow,17,'Woe')
        table.write(startrow,18,'Contribution')
        table.write(startrow,19,'Bad(% of Sum)')
        table.write(startrow,20,'%')
        table.write(startrow,21,'Index')                                             
        if len(v) ==2 :
           table.write(startrow,0,v[0]+'&'+v[1])
           #print (v[0]+'&'+v[1])
        else:
           table.write(startrow,0,v[0])
        table.write(startrow,1,'Good')
        table.write(startrow,2,'Bad')
        
        table.write(startrow,11,'Good')
        table.write(startrow,12,'Bad')  
        row=startrow + 1         
        for key, value in class_of_tree.items():
            table.write(row,0,key)
            table.write(row,1,value[0])
            table.write(row,2,value[1])
            subdf = key2con(key,test_data)
            test_b = sum(subdf.Bad2)
            table.write(row,11,len(subdf) - test_b)
            table.write(row,12,test_b)
            row=row+1

        #print(row)
        for rows in range(startrow + 1,row) :
            table.write(rows,3,xlwt.Formula('B'+str(rows+1)+'+C'+str(rows+1)))
            table.write(rows,4,xlwt.Formula('B'+str(rows+1)+'/B'+str(row+1)))
            table.write(rows,5,xlwt.Formula('C'+str(rows+1)+'/C'+str(row+1)))
            table.write(rows,6,xlwt.Formula('E'+str(rows+1)+'/F'+str(rows+1)))
            table.write(rows,7,xlwt.Formula('LN(G'+str(rows+1)+')'))
            table.write(rows,8,xlwt.Formula('(E'+str(rows+1)+'-F'+str(rows+1)+')*H'+str(rows+1)))
            table.write(rows,9,xlwt.Formula('C'+str(rows+1)+'/D'+str(rows+1)))
            table.write(rows,10,xlwt.Formula('D'+str(rows+1)+'/D'+str(row+1)))
            
            table.write(rows,13,xlwt.Formula('L'+str(rows+1)+'+M'+str(rows+1)))
            table.write(rows,14,xlwt.Formula('L'+str(rows+1)+'/L'+str(row+1)))
            table.write(rows,15,xlwt.Formula('M'+str(rows+1)+'/M'+str(row+1)))
            table.write(rows,16,xlwt.Formula('O'+str(rows+1)+'/P'+str(rows+1)))
            table.write(rows,17,xlwt.Formula('LN(Q'+str(rows+1)+')'))
            table.write(rows,18,xlwt.Formula('(O'+str(rows+1)+'-P'+str(rows+1)+')*R'+str(rows+1)))
            table.write(rows,19,xlwt.Formula('M'+str(rows+1)+'/N'+str(rows+1)))
            table.write(rows,20,xlwt.Formula('N'+str(rows+1)+'/N'+str(row+1)))
            table.write(rows,21,xlwt.Formula('(U'+str(rows+1)+'-K'+str(rows+1)+')*LN(U'+str(rows+1)+'/K'+str(rows+1)+')'))
        table.write(row,0,'total')
        table.write(row,1,xlwt.Formula('SUM(B'+str(startrow + 2)+':B'+str(row)+')'))
        table.write(row,2,xlwt.Formula('SUM(C'+str(startrow + 2)+':C'+str(row)+')'))
        table.write(row,3,xlwt.Formula('SUM(D'+str(startrow + 2)+':D'+str(row)+')'))
        table.write(row,8,xlwt.Formula('SUM(I'+str(startrow + 2)+':I'+str(row)+')'))
        table.write(row,9,xlwt.Formula('C'+str(row+1)+'/D'+str(row+1)))
        
        table.write(row,11,xlwt.Formula('SUM(L'+str(startrow + 2)+':L'+str(row)+')'))
        table.write(row,12,xlwt.Formula('SUM(M'+str(startrow + 2)+':M'+str(row)+')'))
        table.write(row,13,xlwt.Formula('SUM(N'+str(startrow + 2)+':N'+str(row)+')'))
        table.write(row,18,xlwt.Formula('SUM(S'+str(startrow + 2)+':S'+str(row)+')'))
        table.write(row,19,xlwt.Formula('M'+str(row+1)+'/N'+str(row+1)))
        table.write(row,21,xlwt.Formula('SUM(V'+str(startrow + 2)+':V'+str(row)+')'))
        return(table, row)

def tree2woe(train_x, train_y):
    dt1 = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1)
    train_x = train_x.fillna(-9999999999)
    dt1.fit(train_x , train_y)
    p = dt1.predict_proba(train_x) 
    return([np.log(p[i][0] / p[i][1] / N_G * N_B)if p[i][1] != 0 else np.log(10) for i in range(len(train_x))])   

def key2con(key,df):    
    con = key.split(',')[1].split('<')
    if len(con) == 3:
        return(df[(df[con[1]] >= float(con[0]))&(df[con[1]] < float(con[2]))])
    try:return(df[df[con[1]] >= float(con[0])])
    except:return(df[df[con[0]] < float(con[1])])
#%%
sel_DATA = pd.read_excel('D:\\WOE\\data.xlsx')
select = sel_DATA.columns
TARGET = 'nego'

#%%
card = 'ALL_new'

df1 = sel_DATA[['NO',TARGET]]
data = sel_DATA[select]

#total good
N_G = len(sel_DATA[sel_DATA[TARGET] == 0])
#total bad
N_B = len(sel_DATA[sel_DATA[TARGET] == 1])
print(N_G, N_B)
train_y = sel_DATA[TARGET].values
data_2 = data.select_dtypes(include=['float64', 'int64'])
features = list(data_2.columns)
features.remove(TARGET)

woe_1D = {v:tree2woe(data_2.loc[: , [v,v]], train_y)for v in data_2.columns}
woe_1D[TARGET] = list(train_y)
df2 = pd.DataFrame.from_dict(woe_1D)
df2.to_excel('woe_data')
iV_1D = {var:cal_IV(var, df2)for var in df2.columns}
woe_2D = {v[0]+'_X_'+v[1]:tree2woe(data_2.loc[: , [v[0],v[1]]], train_y)for v in itertools.combinations(features, 2)}
woe_2D[TARGET] = list(train_y)
df2 = pd.DataFrame.from_dict(woe_2D)
iV_2D = {var:cal_IV(var, df2)for var in df2.columns}
iv_2D_dict = {}
for pair, iv in iV_2D.items():
    if pair == TARGET:pass
    else:
        v1, v2 = pair.split("_X_")
        if v1 in iV_1D.keys() and v2 in iV_1D.keys():
            if iv >= iV_1D[v1] + 0.1 and iv >= iV_1D[v2] + 0.1:
                iv_2D_dict.update({pair:iv})        
iv_2d_df = pd.DataFrame()
iv_2d_df['combine'] = list(iv_2D_dict.keys())
iv_2d_df['IV'] = [iv_2D_dict[k]for k in iv_2d_df['combine']]
iv_2d_df['IV_1'] = [iV_1D[k.split('_X_')[0]]for k in iv_2d_df['combine']]
iv_2d_df['IV_2'] = [iV_1D[k.split('_X_')[1]]for k in iv_2d_df['combine']]
iv_2d_df['v1'] = [k.split('_X_')[0]for k in iv_2d_df['combine']]
iv_2d_df['v2'] = [k.split('_X_')[1]for k in iv_2d_df['combine']]
iv_2d_df.to_excel(card + '_iv_2d_2.xlsx', index = False)

wb = xlwt.Workbook()
startrow = 0
table = wb.add_sheet('IV_Detial')
for var in iV_1D.keys():
    if iV_1D[var] > 0:
        v = [var, var]
        class_of_tree = clf(sel_DATA, [v[0], v[1]], TARGET)    
        table, end_row = cot2exc(class_of_tree, table, startrow)
        startrow = end_row + 2

table2 = wb.add_sheet('IV_list')
table2.write(0,0,'Feature_Name')
table2.write(0,1,'IV')
startrow = 1
for var in iV_1D.keys():
    table2.write(startrow,0,var)
    table2.write(startrow,1,iV_1D[var])
    startrow += 1
wb.save('D:\\WOE\\test_1D.xls')

wb = xlwt.Workbook()
i = 0
for var in iv_2D_dict.keys():
    v = var.split("_X_")
    class_of_tree = clf(sel_DATA, [v[0], v[1]], TARGET)
    filename = 'NO' + str(i)
    table = wb.add_sheet(filename)
    table = cot2exc(class_of_tree, table)
    i += 1
wb.save(card + '_2D.xls')
#%%
wb = xlwt.Workbook()
startrow = 0
table = wb.add_sheet('IV_Detial')
for var in iV_1D.keys():
    if iV_1D[var] > 0:
        v = [var, var]
        class_of_tree = clf(sel_DATA, [v[0], v[1]], TARGET)    
        table, end_row= cot2exc2(class_of_tree, table, startrow, sel_DATA)
        startrow = end_row + 2
        
wb.save('D:\\WOE\\test_3D.xls')
#%%
exogenous = ["B0600_WoE_value","JCU_NEWINQCNT_V2_WoE_value","JCC_DIV_BALTOLMT_L3M_WoE_value","MARITAL_STATUS_CDxGENDER_CD_WoE_value","EDUCATIONxLONGEST_CARD_WoE_value",
             "RESIDENT_STATUS_CDxB0503_WoE_value", "LONGEST_CARD_INCLUDE_STOPxB0544_WoE_value", "B0430xB0525_WoE_value", "B0575xB0527_WoE_value", "ANNSALARYxB0244_WoE_value",
             "AGE_XxB0482_WoE_value", "PAY_ALLxB0345_WoE_value", "U_CUST_LEVELxB0530_WoE_value", "ANNSALARYxOTHER_Q_TOT_WoE_value"]

nonwhite_val_regression_X = woe_1D[exogenous]
model_x = nonwhite_val_regression_X.copy().round(4)

lr = LogisticRegression(penalty = "l1", solver = "liblinear")
seed = 123457
numpy_seed = np.random.seed(seed)
params = {"C":list(np.random.uniform(low=0.0, high=5.0, size=100)), "class_weight" : [None, "balanced"]}
kfoldCV = StratifiedKFold(n_splits = 4, shuffle = True, random_state = seed)
randomcv = RandomizedSearchCV(estimator = lr, 
                            param_distributions = params, 
                            scoring = "roc_auc", 
                            refit=True,
                            n_iter = 200,
                            cv = kfoldCV,
                            return_train_score = True)
randomcv.fit(model_x, train_y)
print(randomcv.best_params_)
print("Mean Test Score : ", (x := randomcv.cv_results_).get("mean_test_score")[np.where(x.get("rank_test_score") == 1)])
print(randomcv.best_estimator_.coef_)

lr_f = LogisticRegression(penalty = "l1", solver = "liblinear", class_weight = None)
numpy_seed = np.random.seed(seed)
params_f = {"C":list(np.random.uniform(low=0, high=1.0, size=50))}
kfoldCV_f = StratifiedKFold(n_splits = 4, shuffle = True, random_state = seed)
randomcv_f = RandomizedSearchCV(estimator = lr_f, 
                                param_distributions = params_f, 
                                scoring = "roc_auc", 
                                refit=True,
                                n_iter = 50,
                                cv = kfoldCV_f,
                                return_train_score = True)
randomcv_f.fit(model_x, train_y)
print(randomcv_f.best_params_)
print("Mean Test Score : ", (x := randomcv_f.cv_results_).get("mean_test_score")[np.where(x.get("rank_test_score") == 1)])
print(randomcv_f.best_estimator_.coef_)

#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:29:05 2020

@author: 118857
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
def plot_confusion_matrix(CM, target_names,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(CM, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def CM_normalized(CM):
    CM_prob = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
    return CM_prob
random_state = 123457
clf = LogisticRegression(class_weight='balanced',
                         fit_intercept=True, 
                         penalty='l1',
                         solver='liblinear',
                         tol=0.0001
                        )
df = pd.DataFrame.from_dict(woe_1D)
X = df.loc[:, features].values
y = df.loc[:, TARGET].values
target_names = ['Bad', 'Good']
np.random.seed(random_state)
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]
X_test=X
X_train=X
y_test=y
y_train=y
fontsize = 16
plt.rcParams['axes.labelsize']  = fontsize
plt.rcParams['axes.titlesize']  = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
param_grid = {'C': np.logspace(-3, 1, 20)}
scoring = 'accuracy'
#%%
cv = 5
GridSearch_CV = GridSearchCV(estimator=clf,
                             param_grid=param_grid,
                             scoring=scoring,
                             cv=cv,
                             verbose=1,
                             n_jobs=-1)
GridSearch_CV.get_params().keys() 
GridSearch_CV.fit(X_train, y_train)
clf_best = GridSearch_CV.best_estimator_
print(clf_best)
print(GridSearch_CV.best_params_)
clf_best.fit(X_train, y_train)
print(clf_best.coef_,clf_best.intercept_)
#%%
coef_intercept = clf_best.intercept_
clf_coef = clf_best.coef_
Non_zero_parameters = (clf_coef != 0).mean()
clf_best.score(X_train, y_train)
clf_best.score(X_test, y_test)
y_test_pred = clf_best.predict(X_test)
confusion_matrix(y_test, y_test_pred)
CM = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(CM)


CM_prob = CM_normalized(CM)
print('Normalized confusion matrix')
print(CM_prob)

plt.figure(figsize=(22, 20))
plt.subplot(121)
plot_confusion_matrix(CM,target_names)
plt.subplot(122)
plot_confusion_matrix(CM_prob,target_names,title='Normalized confusion matrix')
plt.subplots_adjust(left=0.08,right=0.95,bottom=0.08,top=0.95,wspace=0.18,hspace=0.05)
plt.show()
#%%
#feature = list(df.columns[1:-1])
KF = KFold(n=len(y_train), n_folds=cv,random_state=random_state)
CV_coef = []
CV_coef_nm = []
for k, (train, validation) in enumerate(KF):
    clf_best.fit(X_train[train], y_train[train])
    CV_coef.append(clf_best.coef_*np.std(X[train], 0))
CV_ALLRank=[]
for i in range(len(CV_coef[0][0])):
    CV_Rank=[]
    for ii in range(len(CV_coef)):        
        CV_Rank.append(CV_coef[ii][0][i])
    CV_Rank.sort()
        #print CV_coef[ii][0][i] 
    #CV_ALLRank.append((CV_Rank[4]-CV_Rank[0])/CV_Rank[2])
    CV_ALLRank.append((CV_Rank[4]-CV_Rank[0]))
plt.figure(224)
plt.ylim(0, 1.4)
for k in range(len(CV_coef)):
    plt.plot(range(len(CV_coef[k][0])),CV_coef[k][0],lw=1, label='CV:%d importances' % (k+1))
plt.bar(list(map(lambda x:x-0.1, range(len(CV_ALLRank)))), CV_ALLRank,width=0.2,lw=1, label='Max-Min', color="b")
plt.title('C='+str(GridSearch_CV.best_params_['C']))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('importances')
plt.xlabel('Feature')
plt.savefig('picture\ML_'+str(random_state))
plt.show()
clf_best.fit(X_train, y_train)
importances = clf_best.coef_*np.std(X_train, 0)
importances = importances.T[:,0]
indices = np.argsort(abs(importances))[::-1]
print("Feature ranking:")
for x in range(X_train.shape[1]):
    print("%d %s %f" % (x+1, feature[indices[x]], importances[indices[x]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="b")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
#%%
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
y = white['Bad2'].values
y.sort()
pred = np.array([0]*len(white))

roc_auc_score(y, pred, average=None)
array([0.82..., 0.86..., 0.94..., 0.85... , 0.94...])
>>> from sklearn.linear_model import RidgeClassifierCV
>>> clf = RidgeClassifierCV().fit(X, y)
>>> roc_auc_score(y, clf.decision_function(X), average=None)
array([0.81..., 0.84... , 0.93..., 0.87..., 0.94...])

#%%
def oods(gb, N_B, N_G):
    n_b = gb.sum()
    n_g = len(gb) - n_b
    return(np.log(n_g / n_b / N_G * N_B)if n_b != 0 else np.log(10))
def data2woe(woe_n, TARGET):
    woe = woe_n[[TARGET]]
    feature_list = woe_n.columns
    N_B = woe[TARGET].sum()
    N_G = len(woe) - N_B
    for Var in feature_list:
        for d in set(woe_n[Var].values):
            wos_value = oods(woe_n.loc[woe_n[Var] == d, TARGET], N_B, N_G)
            woe.loc[woe_n[Var] == d, Var] = wos_value
    return(woe)               
            
test_data = pd.read_excel('D:\\資料分析\\中華 變數分組\\WHOSCALL_WHITE_APPROVAL_單變數分析_0829.xlsx')
TARGET = 'ever_m2plus_nego'
woe_test = data2woe(woe_n, TARGET)
#%%
woe_n = test_data.select_dtypes(include='object')
woe_n.loc[:, TARGET] = test_data[TARGET].values
feature_list = woe.columns

Var = 'rank'
for d in set(woe[Var].values):
    print(d, oods(woe.loc[woe[Var] == d, TARGET], N_B, N_G))
woe_n = woe[[TARGET]]
feature_list = woe.columns
N_B = woe_n[TARGET].sum()
N_G = len(woe_n) - N_B

for Var in feature_list:
    print(Var)
    for d in set(woe[Var].values):
        wos_value = oods(woe.loc[woe[Var] == d, TARGET], N_B, N_G)
        print(d , wos_value)
        woe_n.loc[woe[Var] == d, Var] = wos_value

