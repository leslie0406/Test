# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:08:24 2022

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
    #%%
d1 = pd.read_excel('D:\\資料分析\\1\\TOTAL_MEI_20220622.xlsx')
#d2 = pd.read_excel('D:\\資料分析\\1\\TABLE5.xlsx')
#d3 = d1.merge(d2, on = "APPLNO", how = 'left')

df = pd.read_excel('TABLE6.xlsx')

TARGET = 'Bad2'
df1 = df[['APPLNO',TARGET]]
data = df.copy()

#total good
N_G = len(df[df[TARGET] == 0])
#total bad
N_B = len(df[df[TARGET] == 1])
print(N_G, N_B)
train_y = df[TARGET].values
data_2 = data.select_dtypes(include=['float64', 'int64'])
features = list(data_2.columns)
features.remove('APPLNO')
features.remove('Bad2')
features.remove('approp_ymd')
features.remove('appl_ymd')
woe_1D = {v:tree2woe(data_2.loc[: , [v,v]], train_y)for v in features}
woe_1D['Bad2'] = list(train_y)
df2 = pd.DataFrame.from_dict(woe_1D)
iV_1D = {var:cal_IV(var, df2)for var in df2.columns}

wb = xlwt.Workbook()
startrow = 2
table = wb.add_sheet('sheet1')
for var in iV_1D.keys():
    v = [var, var]
    class_of_tree = clf(df, [v[0], v[1]], TARGET)    
    table, end_row = cot2exc(class_of_tree, table, startrow)
    startrow = end_row + 2
wb.save('1D_compare.xls')
#%%
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
            if iv >= iV_1D[v1]  and iv >= iV_1D[v2] :
                iv_2D_dict.update({pair:iv})        
iv_2d_df = pd.DataFrame()
iv_2d_df['combine'] = list(iv_2D_dict.keys())
iv_2d_df['IV'] = [iv_2D_dict[k]for k in iv_2d_df['combine']]
iv_2d_df['IV_1'] = [iV_1D[k.split('_X_')[0]]for k in iv_2d_df['combine']]
iv_2d_df['IV_2'] = [iV_1D[k.split('_X_')[1]]for k in iv_2d_df['combine']]
iv_2d_df['v1'] = [k.split('_X_')[0]for k in iv_2d_df['combine']]
iv_2d_df['v2'] = [k.split('_X_')[1]for k in iv_2d_df['combine']]
iv_2d_df.to_excel('iv_2d_2.xlsx', index = False)


wb = xlwt.Workbook()
i = 0
iv_2D_list = pd.read_excel('iv_2d_2.xlsx')
startrow = 2
table = wb.add_sheet('sheet1')
for var in iv_2D_list['combine']:
    v = var.split("_X_")
    class_of_tree = clf(df, [v[0], v[1]], TARGET)    
    table, end_row = cot2exc(class_of_tree, table, startrow)
    startrow = end_row + 2
    i += 1
wb.save('2D.xls')
#%%
def cc_last_limit(x):
    if str(x) == 'nan':return(np.nan)
    elif x <=50:return('46K以下')
    elif x <=70:return('67K以下')
    else:return('大於67K')
def cc_min_limit(x):
    if str(x) == 'nan':return(np.nan)
    elif x <=70:return('小於70K')
    elif x <=100:return('小於100K')
    else:return('大於101K')    
df['cc_last_limit_type'] = [cc_last_limit(x)for x in df['cc_last_limit']]
df['cc_min_limit_type'] = [cc_min_limit(x)for x in df['cc_min_limit_1y']]
d1['APPLNO'] = [int(app)for app in d1.APPLNO]
df = df.merge(d1, how = 'left', on = 'APPLNO')
df.to_excel('table7.xlsx')


#%%
l7w = df[df.cc_last_limit <= 70]
d1 = pd.read_excel('D:\\資料分析\\1\\新增資料夾 (2)\\data.xlsx')
df = d1[d1.APPLNO.isin(l7w.APPLNO)]

Bad = pd.read_excel('Bad2.xlsx')
df['Bad2'] = [Bad[Bad.APPLNO == a]['Bad2'].values[0] for a in df.APPLNO]
df.to_excel('cc_last_limit_70k.xlsx')
#%%
def Annsalary(x):
    if x <= 300000:return('年收三十萬以下')
    else:return('年收大於三十萬')
def B0244(x):
    if x >=0.35:return('分期付款比0.35以上')
    else:return('分期付款比小於0.35')
def B0576(x):
    if x <=15000:return('一年期總消費金額一萬五以下')
    else:return('一年期總消費金額大於一萬五')
def B0523(x):
    if x >0.97:return('額度使用率大於97%')
    else:return('額度使用率97%以下')
def JCU_NewInqCnt(x):
    if x >4:return('最近三個月被銀行查詢新業務次數五次以上')
    else:return('最近三個月被銀行查詢新業務次數小於五次')   
def other_Q_new(x):
    if x >1:return('最近三個月被銀行查詢新業務次數(不含048)2次以上')
    else:return('最近三個月被銀行查詢新業務次數(不含048)小於2次')
    
f1 = d1[['APPLNO','Bad2']]
f1['Annsalary'] = [Annsalary(x)for x in d1.Annsalary]
f1['B0244'] = [B0244(x)for x in d1.B0244]
f1['B0576'] = [B0576(x)for x in d1.B0576]
f1['B0523'] = [B0523(x)for x in d1.B0523]
f1['JCU_NewInqCnt'] = [JCU_NewInqCnt(x)for x in d1.JCU_NewInqCnt]
f1['other_Q_new'] = [other_Q_new(x)for x in d1.other_Q_new]
f1.to_excel('flag.xlsx')
