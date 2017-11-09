# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:18:53 2017
@author: ZhWang
"""
import os
import sys
import time
import psutil
import re
import json
import nltk
import html2text
from numpy import *
from bs4 import BeautifulSoup
from jieba import analyse
import scipy.cluster.hierarchy as sch
from collections import Counter
##############################################################################
#得到所有关键词集合与每个文本的关键词及词频
#输入为文本，输出为文本构成的关键词集合以及每一篇文本关键词、词频构成的字典
def Preparation_Data(text,num):   
    keywords=set([])
    dictionary={}
    for x, w in analyse.extract_tags(text, topK = num,
        withWeight = True):      
        #print("%s %s" % (x, w)) 
        keywords.add(x)
        dictionary[x]=w
    #print('key words is:\n',dictionary)
    return keywords,dictionary

#得到关键词构成的向量
#keywords_list为所有关键词的集合，inputwords为输入的字典
def Convert_To_Vec(keywords_list,inputwords):
    vector=[0]*len(keywords_list)
    for word in inputwords:
        if word in keywords_list:         
            vector[keywords_list.index(word)]=inputwords[word]
        else:
            print("something wrong")
    return vector

#调参函数（待改进）
#feature_num        新闻关键词数
#class_judge_value  类别选择的阈值
#proportion         新闻选择的比重
#keywords_num       最后得出的关键词个数
def Keywords_Func_Parameters(num1,num2,num3,num4):  
    feature_num=num1
    class_judge_value=num2
    proportion=num3
    keywords_num=num4
    return feature_num,class_judge_value,proportion,keywords_num

#读json文件
def Read_Json(path):
    p = open(path, "r")
    text=p.read()
    p.close()
    text_dic=json.loads(text)
    return text_dic

#找到要加权的标签
def Get_Tag(text,cur_dic,tag_name):
    #去掉注释标签############
    rule_special=re.compile('<!--[^>]*-->',re.S|re.I)
    text=rule_special.sub('',text)
    soup = BeautifulSoup(text,"lxml")
    #print(soup.prettify())    #以标签形式输出内容
    #获取特定的标签内容######
    tag=soup.findAll({tag_name})    #选择不同的标签
    tag_num=len(tag)
    word_tag_a=[]
    for i in range(tag_num):
        word_tag_a.append(tag[i].string)
    #print('contents <a></a> are\n',word_tag_a)
    cur_dic[tag_name]=word_tag_a
    return 

#加权参数函数(待改进)
def Set_Weightest(char,num1,num2,num3):
    weigh_dic={}
    if char=='open':
        weigh_dic['ctrl_char']='open'
        weigh_dic['title']=num1
        weigh_dic['a']=num2
        weigh_dic['strong']=num3
        return weigh_dic
    else:
        weigh_dic['ctrl_char']='close'
        return weigh_dic

#文本加权函数   
def Get_Weightest(path,weigh_dic):
    if weigh_dic['ctrl_char']=='close':
        return ''      #状态为关则不加权
    text_dic=Read_Json(path)    
    if not (text_dic['article_html']): 
        return ''   
    cur_dic={}        #用来记录待加权的信息    
    cur_time=''
    cur_title=''
    cur_time=text_dic['time']
    #print('time is: ',cur_time)
    cur_title=text_dic['title']
    #print('title is: ',cur_title)
    cur_dic['title']=cur_title
    text=text_dic['article_html']
    #print(text)
    Get_Tag(text,cur_dic,'a')
    Get_Tag(text,cur_dic,'strong')
    #print(cur_dic)
    text_extra=''
    text_extra=text_extra + cur_dic['title'] * weigh_dic['title']
    newlist_a =  [a for a in cur_dic['a'] if a not in ['',None]]
    #print(newlist_a)
    for word in newlist_a:
        text_extra=text_extra + word * weigh_dic['a']
    newlist_strong =  [a for a in cur_dic['strong'] if a not in ['',None]]
    #print(newlist_strong)
    for word in newlist_strong:
        text_extra=text_extra + word * weigh_dic['strong']
    return text_extra

#提取纯文本函数(过滤方法可扩充)     
def Get_Initialtext(path):
    text_dic=Read_Json(path)   
    if not (text_dic['article_html']): 
        return ''   
    text=text_dic['article_html']    
    text_maker=html2text.HTML2Text()
    text_maker.ignore_links=True
    text_maker.ignore_images=True
    text_maker.ignore_tables=True
    text = text_maker.handle(text)   
#    rule_all = re.compile(r'<[^>]+>',re.S)
#    text = rule_all.sub('',text)
    return text

#聚类函数
def Cluster_Function(matrix_input,dis_par,class_judge_value):
    disMat = sch.distance.pdist(matrix_input,dis_par)  
    #print('定量化相关性的向量为：\n',disMat)
    Z=sch.linkage(disMat,method='average') 
    #将结果展示并保存###########
    #P=sch.dendrogram(Z)
    cluster= sch.fcluster(Z, class_judge_value, 'inconsistent')
    cluster=list(cluster)
    #print ("Original cluster by hierarchy clustering:\n",cluster)
    class_num=(max(cluster))       #分成的类别的数量
    print('聚类总数为：',class_num)
    common_value = Counter(cluster)
    common_value_list=sorted(common_value.items(),key=lambda e:e[1],reverse=True)   
    return cluster,common_value_list       #得到从大到小排序后的结果

#选足够整数个类别策略
def Find_News(cluster,common_value_list,para):
    select_num=int(len(cluster)*para)          
    #print(select_num)
    i=0
    this_tuple=common_value_list[0]
    class_tot=[this_tuple[0]]
    select_sum=this_tuple[1]          #初始值为首个元组所存的值
    while(select_sum<select_num):
        i=i+1
        this_tuple=common_value_list[i]
        select_sum=select_sum+this_tuple[1]
        class_tot.append(this_tuple[0])        
    print('需要选择的新闻类型为',class_tot,'   总数为',select_sum)
    #根据上面的结果得到新闻序号#   
    i=0
    j=0 
    select_news=[]  
    while(i<select_sum and j<totnum):        #直到找到所有待选新闻为止 
        this_news=cluster[j]           
        while(this_news not in  class_tot):
            j=j+1        
            this_news=cluster[j]
        select_news.append(j)
        j=j+1  
        i=i+1 
    #print(select_news)  
    return select_news

#找到最终的关键词
def Get_Finalkeywords(cur_files_path,filenames,select_news,keywords_num,weigh_dic):
    alltext=''
    for m in select_news:
        text=Get_Initialtext(cur_files_path+filenames[m])+Get_Weightest(cur_files_path+filenames[m],weigh_dic)    
        if text=='':
            continue    
        alltext=alltext+text
    #print(sys.getsizeof(alltext))
    #提取筛选出新闻的关键词####
    words_segment,keywords_dict=Preparation_Data(alltext,keywords_num)
    keywords_list=list(keywords_dict.keys())
    print('关键词列表为：\n ',keywords_list)
    return keywords_list

#主题匹配
def Match_Themes(keywords_list,path,filenames):
    #字典初始化###############
    themes_dict={}
    for word in keywords_list:
        themes_dict[word]=0
    themes_words=['概念','主题','模式']                 #可拓展
    for fn in filenames:
        text=Get_Initialtext(path+fn)  
        print('######################$$$$$$$$$$$$$$$$$$$$\n',text) #查看纯文本
        if text=='':
            continue        
        for a in keywords_list:
            ci_outer=0
            for b in themes_words:
                ci_inner=len(re.findall(a+b,text))
                ci_outer=ci_outer+ci_inner
            themes_dict[a]=themes_dict[a]+ci_outer
    #print(themes_dict)
    return themes_dict

##############################################################################
print('###################程序开始时间为： ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
cur_files_path="E:\\ZhWang\\test_file\\samples3\\"     #路径有待更改
#参数设置####################
feature_num,class_judge_value,proportion,keywords_num=Keywords_Func_Parameters(100,1,0.2,30)
print('新闻向量的维数：',feature_num,'  聚类阈值：',class_judge_value)
print('筛选比重：','%.2f%%' % ((proportion) * 100),'  选取关键词个数：',keywords_num)
#新闻权重设置################
weigh_dic=Set_Weightest('open',10,5,5)
#遍历所有新闻################
for dirpaths,dirnames,filenames in os.walk(cur_files_path):
    pass 
#print (filename)
words_collection=set([])
totlist=[]
totnum=0
#外循环#####################
for fn in filenames: 
    text=Get_Initialtext(cur_files_path+fn)+Get_Weightest(cur_files_path+fn,weigh_dic)    
    if text=='':
        continue   
    #文本加权后的操作#######
    words_segment,words_dict=Preparation_Data(text,feature_num)
    words_collection=words_collection|words_segment
    totlist.append(words_dict)
    totnum=totnum+1                 #有多少个文件就循环多少次
words_collection=list(words_collection)
#print('key words in this material are:\n',words_collection)
print('###################成功遍历文本并提取关键词时间为： ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))   
matrix_input=zeros([totnum,len(words_collection)])
for i in range(0,totnum):
    vector=Convert_To_Vec(words_collection,totlist[i])
    matrix_input[i]=vector    
#print('各向量组成的矩阵为：\n',matrix_input)
print(sys.getsizeof(matrix_input))
print('###################成功构建矩阵时间为： ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print('新闻总数和关键词集合分别为：',shape(matrix_input))
#层次聚类##################
cluster,common_value_list=Cluster_Function(matrix_input,'cosine',class_judge_value)
print('###################成功聚类时间为： ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
#根据类别的结果做筛选#######
select_news=Find_News(cluster,common_value_list,proportion)
#将得到最终的一组关键词#####
keywords_list=Get_Finalkeywords(cur_files_path,filenames,select_news,keywords_num,weigh_dic)
#从关键词中确定主题########
themes_dict=Match_Themes(keywords_list,cur_files_path,filenames)
print('###################程序结束时间为： ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
#查看内存占用情况##########
info = psutil.virtual_memory()
print (u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
#print (u'总内存：',info.total)
#print (u'内存占比：',info.percent)





