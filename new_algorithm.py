# coding=UTF-8
import os
import sys
import math
import numpy as np
import boto3
import threading
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql import Window

#接收参数并初始化各种列表
file1_str = "12aa-001.%08,12aa-001.%22"
file1_list = file1_str.split(",")

file2_str = "s3://lencef/test03/12la-001.$08,s3://lencef/test03/12la-001.$22"
file2_list = file2_str.split(",")

val_str = "Blade 1 pitch angle,Blade 1 ram length,Blade 1 pitch actuator force||Blade root 1 Mx,Blade root 1 My,Blade root 1 Fx,Blade root 1 Fy"
val_list = val_str.split("||")

new_val_str = "MxDisk1,MyDisk1,MxyDisk1,FxDisk1,FyDisk1,FxyDisk1,PFDisk1x,PFDisk1y"
new_val_list = new_val_str.split(",")

type_list = [] #存储解析$二进制文件用到的数据格式
schema_list = [] #存储构建Dataframe用到的schema信息
val_num = [] #存储每个$文件对应的变量个数
df_list = [] #存储Dataframe，一个$文件一个Dataframe

BUCKET_NAME = 'lencef' # replace with your bucket name
s3 = boto3.resource('s3',region_name='cn-north-1')

#开始解析$二进制文件构建Dataframe
for file_name in file1_list:
  KEY = file_name # replace with your object key
  lines = s3.Bucket(BUCKET_NAME).Object(KEY).get()['Body'].read().split("\n")
  type_str = ""
  sField =[]
  for line in lines:
    if line.find("VARIAB") == -1:
      continue
    else:
      tmpval = line[7:].strip("\r\n").split("\' \'")
      val_num.append(len(tmpval))
  for i in tmpval:
    j = i.replace('\'','')
    type_str = type_str + "f4,"
    sField.append(StructField(j.strip(),FloatType(),False))
  dtype = np.dtype(type_str[:-1])
  schema = StructType(sField)
  type_list.append(dtype)
  schema_list.append(schema)

sc = SparkContext()
sqlContext= SQLContext(sc)
for index in range(len(file2_list)):
    frdd = sc.binaryFiles(file2_list[index])
    
    def read(rdd):
        array=np.frombuffer(bytes(rdd[1]),dtype=type_list[index]) 
        array=array.newbyteorder().byteswap() # big Endian
        return array.tolist()
    
    unzipped=frdd.flatMap(read)
    df=sqlContext.createDataFrame(unzipped,schema_list[index])
#    df=df.coalesce(1)
    df_list.append(df)

#拼接Dataframe，将所有$文件对应的Dataframe拼接成一个，为后续计算做准备
for index in range(len(df_list)):
     df_list[index] = df_list[index].select(val_list[index].split(",")).withColumn("row_number",monotonically_increasing_id())
for index in range(len(df_list)): 
    if 2 * index < len(df_list):
     if index == 0:
        DF = df_list[index].join(df_list[index + 1],df_list[index].row_number == df_list[index + 1].row_number,'inner')
     else:
        DF = DF.join(df_list[index + 1],DF.row_number == df_list[index + 1].row_number,'inner')
    else:
     break
DF = DF.drop("row_number")
NDF = DF.withColumnRenamed('Blade 1 pitch angle','$1')\
        .withColumnRenamed('Blade 1 pitch actuator force','$2')\
        .withColumnRenamed('Blade root 1 Mx','$3')\
        .withColumnRenamed('Blade root 1 My','$4')\
        .withColumnRenamed('Blade root 1 Fx','$5')\
        .withColumnRenamed('Blade root 1 Fy','$6')\
        .withColumnRenamed('Blade 1 ram length','$7')
#print("=============================================================================================>>count the row number: %i" % DF.count())

#计算过程
# #1 = $3 * COS( $1 ) - $4 * SIN( $1 )
# #2 = $3 * SIN( $1 ) + $4 * COS( $1 )
# #3 = SQRT(  #1 * #1 + #2 * #2 )
# #4 = $5 * COS( $1 ) - $6 * SIN( $1 )
# #5 = $5 * SIN( $1 ) + $6 * COS( $1 )
# #6 = SQRT(  #4 * #4 + #5 * #5 )
# #7 = $1 + 60.98 / 180 * pi
# #8 = $1 + 60.975 / 180 * pi
# #9 = 1.75 * SIN( #8 ) / $7
# #10 = IF( #8 <= pi / 2 ? SQRT( 1 - #9 * #9 ) ! 0 - SQRT( 1 - #9 * #9 ) )
# #11 = $2
# #12 = #11 / 2 * #10
# #13 = 0 - #11 / 2 * #9
def modify(r):
    if r <= math.pi/2:
        return 1.0
    else:
        return 2.0
change = udf(modify,FloatType())

NDF1 = NDF.withColumn('MxDisk1',NDF['$3']*cos(NDF['$1']) - NDF['$4']*sin(NDF['$1']))\
        .withColumn('MyDisk1',NDF['$3']*sin(NDF['$1']) + NDF['$4']*cos(NDF['$1']))

NDF2 = NDF1.withColumn('MxyDisk1',sqrt(NDF1['MxDisk1']*NDF1['MxDisk1'] + NDF1['MyDisk1']*NDF1['MyDisk1']))\
         .withColumn('FxDisk1',NDF1['$5']*cos(NDF1['$1']) - NDF1['$6']*sin(NDF1['$1']))\
         .withColumn('FyDisk1',NDF1['$5']*sin(NDF1['$1']) + NDF1['$6']*cos(NDF1['$1']))

NDF3 = NDF2.withColumn('FxyDisk1',sqrt(NDF2['FxDisk1']*NDF2['FxDisk1'] + NDF2['FyDisk1']*NDF2['FyDisk1']))

NDF4 = NDF3.withColumn('tmp7',NDF3['$1'] + 60.98/180*math.pi)\
           .withColumn('tmp8',NDF3['$1'] + 60.975/180*math.pi)

NDF5 = NDF4.withColumn('tmp9',1.75*sin(NDF4['tmp8'])/NDF4['$7'])

NDF6 = NDF5.withColumn('tmp',change(NDF5['tmp8']))
NDF7 = NDF6.withColumn('tmp10',when(NDF6['tmp'] == 1.0,sqrt(1-NDF6['tmp9']*NDF6['tmp9'])).otherwise(0-sqrt(1-NDF6['tmp9']*NDF6['tmp9'])))

NDF8 = NDF7.withColumn('PFDisk1x',NDF7['$2']/2*NDF7['tmp10'])\
         .withColumn('PFDisk1y',0-NDF7['$2']/2*NDF7['tmp9'])

#将Dataframe注册为临时表，使用SQL语句筛选出新的变量,结果以csv文件的形式存储在HDFS之上
RDF1 = NDF8.registerTempTable("t1")
RDF2 = sqlContext.sql("select " + new_val_str  + " from t1") 

def writetofile(x):
  a = open("/tmp/series.csv", "a")
  a.write(x)
def writetofile2(x):
  b = open("/tmp/extreme.csv","a")
  b.write(x)

list1 = RDF2.rdd.collect()

for s in list1:
 l3 = ''
 l1 = str(s).split(',')
 for index in range(len(l1)):
    l2 = l1[index].split('=')
    if index == len(l1)-1:
     l3 = l3 + l2[1].replace(')','')
    else:
      l3 = l3 + l2[1] + ','
 writetofile(l3 + "\n")

def writetofile3(lt):
  for s in lt:
    l3 = '' 
    l1 = str(s).split(',')
    for index in range(len(l1)):
      l2 = l1[index].split('=')
      if index == len(l1)-1:
        l3 = l3 + l2[1].replace(')','')
      else:
        l3 = l3 + l2[1] + ',' 
    writetofile2(l3 + "\n")


data = np.loadtxt('/tmp/series.csv',delimiter=',')
np.save('/tmp/series.npy',data)
updata = open('/tmp/series.npy','rb')
file_obj = s3.Bucket(BUCKET_NAME).put_object(Key='series.npy',Body=updata)
os.remove("/tmp/series.csv")
os.remove("/tmp/series.npy")


def max(df,varstr):
  df.registerTempTable("t2")
  list_tmp_1 = sqlContext.sql("select  * from t2 order by " + varstr + " desc limit 1").rdd.collect()
  writetofile3(list_tmp_1)
def min(df,varstr):
  df.registerTempTable("t2")
  list_tmp_2 = sqlContext.sql("select  * from t2 order by " + varstr + " desc limit 1").rdd.collect()
  writetofile3(list_tmp_2)


#将筛选后的结果注册为临时表，使用SQL语句计算出新变量的最大、最小值，结果以csv文件的形式存储在HDFS之上
for index in range(len(new_val_list)):
    t1 = threading.Thread(target=max, args=(RDF2,new_val_list[index]))
    t1.start()
    t2 = threading.Thread(target=min, args=(RDF2,new_val_list[index]))
    t2.start()


list_tmp_3 = sqlContext.sql("select " + "mean(" + new_val_list[0] + ")"\
               + ",mean(" + new_val_list[1] + ")"\
               + ",mean(" + new_val_list[2] + ")"\
               + ",mean(" + new_val_list[3] + ")"\
               + ",mean(" + new_val_list[4] + ")"\
               + ",mean(" + new_val_list[5] + ")"\
               + ",mean(" + new_val_list[6] + ")"\
               + ",mean(" + new_val_list[7] + ") from t2").rdd.collect()
writetofile3(list_tmp_3)
  
updata2 = open('/tmp/extreme.csv','r')
file_obj2 = s3.Bucket(BUCKET_NAME).put_object(Key='ext.csv',Body=updata2)
os.remove("/tmp/extreme.csv")
