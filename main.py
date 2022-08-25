import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns


df = pd.read_csv('Resources/Copy of For Project 1.csv')

df.replace('Varies with device',np.nan)
df=df.dropna()
# rows_count1 = df.shape[0]
# print(rows_count1)
df2=df
SizeList=df2['Size'].values.tolist()
SizeList2=[]
# print(SizeList)

def change(val):
    if val == 'Varies with device' or val == '1,000+':
        return 'None'
    text=val.removesuffix('k')
    text=text.removesuffix('K')
    if "M" in text:
        text=text.removesuffix('M')
        txtnum = float(text)
        text=str(txtnum*1000)
    return float(text)

for i in range(len(SizeList)):
    SizeList2.append(change(SizeList[i]))

# df2.drop("Size", axis=1, inplace=True)

df2['Size']=SizeList2


df2 = df2.replace(to_replace='None', value=np.nan).dropna()

ReviewL=df2['Reviews'].values.tolist()
res = [int(i) for i in ReviewL]
# df2['Reviews']=res

for i in range(0, len(ReviewL)):
    ReviewL[i] = int(ReviewL[i])
# print(ReviewL)
df2['Reviews']=ReviewL

InstallsL=df2['Installs'].values.tolist()
InstallsNew=[]

def removeplus(val):
    val=val.replace('+','')
    val=val.replace(',', '')
    return int(val)
# print(removeplus("100,000+"))
for i in range(len(InstallsL)):
    InstallsNew.append(removeplus(InstallsL[i]))
# print(InstallsNew)
df2['Installs']=InstallsNew

def removedollar(val):
    val = val.replace('$', '')
    return float(val)

PriceL=df2['Price'].values.tolist()

for i in range(len(PriceL)):
    PriceL[i]=removedollar(PriceL[i])

df2['Price']=PriceL



TypeL=df2['Rating'].values.tolist()

for i in range(len(PriceL)):
    if TypeL[i]=="Free":
        PriceL[i]=0

df2['Price']=PriceL

def goodRating(val):
    if val<0 or val>5:
        return 'None'
    return val


RatingL=df2['Rating'].values.tolist()
for i in range(len(RatingL)):
    RatingL[i]=goodRating(RatingL[i])
df2['Rating']=RatingL
df2 = df2.replace(to_replace='None', value=np.nan).dropna()



count=0

for i in range(len(res)):
    if res[i]>InstallsNew[i]:
        res[i]='None'
        count+=1
# print("Count: ",count)
df2['Reviews']=res
df2 = df2.replace(to_replace='None', value=np.nan).dropna()



for (columnName, columnData) in df2.iteritems():
    print('Column Name : ', columnName)
    print('Column Contents : ', columnData.values)


# x1=df2.boxplot(by ='App', column =['Price'], grid = False)

# plt.boxplot(PriceL)
# plt.show()
# p2=plt.boxplot(res)
# plt.show(p2)
# print(res)

# figure size



# *****************************************
# OUTLIERS FOR REVIEWS


arr1=np.array(ReviewL)
# finding the 1st quartile
q1 = np.quantile(arr1, 0.25)

# finding the 3rd quartile
q3 = np.quantile(arr1, 0.75)
med = np.median(arr1)

# finding the iqr region
iqr = q3-q1
# finding upper and lower whiskers
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
print("Reviews : iqr = {} , Upper-bound = {} , Lower Bound={}".format(iqr, upper_bound, lower_bound))

# *****************************************
# OUTLIERS FOR PRICE

arr2=np.array(PriceL)
# finding the 1st quartile
q11 = np.quantile(arr2, 0.25)

# finding the 3rd quartile
q33 = np.quantile(arr2, 0.75)
med = np.median(arr2)

# finding the iqr region
iqr2 = q33-q11
# finding upper and lower whiskers
upper_bound1 = q33+(1.5*iqr2)
lower_bound1 = q11-(1.5*iqr2)
print("Price : iqr = {} , Upper-bound = {} , Lower Bound={}".format(iqr2, upper_bound1, lower_bound1))


#figure size
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
#
# # Plot the dataframe
# ax = df2[['Reviews','Price']].plot(kind='box', title='boxplot')
# # Display the plot
# plt.show()
fig = plt.figure()
fig.suptitle('Box plot of Price', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(df2['Price'].values.tolist())

ax.set_ylabel('Price of Apps')
plt.show()

# figtry = plt.figure(figsize=(10, 7))
#
# # Creating plot
# plt.boxplot(df2['Reviews'].values.tolist())
#
# # show plot
# plt.show()
fig = plt.figure()
fig.suptitle('Box plot of Reviews', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(df2['Reviews'].values.tolist())

ax.set_ylabel('Reviews for Apps')
plt.show()

# Creating histogram
fig1, ax1 = plt.subplots(figsize=(7, 5))
# fig1 = plt.figure()
fig1.suptitle('Histogram of Ratings', fontsize=14, fontweight='bold')
# ax1 = fig.add_subplot(111)
ax1.hist(RatingL, bins=[1, 2, 3, 4, 5])
ax1.set_ylabel('Ratings for Apps')
# Show plot
plt.show()

# print(df2['Size'].max(),df2['Size'].min())
SizeL=df2['Size'].values.tolist()
SizeRange=[]
for i in range(1,101):
    SizeRange.append(i*1000)

# Creating histogram
fig2, ax2 = plt.subplots(figsize=(7, 5))
fig2.suptitle('Histogram of Size', fontsize=14, fontweight='bold')
ax2.hist(SizeL, bins=SizeRange)
ax2.set_ylabel('Size of Apps')
# Show plot
plt.show()

# print(df2['Price'].max(),df2['Price'].min())
# print(df2.shape[0])
ModifiedPrice=df2['Price'].values.tolist()
for i in range(len(ModifiedPrice)):
    if ModifiedPrice[i]>200:
        ModifiedPrice[i] = 'None'
df2['Price']=ModifiedPrice
df2 = df2.replace(to_replace='None', value=np.nan).dropna()
# print(df2.shape[0])


ModifiedReview=df2['Reviews'].values.tolist()
for i in range(len(ModifiedReview)):
    if ModifiedReview[i]>2000000:
        ModifiedReview[i] = 'None'
df2['Reviews']=ModifiedReview
# print(df2.shape[0])
df2 = df2.replace(to_replace='None', value=np.nan).dropna()
# print(df2.shape[0])

# figtry = plt.figure(figsize=(10, 7))
#
# # Creating plot
# plt.boxplot(df2['Installs'].values.tolist())
#
# # show plot
# plt.show()
# print(df2.shape[0])
InstallsModified=df2['Installs'].values.tolist()
for i in range(len(InstallsModified)):
    if InstallsModified[i]>200000000:
        InstallsModified[i]='None'
df2['Installs']=InstallsModified
df2 = df2.replace(to_replace='None', value=np.nan).dropna()
# print(df2.shape[0])

# fig = plt.figure()
# fig.suptitle('box box box', fontsize=14, fontweight='bold')
#
# ax = fig.add_subplot(111)
# ax.boxplot(df2['Installs'].values.tolist())
#
# ax.set_ylabel('ylabel')
# plt.show()


# BIVARIATE ANALYSIS
fig=plt.figure()
fig.suptitle('Rating vs. Price', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
plt.scatter(df2['Rating'], df2['Price'])
ax.set_ylabel('Price')
ax.set_xlabel('Rating')
plt.show()
# ------------------
fig=plt.figure()
fig.suptitle('Rating vs. Size', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
plt.scatter(df2['Rating'], df2['Size'])
ax.set_ylabel('Size')
ax.set_xlabel('Rating')
plt.show()
# ------------------------
fig=plt.figure()
fig.suptitle('Rating vs. Reviews', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
plt.scatter(df2['Rating'], df2['Reviews'])
ax.set_ylabel('Reviews')
ax.set_xlabel('Rating')
plt.show()
# --------------------
sns.boxenplot(x="Rating", y="Content Rating", data=df2)
plt.show()
# fig=plt.figure()
# fig.suptitle('Rating vs. Content Rating', fontsize=14, fontweight='bold')
# ax = fig.add_subplot(111)
# plt.scatter(df2['Rating'], df2['Content Rating'])
# ax.set_ylabel('Content Rating')
# ax.set_xlabel('Rating')
# plt.show()
# -----------------------
sns.boxplot(x='Rating',y='Category',data=df2)
plt.show()
# fig=plt.figure()
# fig.suptitle('Rating vs. Category', fontsize=14,fontweight='bold')
# fig.set_figwidth(8)
# fig.set_figheight(5)
# ax = fig.add_subplot(111)
# plt.scatter(df2['Rating'],df2['Genres'])
# ax.set_ylabel('Category')
# ax.set_xlabel('Rating')
# plt.show()

# data preprocessing
inp1=df2.copy()
inp1.drop(columns = { 'App','Last Updated','Current Ver','Android Ver'},inplace=True)
#Apply log transformation (np.log1p) to Reviews and Installs.
inp1['Reviews'] = np.log1p(inp1['Reviews'])
inp1['Installs'] = np.log1p(inp1['Installs'])
print(inp1)

# Get dummy columns for Category, Genres, and Content Rating. This needs to be done as the models do not understand categorical data, and all data should be numeric.
dum_cols = ['Category','Genres','Content Rating']
inp2 = pd.get_dummies(inp1,columns=dum_cols,drop_first=True)
print(inp2)
inp2.pop('Type')


# ------------------------
#Train test split and apply 70-30 split
df_train, df_test = train_test_split(inp2, train_size = 0.7, random_state = 100)
y_train = df_train.Rating
X_train = df_train
y_test = df_test.Rating
X_test = df_test
# -----------------------------

lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred= lr.predict(X_train)
print(r2_score(y_train, y_train_pred)) #R2 of 1 indicates that the regression predictions perfectly fit the data.
y_test_pred= lr.predict(X_test)
print(r2_score(y_test, y_test_pred))