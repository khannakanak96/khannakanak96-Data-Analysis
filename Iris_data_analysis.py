
# 1. load data into Pandas
import pandas as pd
df = pd.read_csv("C:/Users/carlo/OneDrive/Documents/Documentos/CJ/NYU Teaching/Data Engineering/Codes/data/iris.csv")
# ---

# 2. sanity check with Pandas
print("shape of data in (rows, columns) is " + str(df.shape))
print(df.head())
print(df.describe().transpose())
# ---

# 3. explore with Seaborn pairplot
import seaborn as sns
sns.pairplot(df,hue='species')
# ---

# 4. add histograms to diagonals of Seaborn pairplot
sns.pairplot(df,hue='species',diag_kind='hist',
             palette='bright',markers=['o','x','v'])
# ---

# 5. plot bivariate scatter with Seaborn
sns.lmplot(x='petal length in cm', y='petal width in cm', 
           hue="species", data=df, fit_reg=False,
           palette='bright',markers=['o','x','v'])
# ---

# 6. plot bivariate scatter with Seaborn
sns.lmplot(x='petal length in cm', y='petal width in cm', 
           hue="species", data=df, fit_reg=False,
           palette='bright',markers=['o','x','v'])
# ---

# 7. violin plot
# Violin plots are similar to box plots,  except that
# they also show the probability density of the data at different values, 
# usually smoothed by a kernel density estimator
sns.violinplot(x='species',y='petal length in cm', data=df)
# ---

# 8. reduce dimensions with PCA with Pandas DataFrame
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
out_pca = pca.fit_transform(df[['sepal length in cm',
                                'sepal width in cm',
                                'petal length in cm',
                                'petal width in cm']])

df_pca = pd.DataFrame(data = out_pca, columns = ['pca1', 'pca2'])
print(df_pca.head())

# This looks good, but we are missing the target or label column (species). 
# Let's add the column by concatenating with the original DataFrame. 
# This gives us a PCA DataFrame (df_pca) ready for downstream work and predictions. 
# Then, let's plot it and see what our transformed data looks like in two dimensions.

df_pca = pd.concat([df_pca, df[['species']]], axis = 1)
print(df_pca.head())
sns.lmplot(x="pca1", y="pca2", hue="species", data=df_pca, fit_reg=False)

# IMPORTANT NOTE:
# Ignoring labels in the transformation step can be desirable 
# for some problem statements (especially those with unreliable class labels) 
# to avoid pulling the reduced component vectors in an unhelpful direction. 
# For this reason, *** I RECOMMEND THAT YOU ALWAYS START WITH PCA ***
# before deciding whether you need to do any further work or not. In general, 
# the computation time for PCA is short,so there's no harm in starting here. 
# ---

# 9. reduce dimensions with LDA - nothing else than a PCA with labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)

# format dataframe
out_lda = lda.fit_transform(X=df.iloc[:,:4], y=df['species'])
df_lda = pd.DataFrame(data = out_lda, columns = ['lda1', 'lda2'])
df_lda = pd.concat([df_lda, df[['species']]], axis = 1)

# sanity check
print(df_lda.head())

# plot
sns.lmplot(x="lda1", y="lda2", hue="species", data=df_lda, fit_reg=False)

# The goal of PCA is to orient the data in the direction of the greatest variation. 
# However, it ignores some important information from our dataset – for instance, the labels are not used; 
# in some cases, we can extract even better transformation vectors if we include the labels. 
# The most popular labeled dimension-reduction technique is called linear discriminant analysis (LDA). 

# 10. comparison PCA vs. LDA - run separately 
sns.violinplot(x='species',y='pca1', data=df_pca).set_title("Violin plot: Feature = PCA_1")
sns.violinplot(x='species',y='lda1', data=df_lda).set_title("Violin plot: Feature = LDA_1")
# ---

# 11. k-means clustering and the silhouette score
# cluster With k-means and check silhouette score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# initialize k-means algo object
kmns = KMeans(n_clusters=3, random_state=42)

# fit algo to PCA and find silhouette score
out_kms_pca = kmns.fit_predict(out_pca)
silhouette = silhouette_score(out_pca, out_kms_pca)
print("PCA silhouette score = " + str(silhouette))

# fit algo to LDA and find silhouette score
out_kms_lda = kmns.fit_predict(out_lda)
silhouette = silhouette_score(out_lda, out_kms_lda)
print("LDA silhouette score = %2f " % silhouette)

# The silhouette value is a measure of how similar an object is 
# to its own cluster (cohesion) compared to other clusters (separation). 
# The silhouette ranges from −1 to +1, where a high value indicates that the object 
# is well matched to its own cluster and poorly matched to neighboring clusters.

# ---

# 12. making decisions
# IMPORTANT NOTE: 
# Before we make a decision, we need to separate our data into training and test sets. 
# Model validation is a large and very important topic that will be covered later, 
# but for the purpose of this end-to-end example, we will do a basic train-test split. 
# We will then build the decision model on the training data 
# and score it on the test data using the F1 score. 
# I recommend using a random seed for the most randomized data selection. 
# This seed tells the pseudo-random number generator where to begin its randomization routine. 
# The result is the same random choice every time. 
# In this example, I've used the random seed 42 when splitting into test and training sets. 
# Now, if I stop working on the project and pick it back up later, 
# I can split with the random seed and get the exact same training and test sets.

# Split into train/validation/test set
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_lda, test_size=0.3, random_state=42)

# Sanity check
print('train set shape = ' + str(df_train.shape))
print('test set shape = ' + str(df_test.shape))
print(df_train.head())

# classify with SVM
from sklearn.svm import SVC
from sklearn.metrics import f1_score
clf = SVC(kernel='rbf', C=0.8, gamma=10)
# C is a penalty term and is called a hyperparameter; this means that it is 
# a setting that an analyst can use to steer a fit in a certain direction. 
clf.fit(df_train[['lda1', 'lda2']], df_train['species'])

# predict on test set
y_pred = clf.predict(df_test[['lda1', 'lda2']])
f1 = f1_score(df_test['species'], y_pred, average='weighted')

# check prediction score
print("f1 score for SVM classifier = %2f " % f1)

# IMPORTANT NOTE:
# C is the penalty term in an SVM. It controls how large the penalty is 
# for a mis-classed example internally during the model fit. 
# For a utilitarian understanding, it is called the soft margin penalty 
# because it tunes how hard or soft the resulting separation line is drawn. 
# Common hyperparameters for SVMs will be covered in more detail later.

#  Let's change it from 0.8 to 1, which will effectively raise the penalty term.

# classify with SVM
from sklearn.svm import SVC
from sklearn.metrics import f1_score
clf = SVC(kernel='rbf', C=1, gamma=10)
clf.fit(df_train[['lda1', 'lda2']], df_train['species'])
y_pred = clf.predict(df_test[['lda1', 'lda2']])
f1 = f1_score(df_test['species'], y_pred, average='weighted')
print("f1 score for SVM classifier = %2f " % f1)

# The F1 score for this classifier is now 0.85. 
# The obvious next step is to tune the parameters and maximize the F1 score. 
# Of course, it will be very tedious to change a parameter (refit, analyze, and repeat). 
# Instead, you can employ a grid search to automate this parameterization. 
# Grid search and cross-validation will be covered in more detail later. 
# An alternative method to employing a grid search is to choose an algorithm that doesn't require tuning. 
# A popular algorithm that requires little-to-no tuning is Random Forest. 
# The forest refers to how the method adds together multiple decision trees into a voted prediction.

# classify with RF
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=2, random_state=42)
clf.fit(df_train[['lda1', 'lda2']], df_train['species'])
y_pred = clf.predict(df_test[['lda1', 'lda2']])
f1 = f1_score(df_test['species'], y_pred, average='weighted')

# check prediction score
print("f1 score for SVM classifier = %2f " % f1)

# The F1 score for this classifier is 0.96 - that is very good!




