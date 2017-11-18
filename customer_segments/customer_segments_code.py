
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[1]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
    
print data.head(n=5)


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[2]:

# Display a description of the dataset
display(data.describe())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[3]:

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [3, 80, 100]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

totals = []

for i in range(3):
    #print "i:", str(i)
    total = 0
    for column in samples:
        #print samples[column][i]
        total += samples[column][i]
    totals.append(total)
    
print totals


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# *What kind of establishment (customer) could each of the three samples you've chosen represent?*  
# **Hint:** Examples of establishments include places like markets, cafes, and retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant.

# **Answer:**
# 
# 0 - The total purchase cost is relatively high, at around 27,000. It seems the customer is somewhat of an average case; they purchase above the mean value for Frozen, Delicatessen and Frozen goods, but lower than the mean for the others. Looking at the individual product categories, the customer's spending values for each category is generally quite close to average spending, being within the 75th and 25th percentile values. This is true for all the categories, except 'Milk' and 'Frozen'. The value for Milk is below the 25th percentile value, whereas the value for frozen is above the 75th percentile. Given this behaviour, the substantial preference for Frozen goods and the low demand for Milk goods, the customer could potentially be a vendor specializing in healthy eating options, such as fruits, vegetables and juices. 
# 
# 1 - The total purchase cost for this customer is substantially less than that of the first, being around 12,000. Like the first customer, the majority of goods purchased fall within the 25th and 75th percentile values, with the exception of Milk and Frozen goods. However, in this case, both the amounts of Milk and Frozen goods being lower than the 25th percentile value. It seems this customer is smaller in size relative to other customers, as they purchase below the mean value for each of the categories. With this in mind, it's possible the customer may be a small-scale supermarket.
# 
# 2 - This customer overwhelmingly had the highest purchase cost of all three samples, of around 46,000. In each of the categories, the customer purchases an amount above the mean, with the only exception being Fresh goods, for which the spending is only slightly below the mean. As well, for each of the other categories, the spending is above the 75th percentile value. Given this large spending in a variety of different areas, it is likely that this customer is a large scale supermarket.

# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[4]:

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score


dropped_feature = 'Grocery'


# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop(dropped_feature, axis = 1)

# TODO: Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(new_data, data[dropped_feature], test_size = 0.25, random_state = 77)

# TODO: Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state = 5)

regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
print regressor.score(X_test, y_test)


# ### Question 2
# *Which feature did you attempt to predict? What was the reported prediction score? Is this feature necessary for identifying customers' spending habits?*  
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data.

# **Answer:** I attempted to predict the value of the 'Grocery' feature. The prediction score for the regression was about 0.638. This means a fair amount of the information in the 'Grocery' feature can be explained by other features, but not all. This means there may still be information in the 'Grocery' feature that may prove helpful, therefore it would be wise to keep this feature in the dataset to best identify customers' spending habits.

# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[5]:

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Question 3
# *Are there any pairs of features which exhibit some degree of correlation? Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? How is the data for those features distributed?*  
# **Hint:** Is the data normally distributed? Where do most of the data points lie? 

# **Answer:** The 'Grocery' and 'Detergents' feature show quite a strong linear relationship, converse to what I suspected in the previous section. The data for these features appear to follow a non-normal distribution, with the overwhelming majority of points at low values for each of the features, with a sparse amount at higher values. This shows that in fact, it is quite likely that the 'Grocery' feature could be dropped from the dataset for our analysis. This is because it is possible to infer these values from the values for 'Detergents', and not much information is gained from each of these two features, given we have looked at the other.

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

# In[6]:

# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[15]:

# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[16]:

# For each feature find the data points with extreme high or low values

from collections import Counter

category_outliers_count = Counter()

for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    
    outliers_df = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    
    display(outliers_df)
    
    for index_ in outliers_df.index.values:
        category_outliers_count[index_] += 1
    
    
# OPTIONAL: Select the indices for data points you wish to remove
outliers = []

for index_ in category_outliers_count:
    if category_outliers_count[index_] >= 2:
        outliers.append(index_)

print "Outliers are:"        
print outliers

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# ### Question 4
# *Are there any data points considered outliers for more than one feature based on the definition above? Should these data points be removed from the dataset? If any data points were added to the `outliers` list to be removed, explain why.* 

# **Answer:** It was found that data points 128, 154, 65, 66 and 75 were outliers in more than one feature, and as a result were removed. This was because, as mentioned above, these data points may undesirably skew results in a way that would be detrimental to our clustering of the customers. Our goal is to organize customers into groups of similar customers; since these outliers are such vastly different to the majority of customers in our dataset, it may result in us having unnecessarily large clusters to accommodate one of these points into a group.

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[27]:

from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6)

pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

print pca_results['Explained Variance'].cumsum()


# ### Question 5
# *How much variance in the data is explained* ***in total*** *by the first and second principal component? What about the first four principal components? Using the visualization provided above, discuss what the first four dimensions best represent in terms of customer spending.*  
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.

# **Answer:** 71.15% of the variance of the data is explained by the first and second principal components alone. The first four explain 92.92% of the variance.
# 
# A customer would record a high value in the first principal component if they: purchased substantially higher amounts of fresh and frozen goods as compared to all others, purchased slightly higher amounts of the other goods as compared to fresh/frozen goods, or if they purchased much higher amounts of the others, as compared to fresh/frozen (a very high value would be gotten in this case). Looking at the magnitude of the value for this principal component, we can tell whether the customer falls into one of these different categories of customer or not, and how strongly so.
# 
# For the second principal component, there are high weights given to fresh, frozen and delicatessen goods. Customers that record a higher value for this principal component likely purchased high amounts of these types of goods. Again, looking at the magnitude of the value for this principal component will give us insight into whether a customer spends highly in these three types of goods or not, and to what extent.
# 
# The value in the third principal component also shows how much a customer falls into a particular category. Given the opposite signs for the weights for fresh, detergents/paper and frozen, delicatessen, we will know that a large magnitude in this principal component means that a customer purchases one group of these items and not the other. For example, a customer may record a high value for this principal component if they purchased high amounts of fresh and detergents/paper products, and a low amount of delicatessen and frozen goods. The same would hold if the opposite was true: if a high amount of delicatessen and frozen goods were purchased, together with a lower amount of fresh and detergents/paper products.
# 
# The fourth principal component is similar to the third, in that we have a few features that have much higher weight magnitudes than the others, and with weights being both positive and negative. This principal component gives us insight into whether a customer purchased high amounts of frozen and detergents/paper goods (with an emphasis on frozen, given the high weight value) and low amounts of delicatessen and fresh goods, or the other way around. 

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[18]:

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[19]:

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)

pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[20]:

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

# In[21]:

# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# *What are the advantages to using a K-Means clustering algorithm? What are the advantages to using a Gaussian Mixture Model clustering algorithm? Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?*

# **Answer:** A K-Means clustering algorithm generates definitive clusters; that is, it decides with 100% certainty that a data point belongs to one cluster, and none else. Gaussian Mixture Modelling (GMM), however, is based on Expectation Maximization in that it performs a kind of soft clustering, in which it assigns to each data point a confidence value that describes how strongly the algorithm believes it should be in a certain cluster. K-means operates faster, however given that the dataset is not overly large, using GMM shouldn't pose any issues.
# 
# K-means also assumes that the clusters have similar variance; it can be seen in the plot with the first two principal components that the ideal clusters do indeed have similar variances, although we can't guarantee this when we start incorporating more principal components. GMM would still be expected to give us strong results if the variances are different. 
# 
# Given the domain, it seems better to have a soft clustering mechanism, as it would be expected that there are some customers that exhibit behaviour of more than one standard 'customer segment.' Therefore, a Gaussian Mixture Model clustering algorithm would be more suitable. 

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[22]:

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# TODO: Apply your clustering algorithm of choice to the reduced data 
clusterer = GaussianMixture(n_components = 2)

clusterer.fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.means_

# print centers

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data,preds)

print score


# ### Question 7
# *Report the silhouette score for several cluster numbers you tried. Of these, which number of clusters has the best silhouette score?* 

# **Answer:** 
# 2 - 0.41417,
# 3 - 0.40888,
# 4 - 0.33142,
# 5 - 0.34365,
# 10 - 0.31437,
# 15 - 0.32492,
# 30 - 0.25187,
# 50 - 0.31744
# 
# It can be seen that we get the maximal silhouette score by using just two clusters.

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[23]:

# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)


# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

# In[24]:

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# ### Question 8
# Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project. *What set of establishments could each of the customer segments represent?*  
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`.

# **Answer:** Customers assigned to Cluster 0 appear to purchase a relatively average amount of product, given that cost of the product purchases for the center point falls within the 25th and 75th percentile values from the statistical description. Of these, it's only Fresh and Frozen product that are above the median value. Given this distribution of product purchases over the different categories, it is likely that customers in this cluster are smaller scale food vendors, potentially with specialization in selling fresh and frozen produce (fruits, vegetables, meats).
# 
# Looking at the center point for Cluster 1, much higher amounts of product purchases can be seen. Similar to Cluster 0, the purchases for the majority of product fall within the 25th and 75th percentile values, with the exception of Milk, Grocery and Detergents/Paper goods, that have been purchased above the 75th percentile values. It is likely then, that customers that belong to this cluster are larger scale supermarkets, given the large spending in a variety of areas, and the substantially high purchasing in the Milk, Grocery and Detergents/Paper products.

# ### Question 9
# *For each sample point, which customer segment from* ***Question 8*** *best represents it? Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[28]:

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred


# **Answer:** It would be expected that Sample Point 0 would belong to the first cluster. Although the actual values of product purchased for each category are not too close in magnitude, it seems that the sample point follows the same pattern of spending as described in the center point for Cluster 0. It can be seen that for both, a large amount of fresh goods have been purchased relative to the rest of the categories,  with moderate amounts of Milk, Grocery and Frozen goods, and low amounts of Delicatessen and Detergents/Paper product. Overall the characteristics of the first sample point match the first cluster centre, moreso than the second.
# 
# As for sample point 1, we again see the high comparative spending on Frozen goods that we saw in the center point for Cluster 0. However, there is a low amount of spending on frozen goods, which is behaviour that is seen in Cluster 1. Looking at the Detergents/Paper feature though, we see a characteristic low spending, which is much more indicative of Cluster 0 behaviour. Looking at the overall purchasing price as well, it seems more characteristic of Cluster 0, since it isn't too high. Given this, it would be expected that sample point 1 also belongs to cluster 0, however perhaps with a lower confidence value than sample point 0.
# 
# Looking at sample point 2, it can be seen that the overall purchase price is substantially higher than the other two sample points, with many feature purchase prices beyond the 75th percentile range - Milk, Grocery, Detergents/Paper and Delicatessen. This gives strong credence to this sample point belonging in Cluster 1. Looking comparatively at the features, we also see relatively lower values for both frozen and delicatessen product, another characteristic of the center point for Cluster 1. With all of this in mind, it is highly likely that sample point 1 belongs to Cluster 1. 
# 
# After running the code, it can be seen that all of the cluster predictions for each of the sample points was correct. This shows just how looking at the centre point for the cluster can give us a great amount of insight into the spending behaviour and patterns of customers in the cluster.

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. *How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*  
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

# **Answer:** In order to conduct A/B testing, it is required that the experimentation group (the group of customers for which the change will be applied) is similar to the control group (a group of customers for which no change is made, in order to provide comparison). If the experimentation and control group are chosen at random, it is difficult to ascertain what positive/negative effects implementing a change will actually have - since the customers are likely quite different, any interesting data may be the result of some other sort of variance, than that caused by A/B testing.
# 
# By using the customer segments, the wholesale retailer can pick an experimentation and control group from the same cluster. Given that customers in a given segment exhibit similar behaviour, it is easier to isolate and ascertain what the effects are of implementing changes to the distributors's policies. Looking back to the log-transformed data, and how it exhibits Gaussian behaviour - the company could opt to pick customers close to the mean for each cluster, to ensure they are as similar as possible.

# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# *How can the wholesale distributor label the new customers using only their estimated product spending and the* ***customer segment*** *data?*  
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

# **Answer:** Given that we have paired each original customer with a cluster, and that the cluster can be thought of as an 'engineered feature' as said above, we could run a supervised learning classification algorithm that attempts to pair the features of the original clusters to these new engineered features (which would act as the labels). Once the algorithm has been trained on the original customers, it can then be used on the new customers to classify them into the different customer segments.

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[26]:

# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)


# ### Question 12
# *How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? Would you consider these classifications as consistent with your previous definition of the customer segments?*

# **Answer:** The clustering algorithm produced quite similar clusters to the above plot. It seems that cluster 1 is associated mostly with Retailer customers, whereas cluster 0 corresponds to Hotel/Restaurant/Cafe customers. These descriptions are somewhat similar to the definition of customer segments explained before; it makes sense that large scale supermarkets (description given to cluster 1) would fall under the category of being a Retailer. However, it seems that cluster 0 corresponds not only to specialists in fresh goods as discussed above, but instead to Hotels, Restaurants and Cafes in general. This is likely due to my inability to connect typical feature values with real world examples. Looking back at the features now, it makes much more sense that cluster 0 customers are Hotel/Restaurant/Cafes.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
