import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings("ignore")
sns.set(style="white")
data = pd.read_csv("telco_customer_churn.csv")
data.head()

data.columns.values

data.dtypes

# Impute missing values with 0
data["TotalCharges"] = data["TotalCharges"].replace(" ", 0).astype("float32")

sns.pairplot(data)

# People having lower tenure and higher monthly charges are tend to churn more. Also as you can see below, having month-to-month contract and fiber optic internet have a really huge effect on churn probability.


# Categorical plotting
sns.catplot(y="Churn", kind="count", data=data, height=3.0, aspect=2.5, orient="h")

# Categorical bar plot
sns.catplot(x="Churn", kind="count", palette="ch:.25", data=data)

# Count plot
f, ax = plt.subplots(figsize=(9, 4))
sns.countplot(y="Churn", data=data, color="c")

'''
Categorical scatterplots:
- stripplot(): with kind="strip" the default
- swarmplot(): with kind="swarm"
Categorical distribution plots
- boxplot(): with kind="box"
- violinplot(): with kind="violin"
- boxenplot(): with kind="boxen"
Categorical estimate plots
- pointplot(): with kind="point"
- barplot(): with kind="bar"
- countplot(): with kind="count"
'''


# Numerical features (tenure, monthly charges, total charges)

def kdeplot(feature):
    # to estimate probability density distribution
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    sns.kdeplot(data[data["Churn"] == "No"][feature].dropna(), color="blue", label="Not churn")
    sns.kdeplot(data[data["Churn"] == "Yes"][feature].dropna(), color="orange", label="Churn")

kdeplot("tenure")
kdeplot("MonthlyCharges")
kdeplot("TotalCharges")


'''
From the plots above we can conclude that:
- Recent clients are more likely to churn
- Clients with higher MonthlyCharges are also more likely to churn
- Tenure and MonthlyCharges are probably important features
'''

# Boundaries with scatter plots
sns.PairGrid(data, y_vars=["tenure"], x_vars=["MonthlyCharges", "TotalCharges"], height=4.5, hue="Churn", aspect=1.5).map(plt.scatter, alpha=0.6)

g = sns.PairGrid(data, y_vars=["tenure"], x_vars=["MonthlyCharges", "TotalCharges"], height=4.5, aspect=1.5, hue="Churn")
g.map(sns.regplot, color=".3")
g.set(ylim=(-1, 11), yticks=[0, 5, 10])


# Calculating features
data["total_charges_to_tenure_ratio"] = data["TotalCharges"] / data["tenure"]
data["monthly_charges_diff"] = data["MonthlyCharges"] - data["total_charges_to_tenure_ratio"]
kdeplot("monthly_charges_diff")

'''
There are 16 categorical features:
- 6 binary features (Yes/No)
- 9 features with three unique values
- 1 feature with four unique values
'''

# SeniorCitizen (binary feature)
def plot_percentages(plot_type, feature, orient="v", axis_name="Percentage of customers"):
    ratios = pd.DataFrame()
    g = data.groupby(feature)["Churn"].value_counts().to_frame()
    g = g.rename({"Churn": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name] / len(data)
    if orient == "v":
        ax = sns.barplot(x=feature, y=axis_name, hue="Churn", data=g, orient=orient)
        ax.set_yticklabels(["{:,.0%}".format(y) for y in ax.get_yticks()])
    else:
        if plot_type == "barplot":
            ax = sns.barplot(x=axis_name, y=feature, hue="Churn", data=g, orient=orient)
        elif plot_type == "countplot":
            ax = sns.catplot(y=feature, kind="count", hue="Churn", data=g, orient=orient)
plot_percentages("barplot", "SeniorCitizen")

sns.catplot(y="SeniorCitizen", hue="Churn", kind="count", palette="pastel", edgecolor=".6", data=data)


# Gender distribution
df = data["gender"].value_counts() * 100.0 / len(data)
ax = df.plot(kind="bar", stacked=True, rot=0)
ax.set_ylabel("% Customers")
ax.set_xlabel("Gender")
ax.set_title("Gender Distribution")

# Churn rates to gender
data["churn_rate"] = data["Churn"].replace("No", 0).replace("Yes", 1)
g = sns.FacetGrid(data, col="SeniorCitizen", height=4, aspect=.9)
ax = g.map(sns.barplot, "gender", "churn_rate", palette = "Blues_d", order=["Female", "Male"])

# Senior citizens
df = data["SeniorCitizen"].value_counts() * 100.0 / len(data)
ax = df.plot.pie(autopct="%.2f", labels=["No", "Yes"], figsize=(6,6), fontsize=13)
ax.set_ylabel("Senior Citizens", fontsize=13)
ax.set_title("Senior Citizens Distribution", fontsize=13)


# There are only 16% of the customers who are senior citizens. Thus most of our customers in the data are younger people

# Phone services
plt.figure(figsize=(9, 5))
plot_percentages("barplot", "MultipleLines", orient="h")

# Effect of multiple lines to MonthlyCharges
sns.catplot(x="MultipleLines", y="MonthlyCharges", hue="Churn", kind="violin",
            split=True, palette="pastel", data=data, height=4.5, aspect=1.6)


sns.catplot(x="MultipleLines", y="MonthlyCharges", hue="Churn", kind="bar", 
            palette="pastel", edgecolor=".6", data=data)


# Internet services
plt.figure(figsize=(9, 5))
plot_percentages("barplot", "InternetService", orient="h")

'''
- Clients without internet have a very low churn rate
- Customers with fiber are more probable to churn than those with DSL connection
'''

sns.catplot(x="InternetService", y="MonthlyCharges", hue="Churn", kind="violin",
            split=True, palette="pastel", data=data, height=4.5, aspect=1.6);

sns.catplot(x="InternetService", y="MonthlyCharges", hue="Churn", kind="bar", 
            palette="pastel", edgecolor=".6", data=data)

# Additional services (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df = pd.melt(data[data["InternetService"] != "No"][cols]).rename({'value': 'Has service'}, axis=1)
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="variable", hue="Has service").set(xlabel="Additional service", ylabel="Number of customers")
plt.show()


plt.figure(figsize=(10, 5))
df = data[(data.InternetService != "No") & (data.Churn == "Yes")]
df = pd.melt(df[cols]).rename({"value": "Has service"}, axis=1)
sns.countplot(data=df, x="variable", hue="Has service", hue_order=["No", "Yes"]).set(xlabel="Additional service", ylabel="Number of churns")
plt.show()

sns.catplot(y="TechSupport", kind="count", palette="pastel", edgecolor=".6", data=data)

'''
- Customers with the first 4 additionals (security to tech support) are more unlikely to churn
- Streaming service is not predictive for churn
'''

# Contract and payment
g = sns.FacetGrid(data, col="PaperlessBilling", height=5, aspect=1.0)
ax = g.map(sns.barplot, "Contract", "churn_rate", palette = "Blues_d", order=["Month-to-month", "One year", "Two year"])

plt.figure(figsize=(10, 5))
plot_percentages("barplot", "PaymentMethod", orient="h")

'''
- Customers with paperless billing are more probable to churn
- The preferred payment method is Electronic check with around 35% of customers. This method also has a very high churn rate
- Short term contracts have higher churn rates
'''

sns.catplot(x="Contract", y="MonthlyCharges", hue="Churn", kind="box", data=data, height=4.5, aspect=1.5)

sns.catplot(y="Churn", x="MonthlyCharges", row="PaymentMethod", kind="box", data=data, height=1.6, aspect=4, orient="h")

'''
- Longer contracts are more affected by higher monthly charges (for churn rate)
- Mailed checks have lower charges
- There is a huge gap in charges between customers that churn and those that don't with respect to Mailed Check
'''

# Correlation between features (Correlation heatmap using Pearson correlation coefficient method)
plt.figure(figsize=(15, 7))
data.drop(["customerID", "churn_rate", "total_charges_to_tenure_ratio", "monthly_charges_diff"], axis=1, inplace=True)
corr = data.apply(lambda x: pd.factorize(x)[0]).corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2, cmap="RdYlBu")


# Correlation heatmap of numerical and binary columns with plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.subplots as subplots
import plotly.figure_factory as ff

# Correlation Matrix
data = pd.read_csv("telco_customer_churn.csv")
correlation = data.corr()

# Tick labels
matrix_cols = correlation.columns.tolist()

# Convert to array
corr_array = np.array(correlation)

# Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = "Viridis",
                   colorbar = dict(title = "Pearson Correlation coefficient",
                                   titleside = "right"))

layout = go.Layout(dict(title = "Correlation Matrix for variables",
                        autosize = False,
                        height = 720,
                        width = 800,
                        margin = dict(r = 0, l = 210, t = 25, b = 210),
                        yaxis = dict(tickfont = dict(size = 9)),
                        xaxis = dict(tickfont = dict(size = 9))))

data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)

# https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
# Correlation matrix with heatmapz

from heatmap import heatmap, corrplot

data = pd.read_csv("telco_customer_churn.csv")
plt.figure(figsize=(8,8))
corrplot(data.corr(), size_scale=300);

# Contracts information (month-to-month contract, two year contract, one year contract)
ax1 = sns.distplot(data[data["Contract"]=="Month-to-month"]["tenure"], 
             hist=True, kde=False, bins=int(180/5), hist_kws={'edgecolor':'black'}, 
             kde_kws={'linewidth': 4})
ax1.set_ylabel('Number of Customers')
ax1.set_xlabel('Tenure (months)')
ax1.set_title('Month-to-month Contract')

ax2 = sns.distplot(data[data["Contract"]=="One year"]["tenure"],
             hist=True, kde=False, bins=int(180/5), hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
ax2.set_xlabel('Tenure (months)',size = 14)
ax2.set_title('One Year Contract',size = 14)

ax3 = sns.distplot(data[data["Contract"]=="Two year"]["tenure"],
                  hist=True, kde=False, bins=int(180/5), hist_kws={'edgecolor':'black'},
                  kde_kws={'linewidth': 4})
ax3.set_xlabel('Tenure (months)')
ax3.set_title('Two Year Contract')


# Additional Documents
# - https://www.kaggle.com/salibbigt/customer-churn-analysis (Explaratory data analysis with Tableau)
