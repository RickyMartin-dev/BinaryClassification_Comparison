{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Last Updated: 2/1/2021\n",
    "\n",
    "UCSD Class Project: Cogs 188"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Supervised Comparison"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Description: Below is a comparison of different Supervised Learning Models on a single data set. First one made entirely by myself and the others are imported from sklearn. Due to the binary classification yet multifeature dataset, I thought it would be interesting to explore the different results given different methodologies."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Breast Cancer Prediction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Breast cancer is one of the most common cancers among women worldwide, representing the majority of new cancer cases and cancer-related deaths according to global statistics, making it a significant public health problem in today’s society.\n",
    "\n",
    "The early diagnosis of Breast Cancer can improve the prognosis and chance of survival significantly, as it can promote timely clinical treatment to patients. Further accurate classification of benign tumors can prevent patients undergoing unnecessary treatments. Thus, the correct diagnosis of BC and classification of patients into malignant or benign groups is the subject of much research. Because of its unique advantages in critical features detection from complex Brest Cancer datasets, Machine Learning is widely recognized as the methodology of choice in Breast Cancer pattern classification and forecast modelling.\n",
    "\n",
    "Link to data set: http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29\n",
    "\n",
    "Link to sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "import sklearn.datasets\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load the breast cancer data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Data Set\n",
    "breast_cancer = sklearn.datasets.load_breast_cancer()\n",
    "\n",
    "# Turn Data into Dataframe\n",
    "data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)\n",
    "data['class'] = breast_cancer.target\n",
    "\n",
    "# Seperate the Data\n",
    "X = data.drop('class', axis = 1)\n",
    "Y = data['class']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Some Visualizations of the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>mean radius</th>\n",
       "      <th>mean texture</th>\n",
       "      <th>mean perimeter</th>\n",
       "      <th>mean area</th>\n",
       "      <th>mean smoothness</th>\n",
       "      <th>mean compactness</th>\n",
       "      <th>mean concavity</th>\n",
       "      <th>mean concave points</th>\n",
       "      <th>mean symmetry</th>\n",
       "      <th>mean fractal dimension</th>\n",
       "      <th>...</th>\n",
       "      <th>worst texture</th>\n",
       "      <th>worst perimeter</th>\n",
       "      <th>worst area</th>\n",
       "      <th>worst smoothness</th>\n",
       "      <th>worst compactness</th>\n",
       "      <th>worst concavity</th>\n",
       "      <th>worst concave points</th>\n",
       "      <th>worst symmetry</th>\n",
       "      <th>worst fractal dimension</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>17.99</td>\n",
       "      <td>10.38</td>\n",
       "      <td>122.80</td>\n",
       "      <td>1001.0</td>\n",
       "      <td>0.11840</td>\n",
       "      <td>0.27760</td>\n",
       "      <td>0.3001</td>\n",
       "      <td>0.14710</td>\n",
       "      <td>0.2419</td>\n",
       "      <td>0.07871</td>\n",
       "      <td>...</td>\n",
       "      <td>17.33</td>\n",
       "      <td>184.60</td>\n",
       "      <td>2019.0</td>\n",
       "      <td>0.1622</td>\n",
       "      <td>0.6656</td>\n",
       "      <td>0.7119</td>\n",
       "      <td>0.2654</td>\n",
       "      <td>0.4601</td>\n",
       "      <td>0.11890</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20.57</td>\n",
       "      <td>17.77</td>\n",
       "      <td>132.90</td>\n",
       "      <td>1326.0</td>\n",
       "      <td>0.08474</td>\n",
       "      <td>0.07864</td>\n",
       "      <td>0.0869</td>\n",
       "      <td>0.07017</td>\n",
       "      <td>0.1812</td>\n",
       "      <td>0.05667</td>\n",
       "      <td>...</td>\n",
       "      <td>23.41</td>\n",
       "      <td>158.80</td>\n",
       "      <td>1956.0</td>\n",
       "      <td>0.1238</td>\n",
       "      <td>0.1866</td>\n",
       "      <td>0.2416</td>\n",
       "      <td>0.1860</td>\n",
       "      <td>0.2750</td>\n",
       "      <td>0.08902</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>19.69</td>\n",
       "      <td>21.25</td>\n",
       "      <td>130.00</td>\n",
       "      <td>1203.0</td>\n",
       "      <td>0.10960</td>\n",
       "      <td>0.15990</td>\n",
       "      <td>0.1974</td>\n",
       "      <td>0.12790</td>\n",
       "      <td>0.2069</td>\n",
       "      <td>0.05999</td>\n",
       "      <td>...</td>\n",
       "      <td>25.53</td>\n",
       "      <td>152.50</td>\n",
       "      <td>1709.0</td>\n",
       "      <td>0.1444</td>\n",
       "      <td>0.4245</td>\n",
       "      <td>0.4504</td>\n",
       "      <td>0.2430</td>\n",
       "      <td>0.3613</td>\n",
       "      <td>0.08758</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>11.42</td>\n",
       "      <td>20.38</td>\n",
       "      <td>77.58</td>\n",
       "      <td>386.1</td>\n",
       "      <td>0.14250</td>\n",
       "      <td>0.28390</td>\n",
       "      <td>0.2414</td>\n",
       "      <td>0.10520</td>\n",
       "      <td>0.2597</td>\n",
       "      <td>0.09744</td>\n",
       "      <td>...</td>\n",
       "      <td>26.50</td>\n",
       "      <td>98.87</td>\n",
       "      <td>567.7</td>\n",
       "      <td>0.2098</td>\n",
       "      <td>0.8663</td>\n",
       "      <td>0.6869</td>\n",
       "      <td>0.2575</td>\n",
       "      <td>0.6638</td>\n",
       "      <td>0.17300</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20.29</td>\n",
       "      <td>14.34</td>\n",
       "      <td>135.10</td>\n",
       "      <td>1297.0</td>\n",
       "      <td>0.10030</td>\n",
       "      <td>0.13280</td>\n",
       "      <td>0.1980</td>\n",
       "      <td>0.10430</td>\n",
       "      <td>0.1809</td>\n",
       "      <td>0.05883</td>\n",
       "      <td>...</td>\n",
       "      <td>16.67</td>\n",
       "      <td>152.20</td>\n",
       "      <td>1575.0</td>\n",
       "      <td>0.1374</td>\n",
       "      <td>0.2050</td>\n",
       "      <td>0.4000</td>\n",
       "      <td>0.1625</td>\n",
       "      <td>0.2364</td>\n",
       "      <td>0.07678</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \\\n",
       "0        17.99         10.38          122.80     1001.0          0.11840   \n",
       "1        20.57         17.77          132.90     1326.0          0.08474   \n",
       "2        19.69         21.25          130.00     1203.0          0.10960   \n",
       "3        11.42         20.38           77.58      386.1          0.14250   \n",
       "4        20.29         14.34          135.10     1297.0          0.10030   \n",
       "\n",
       "   mean compactness  mean concavity  mean concave points  mean symmetry  \\\n",
       "0           0.27760          0.3001              0.14710         0.2419   \n",
       "1           0.07864          0.0869              0.07017         0.1812   \n",
       "2           0.15990          0.1974              0.12790         0.2069   \n",
       "3           0.28390          0.2414              0.10520         0.2597   \n",
       "4           0.13280          0.1980              0.10430         0.1809   \n",
       "\n",
       "   mean fractal dimension  ...  worst texture  worst perimeter  worst area  \\\n",
       "0                 0.07871  ...          17.33           184.60      2019.0   \n",
       "1                 0.05667  ...          23.41           158.80      1956.0   \n",
       "2                 0.05999  ...          25.53           152.50      1709.0   \n",
       "3                 0.09744  ...          26.50            98.87       567.7   \n",
       "4                 0.05883  ...          16.67           152.20      1575.0   \n",
       "\n",
       "   worst smoothness  worst compactness  worst concavity  worst concave points  \\\n",
       "0            0.1622             0.6656           0.7119                0.2654   \n",
       "1            0.1238             0.1866           0.2416                0.1860   \n",
       "2            0.1444             0.4245           0.4504                0.2430   \n",
       "3            0.2098             0.8663           0.6869                0.2575   \n",
       "4            0.1374             0.2050           0.4000                0.1625   \n",
       "\n",
       "   worst symmetry  worst fractal dimension  class  \n",
       "0          0.4601                  0.11890      0  \n",
       "1          0.2750                  0.08902      0  \n",
       "2          0.3613                  0.08758      0  \n",
       "3          0.6638                  0.17300      0  \n",
       "4          0.2364                  0.07678      0  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>mean radius</th>\n",
       "      <th>mean texture</th>\n",
       "      <th>mean perimeter</th>\n",
       "      <th>mean area</th>\n",
       "      <th>mean smoothness</th>\n",
       "      <th>mean compactness</th>\n",
       "      <th>mean concavity</th>\n",
       "      <th>mean concave points</th>\n",
       "      <th>mean symmetry</th>\n",
       "      <th>mean fractal dimension</th>\n",
       "      <th>...</th>\n",
       "      <th>worst texture</th>\n",
       "      <th>worst perimeter</th>\n",
       "      <th>worst area</th>\n",
       "      <th>worst smoothness</th>\n",
       "      <th>worst compactness</th>\n",
       "      <th>worst concavity</th>\n",
       "      <th>worst concave points</th>\n",
       "      <th>worst symmetry</th>\n",
       "      <th>worst fractal dimension</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>...</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "      <td>569.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>14.127292</td>\n",
       "      <td>19.289649</td>\n",
       "      <td>91.969033</td>\n",
       "      <td>654.889104</td>\n",
       "      <td>0.096360</td>\n",
       "      <td>0.104341</td>\n",
       "      <td>0.088799</td>\n",
       "      <td>0.048919</td>\n",
       "      <td>0.181162</td>\n",
       "      <td>0.062798</td>\n",
       "      <td>...</td>\n",
       "      <td>25.677223</td>\n",
       "      <td>107.261213</td>\n",
       "      <td>880.583128</td>\n",
       "      <td>0.132369</td>\n",
       "      <td>0.254265</td>\n",
       "      <td>0.272188</td>\n",
       "      <td>0.114606</td>\n",
       "      <td>0.290076</td>\n",
       "      <td>0.083946</td>\n",
       "      <td>0.627417</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3.524049</td>\n",
       "      <td>4.301036</td>\n",
       "      <td>24.298981</td>\n",
       "      <td>351.914129</td>\n",
       "      <td>0.014064</td>\n",
       "      <td>0.052813</td>\n",
       "      <td>0.079720</td>\n",
       "      <td>0.038803</td>\n",
       "      <td>0.027414</td>\n",
       "      <td>0.007060</td>\n",
       "      <td>...</td>\n",
       "      <td>6.146258</td>\n",
       "      <td>33.602542</td>\n",
       "      <td>569.356993</td>\n",
       "      <td>0.022832</td>\n",
       "      <td>0.157336</td>\n",
       "      <td>0.208624</td>\n",
       "      <td>0.065732</td>\n",
       "      <td>0.061867</td>\n",
       "      <td>0.018061</td>\n",
       "      <td>0.483918</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>6.981000</td>\n",
       "      <td>9.710000</td>\n",
       "      <td>43.790000</td>\n",
       "      <td>143.500000</td>\n",
       "      <td>0.052630</td>\n",
       "      <td>0.019380</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.106000</td>\n",
       "      <td>0.049960</td>\n",
       "      <td>...</td>\n",
       "      <td>12.020000</td>\n",
       "      <td>50.410000</td>\n",
       "      <td>185.200000</td>\n",
       "      <td>0.071170</td>\n",
       "      <td>0.027290</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.156500</td>\n",
       "      <td>0.055040</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>11.700000</td>\n",
       "      <td>16.170000</td>\n",
       "      <td>75.170000</td>\n",
       "      <td>420.300000</td>\n",
       "      <td>0.086370</td>\n",
       "      <td>0.064920</td>\n",
       "      <td>0.029560</td>\n",
       "      <td>0.020310</td>\n",
       "      <td>0.161900</td>\n",
       "      <td>0.057700</td>\n",
       "      <td>...</td>\n",
       "      <td>21.080000</td>\n",
       "      <td>84.110000</td>\n",
       "      <td>515.300000</td>\n",
       "      <td>0.116600</td>\n",
       "      <td>0.147200</td>\n",
       "      <td>0.114500</td>\n",
       "      <td>0.064930</td>\n",
       "      <td>0.250400</td>\n",
       "      <td>0.071460</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>13.370000</td>\n",
       "      <td>18.840000</td>\n",
       "      <td>86.240000</td>\n",
       "      <td>551.100000</td>\n",
       "      <td>0.095870</td>\n",
       "      <td>0.092630</td>\n",
       "      <td>0.061540</td>\n",
       "      <td>0.033500</td>\n",
       "      <td>0.179200</td>\n",
       "      <td>0.061540</td>\n",
       "      <td>...</td>\n",
       "      <td>25.410000</td>\n",
       "      <td>97.660000</td>\n",
       "      <td>686.500000</td>\n",
       "      <td>0.131300</td>\n",
       "      <td>0.211900</td>\n",
       "      <td>0.226700</td>\n",
       "      <td>0.099930</td>\n",
       "      <td>0.282200</td>\n",
       "      <td>0.080040</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>15.780000</td>\n",
       "      <td>21.800000</td>\n",
       "      <td>104.100000</td>\n",
       "      <td>782.700000</td>\n",
       "      <td>0.105300</td>\n",
       "      <td>0.130400</td>\n",
       "      <td>0.130700</td>\n",
       "      <td>0.074000</td>\n",
       "      <td>0.195700</td>\n",
       "      <td>0.066120</td>\n",
       "      <td>...</td>\n",
       "      <td>29.720000</td>\n",
       "      <td>125.400000</td>\n",
       "      <td>1084.000000</td>\n",
       "      <td>0.146000</td>\n",
       "      <td>0.339100</td>\n",
       "      <td>0.382900</td>\n",
       "      <td>0.161400</td>\n",
       "      <td>0.317900</td>\n",
       "      <td>0.092080</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>28.110000</td>\n",
       "      <td>39.280000</td>\n",
       "      <td>188.500000</td>\n",
       "      <td>2501.000000</td>\n",
       "      <td>0.163400</td>\n",
       "      <td>0.345400</td>\n",
       "      <td>0.426800</td>\n",
       "      <td>0.201200</td>\n",
       "      <td>0.304000</td>\n",
       "      <td>0.097440</td>\n",
       "      <td>...</td>\n",
       "      <td>49.540000</td>\n",
       "      <td>251.200000</td>\n",
       "      <td>4254.000000</td>\n",
       "      <td>0.222600</td>\n",
       "      <td>1.058000</td>\n",
       "      <td>1.252000</td>\n",
       "      <td>0.291000</td>\n",
       "      <td>0.663800</td>\n",
       "      <td>0.207500</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       mean radius  mean texture  mean perimeter    mean area  \\\n",
       "count   569.000000    569.000000      569.000000   569.000000   \n",
       "mean     14.127292     19.289649       91.969033   654.889104   \n",
       "std       3.524049      4.301036       24.298981   351.914129   \n",
       "min       6.981000      9.710000       43.790000   143.500000   \n",
       "25%      11.700000     16.170000       75.170000   420.300000   \n",
       "50%      13.370000     18.840000       86.240000   551.100000   \n",
       "75%      15.780000     21.800000      104.100000   782.700000   \n",
       "max      28.110000     39.280000      188.500000  2501.000000   \n",
       "\n",
       "       mean smoothness  mean compactness  mean concavity  mean concave points  \\\n",
       "count       569.000000        569.000000      569.000000           569.000000   \n",
       "mean          0.096360          0.104341        0.088799             0.048919   \n",
       "std           0.014064          0.052813        0.079720             0.038803   \n",
       "min           0.052630          0.019380        0.000000             0.000000   \n",
       "25%           0.086370          0.064920        0.029560             0.020310   \n",
       "50%           0.095870          0.092630        0.061540             0.033500   \n",
       "75%           0.105300          0.130400        0.130700             0.074000   \n",
       "max           0.163400          0.345400        0.426800             0.201200   \n",
       "\n",
       "       mean symmetry  mean fractal dimension  ...  worst texture  \\\n",
       "count     569.000000              569.000000  ...     569.000000   \n",
       "mean        0.181162                0.062798  ...      25.677223   \n",
       "std         0.027414                0.007060  ...       6.146258   \n",
       "min         0.106000                0.049960  ...      12.020000   \n",
       "25%         0.161900                0.057700  ...      21.080000   \n",
       "50%         0.179200                0.061540  ...      25.410000   \n",
       "75%         0.195700                0.066120  ...      29.720000   \n",
       "max         0.304000                0.097440  ...      49.540000   \n",
       "\n",
       "       worst perimeter   worst area  worst smoothness  worst compactness  \\\n",
       "count       569.000000   569.000000        569.000000         569.000000   \n",
       "mean        107.261213   880.583128          0.132369           0.254265   \n",
       "std          33.602542   569.356993          0.022832           0.157336   \n",
       "min          50.410000   185.200000          0.071170           0.027290   \n",
       "25%          84.110000   515.300000          0.116600           0.147200   \n",
       "50%          97.660000   686.500000          0.131300           0.211900   \n",
       "75%         125.400000  1084.000000          0.146000           0.339100   \n",
       "max         251.200000  4254.000000          0.222600           1.058000   \n",
       "\n",
       "       worst concavity  worst concave points  worst symmetry  \\\n",
       "count       569.000000            569.000000      569.000000   \n",
       "mean          0.272188              0.114606        0.290076   \n",
       "std           0.208624              0.065732        0.061867   \n",
       "min           0.000000              0.000000        0.156500   \n",
       "25%           0.114500              0.064930        0.250400   \n",
       "50%           0.226700              0.099930        0.282200   \n",
       "75%           0.382900              0.161400        0.317900   \n",
       "max           1.252000              0.291000        0.663800   \n",
       "\n",
       "       worst fractal dimension       class  \n",
       "count               569.000000  569.000000  \n",
       "mean                  0.083946    0.627417  \n",
       "std                   0.018061    0.483918  \n",
       "min                   0.055040    0.000000  \n",
       "25%                   0.071460    0.000000  \n",
       "50%                   0.080040    1.000000  \n",
       "75%                   0.092080    1.000000  \n",
       "max                   0.207500    1.000000  \n",
       "\n",
       "[8 rows x 31 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "#sns.pairplot(data,hue = 'class', palette= 'coolwarm', vars = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean','smoothness_mean'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Split Training and Testing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Encoding categorical data values\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "labelencoder_Y = LabelEncoder()\n",
    "Y = labelencoder_Y.fit_transform(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Splitting\n",
    "\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify= Y, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Valu Extraction\n",
    "\n",
    "X_train = X_train.values\n",
    "X_test = X_test.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Scaling\n",
    "\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "sc = StandardScaler()\n",
    "X_train = sc.fit_transform(X_train)\n",
    "X_test = sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dictionary to keep all results in\n",
    "results = {}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 1: Perceptron Class Built"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Perceptron:\n",
    "    def __init__ (self):\n",
    "        self.w = None\n",
    "        self.b = None\n",
    "    \n",
    "    def model(self, x):\n",
    "        # returns the prediction for a single example x\n",
    "        pred = 0\n",
    "        decision = np.dot(self.w, x)\n",
    "        \n",
    "        if decision >= self.b:\n",
    "            pred = 1\n",
    "            \n",
    "        return pred \n",
    "  \n",
    "    def predict(self, X): \n",
    "        # returns the predictions for multiple examples X\n",
    "        Y = []\n",
    "        for x in X:\n",
    "            res = self.model(x)\n",
    "            Y.append(res)\n",
    "        return np.array(Y)\n",
    "    \n",
    "    def fit(self, X, Y, epochs = 500, learning_rate = .01):\n",
    "        accuracy = {}\n",
    "        wt_matrix = []\n",
    "        max_accuracy = 0\n",
    "        self.w = np.ones(X.shape[1])\n",
    "        self.b = 0\n",
    "        \n",
    "        for i in tqdm(range(epochs)):\n",
    "            for x, y in zip(X, Y):\n",
    "                y_pred = self.model(x)\n",
    "                \n",
    "                # Learning\n",
    "                if y == 1 and y_pred == 0:\n",
    "                    self.w = self.w + learning_rate * x \n",
    "                    self.b = self.b - learning_rate * 1 \n",
    "                elif y == 0 and y_pred == 1:\n",
    "                    self.w = self.w - learning_rate * x \n",
    "                    self.b = self.b + learning_rate * 1 \n",
    "          \n",
    "                wt_matrix.append(self.w)    \n",
    "            \n",
    "                accuracy[i] = accuracy_score(self.predict(X), Y)\n",
    "        \n",
    "            if (accuracy[i] > max_accuracy):\n",
    "                max_accuracy = accuracy[i]\n",
    "                chkptw = self.w\n",
    "                chkptb = self.b\n",
    "        \n",
    "        self.w = chkptw \n",
    "        self.b = chkptb \n",
    "        print(max_accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Initialize and train Perceptron (Method 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1000/1000 [08:41<00:00,  1.92it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.998046875\n",
      "0.9122807017543859\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "# Initialize Perceptrion\n",
    "perceptron = Perceptron()\n",
    "\n",
    "# Learning\n",
    "perceptron.fit(X_train, Y_train, 1000, 0.1)\n",
    "\n",
    "# Predict X_test Results\n",
    "Y_pred_test = perceptron.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['Perceptron_M1'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 2: SKLearn's Perceptron Import "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9122807017543859\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import Perceptron\n",
    "\n",
    "# Train Perceptron\n",
    "p = Perceptron(random_state = 42) # default epoch=1000, # default learning rate .1\n",
    "p.fit(X_train, Y_train)\n",
    "\n",
    "# Print Accuracy\n",
    "Y_pred_test = p.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['Perceptron_M2'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 3: SKLearn's Logistic Regression "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9649122807017544\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# Train Logistic Regression Model\n",
    "log = LogisticRegression(random_state = 42) \n",
    "log.fit(X_train, Y_train)\n",
    "\n",
    "# Print Accuracy\n",
    "Y_pred_test = log.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['LogRegression'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 4: SKLearn's K Nearest Neighbors "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9824561403508771\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "# Train K Nearest Neighbor Model\n",
    "knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)\n",
    "knn.fit(X_train, Y_train)\n",
    "\n",
    "# Print Accuracy\n",
    "Y_pred_test = knn.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['KNN'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 5: SKLearn's Support Vector Machine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9473684210526315\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "\n",
    "# Train K Nearest Neighbor Model\n",
    "svm = SVC(kernel = 'linear', random_state = 42)\n",
    "svm.fit(X_train, Y_train)\n",
    "\n",
    "# Print Accuracy\n",
    "Y_pred_test = svm.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['SVM'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 6: Kernel SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9649122807017544\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "\n",
    "# Train K Nearest Neighbor Model\n",
    "svm_k = SVC(kernel = 'rbf', random_state = 42)\n",
    "svm_k.fit(X_train, Y_train)\n",
    "\n",
    "# Print Accuracy\n",
    "Y_pred_test = svm_k.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['Kernel_SVM'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 7: SKLearn's Naive Bayes "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9473684210526315\n"
     ]
    }
   ],
   "source": [
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "# Train K Nearest Neighbor Model\n",
    "gaus = GaussianNB()\n",
    "gaus.fit(X_train, Y_train)\n",
    "\n",
    "# Print Accuracy\n",
    "Y_pred_test = gaus.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['NaiveBayes'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 8: SKLearn's Decision Tree "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8771929824561403\n"
     ]
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "# Train K Nearest Neighbor Model\n",
    "dec = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)\n",
    "dec.fit(X_train, Y_train)\n",
    "\n",
    "# Print Accuracy\n",
    "Y_pred_test = dec.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['DecTree'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Method 9: SKLearn's Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9298245614035088\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Train K Nearest Neighbor Model\n",
    "rm = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)\n",
    "rm.fit(X_train, Y_train)\n",
    "\n",
    "# Print Accuracy\n",
    "Y_pred_test = rm.predict(X_test)\n",
    "acc = accuracy_score(Y_pred_test, Y_test)\n",
    "print(acc)\n",
    "\n",
    "# Confusion Matrix\n",
    "cm = confusion_matrix(Y_test, Y_pred_test)\n",
    "\n",
    "# Add to results\n",
    "results['RandomForest'] = [acc,cm]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'Perceptron_M1': [0.9122807017543859, array([[19,  2],\n",
      "       [ 3, 33]], dtype=int64)], 'Perceptron_M2': [0.9122807017543859, array([[20,  1],\n",
      "       [ 4, 32]], dtype=int64)], 'LogRegression': [0.9649122807017544, array([[20,  1],\n",
      "       [ 1, 35]], dtype=int64)], 'KNN': [0.9824561403508771, array([[20,  1],\n",
      "       [ 0, 36]], dtype=int64)], 'SVM': [0.9473684210526315, array([[20,  1],\n",
      "       [ 2, 34]], dtype=int64)], 'Kernel_SVM': [0.9649122807017544, array([[20,  1],\n",
      "       [ 1, 35]], dtype=int64)], 'NaiveBayes': [0.9473684210526315, array([[20,  1],\n",
      "       [ 2, 34]], dtype=int64)], 'DecTree': [0.8771929824561403, array([[20,  1],\n",
      "       [ 6, 30]], dtype=int64)], 'RandomForest': [0.9298245614035088, array([[20,  1],\n",
      "       [ 3, 33]], dtype=int64)]}\n"
     ]
    }
   ],
   "source": [
    "print(results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Perceptron_M1 \tacc: 91.22807017543859\n",
      "[[19  2]\n",
      " [ 3 33]]\n",
      "\n",
      "Perceptron_M2 \tacc: 91.22807017543859\n",
      "[[20  1]\n",
      " [ 4 32]]\n",
      "\n",
      "LogRegression \tacc: 96.49122807017544\n",
      "[[20  1]\n",
      " [ 1 35]]\n",
      "\n",
      "KNN \t\tacc: 98.24561403508771\n",
      "[[20  1]\n",
      " [ 0 36]]\n",
      "\n",
      "SVM \t\tacc: 94.73684210526315\n",
      "[[20  1]\n",
      " [ 2 34]]\n",
      "\n",
      "Kernel_SVM \tacc: 96.49122807017544\n",
      "[[20  1]\n",
      " [ 1 35]]\n",
      "\n",
      "NaiveBayes \tacc: 94.73684210526315\n",
      "[[20  1]\n",
      " [ 2 34]]\n",
      "\n",
      "DecTree \tacc: 87.71929824561403\n",
      "[[20  1]\n",
      " [ 6 30]]\n",
      "\n",
      "RandomForest \tacc: 92.98245614035088\n",
      "[[20  1]\n",
      " [ 3 33]]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "for key, val in results.items():\n",
    "    if len(key) < 4:\n",
    "        print(key,'\\t\\tacc:', val[0]*100)\n",
    "        print(val[1])\n",
    "        print()\n",
    "    else:\n",
    "        print(key,'\\tacc:', val[0]*100)\n",
    "        print(val[1])\n",
    "        print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Done"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
