import os
import math
import scipy.optimize
import numpy as np
import string
import random
import gzip

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn import linear_model

from predict_read_task import *
from predict_cat_task import *
from predict_rating_task import *

if __name__ == "__main__":
  # predictRating()
  # predictRead()
  predictCat()