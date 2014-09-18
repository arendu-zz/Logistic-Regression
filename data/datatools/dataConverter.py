'''
Created on Jun 17, 2013

@author: arenduchintala
'''

import numpy as np
'''
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

'''
def binarize(col):
    value_types = sorted(list(set(col)))
    expanded_col = np.zeros((len(col),len(value_types)))
    for i in range(len(col)):
        j = value_types.index(col[i] )
        expanded_col[i][j] = 1    
    return expanded_col.astype(np.int8)

def normalize(col):
    col = col.astype(np.float32)
    return np.matrix(col / 1.0).T

file = '../adult/adult.train'
lines = np.genfromtxt(file, dtype=None, delimiter=',', names=True)
age = normalize(lines['age'])
workclass = binarize(lines['workclass'])
fnlwgt = normalize(lines['fnlwgt'])
education = binarize(lines['education'])
education_num= normalize(lines['education_num'])
marital_status= binarize(lines['marital_status'])
occupation= binarize(lines['occupation'])
relationship= binarize(lines['relationship'])
race= binarize(lines['race'])
sex= binarize(lines['sex'])
capital_gain = normalize(lines['capital_gain'])
capital_loss = normalize(lines['capital_loss'])
hours_per_week = normalize(lines['hours_per_week'])
native_country = binarize(lines['native_country'])
print lines['label'][9].strip(), lines['label'][90].strip()
l = [ 0 if (x.strip() == '>50K') else 1 for x in lines['label']]
label = np.matrix(l).T
print l[9], label[9], l[90],label[90]
full = np.hstack((age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,label))
print np.shape(full)
np.savetxt(file+'.num', full,delimiter=',',fmt='%10.4f')


