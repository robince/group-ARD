#%%
from groupard import GroupARDRegression
from sklearn.linear_model import BayesianRidge, LinearRegression, ARDRegression
from sklearn.model_selection import cross_validate, cross_val_score, KFold, RepeatedKFold

# %%
mdl = GroupARDRegression(prior='GroupARD',groups=[0,1],verbose=True)
mdl.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
print(mdl.coef_)
mdl.predict([[1,1]])


#%%
from sklearn import datasets
X, y = datasets.load_diabetes(return_X_y=True)
cv = RepeatedKFold(n_splits=5,n_repeats=10)
# cv = KFold(5,shuffle=True)

print(cross_val_score(LinearRegression(),X,y,cv=cv).mean())
print()
print(cross_val_score(BayesianRidge(),X,y,cv=cv).mean())
print(cross_val_score(GroupARDRegression(prior='Ridge'),X,y,cv=cv).mean())
print()
print(cross_val_score(ARDRegression(),X,y,cv=cv).mean())
print(cross_val_score(GroupARDRegression(prior='ARD'),X,y,cv=cv).mean())
print()
print(cross_val_score(GroupARDRegression(prior='GroupARD',groups=[0]*10),X,y,cv=cv).mean())
print(cross_val_score(GroupARDRegression(prior='GroupARD',groups=[0,1,2,3,4,5,6,7,8,9]),X,y,cv=cv).mean())
print(cross_val_score(GroupARDRegression(prior='GroupARD',groups=[0,0,0,0,0,1,1,1,1,1]),X,y,cv=cv).mean())