#%%
from groupard import GroupARDRegression
from sklearn.model_selection import cross_validate, cross_val_score, KFold, RepeatedKFold

# %%
mdl = GroupARDRegression(prior='Group ARD',groups=[0,1],verbose=True)
mdl.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
print(mdl.coef_)
mdl.predict([[1,1]])