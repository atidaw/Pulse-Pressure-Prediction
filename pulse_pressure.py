import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
X = []
Y = []

for line in open('C:\pycharm\kk.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2)])
    Y.append(float(y))


df2=pd.DataFrame(X,columns=['upper','lower'])
df2['Pulse_Pressure']=pd.Series(Y)
df2


model = smf.ols(formula='Pulse_Pressure ~ upper - lower', data=df2)
results_formula = model.fit()
results_formula.params


x_surf, y_surf = np.meshgrid(np.linspace(df2.upper.min(),
                                         df2.upper.max(), 100),
                                         np.linspace(df2.lower.min(),
                                         df2.lower.max(), 100))


onlyX = pd.DataFrame({'upper': x_surf.ravel(),
                      'lower': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)



fittedY=np.array(fittedY)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['upper'],
           df2['lower'],
           df2['Pulse_Pressure'],
           c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,
                y_surf,fittedY.reshape(x_surf.shape),
                color='b', alpha=0.3)
ax.set_xlabel('upper')
ax.set_ylabel('lower')
ax.set_zlabel('Pulse_Pressure')

plt.show()
