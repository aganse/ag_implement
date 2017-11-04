import numpy as np
from timeit import default_timer as timer

from sklearn.metrics import mean_squared_error, r2_score


def use_model(model, X_train, y_train, X_test, y_test, notes=''):
    # wrapper function to fit/train/test a specified model to the dataset
    # and return list of performance metrics
    Tstart = timer()
    modname = model.__class__.__name__
    losshist = model.fit(X_train, y_train)
    y_trainpred = model.predict(X_train)
    y_testpred = model.predict(X_test)
    # could test modname and if regression model do mse and if classification do acc/prec/etc
    mse_train = mean_squared_error(y_train, y_trainpred)
    mse_test = mean_squared_error(y_test, y_testpred)
    r2 = r2_score(y_test, y_testpred)
    Tend = timer()
    if (modname=='Keras_MLP') or (modname=='TF_MLP'):
        endloss = losshist[-1]
        p = model.get_params()
        results = (modname, endloss, mse_train, mse_test, r2, p[0], p[1],','.join(str(e) for e in p[2]), p[3])
        print('%s:  FinLoss=%5.3f  TrainMSE=%5.3f  TestMSE=%5.3f  R^2=%5.3f  lr=%6.4f  epochs=%d  hlay=%s  a=%g' % results)
    else:
        results = (modname, np.nan, mse_train, mse_test, r2, np.nan, np.nan, np.nan, np.nan)
        print('%s:  TrainMSE=%5.3f  TestMSE=%5.3f  R^2=%5.3f' % tuple([results[i] for i in [0,2,3,4]]))
    print('Elapsed time (seconds): %-12.6f' % (Tend - Tstart))
    return results, y_testpred, y_trainpred


def summarize_regrmodels(results):
    # list out all results/metrics for each of the regression model runs contained in 'results'
    import pandas as pd
    pd.set_option('precision',4)
    pd.set_option('expand_frame_repr', False)
    pd.set_option('display.max_colwidth', 100)
    pd.set_option('display.max_columns', 20)
    df = pd.DataFrame(results,
            columns=['model_name','final_loss','trainingMSE','testingMSE','R^2','learn_rate','epochs','hlay','alpha'])
    print(df.to_string(index=True, na_rep='-'))


#def summarize_classifmodels(results):
    # list out all results/metrics for each of the classification model runs contained in 'results'
