from ag_implement.run import use_model, summarize_regrmodels
from ag_implement.nn import TF_MLP, Keras_MLP
from ag_implement.skl_rf import skl_rf, create_tree_plots, plotROC, plotFI
# Typical usage:
# from ag_implement import use_model, summarize_regrmodels, Keras_MLP as mlp
# or
# from ag_implement import skl_rf, create_tree_plots, plotROC, plotFI


# My use with SPutils (in Jupyter cells):
# %autoreload
# from ag_implement import use_model, summarize_regrmodels, Keras_MLP as mlp
#
# results=[]
# metrics,y_testpred,y_trainpred = use_model(mlp(lr=0.0025, epochs=100, hlay=[20,20,20], alpha=0.),
#                                             X_train, y_train, X_test, y_test)
# results.append(metrics)
#
# [some other use_model() and results.append() runs with other models...]
#
# summarize_regrmodels(results)
