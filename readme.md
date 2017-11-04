## My messy quick implementations of some models for personal use.
I'd be embarrassed if someone actually looked at this messy code -
__not for public consumption!!!__  Just some constantly-evolving ML model wrappers for
quick & dirty work.  Note code is not documented and/or contains comments leftover from cut/pasted code and are often no longer relevant.  Maybe someday I'll clean this up.

For self-reminder, typical usage:

```python
from ag_implement import use_model, summarize_regrmodels, Keras_MLP as mlp
# X_train, y_train, X_test, y_test are all np.ndarrays
# Calls are interchangable for Keras_MLP, TF_MLP, SKL_MLP, just swap the model

results=[]
metrics,y_testpred,y_trainpred = use_model(mlp(lr=0.0025, epochs=100, hlay=[20,20,20], alpha=0.),
                                            X_train, y_train, X_test, y_test)
results.append(metrics)

# [some other use_model() and results.append() runs with other models...]

summarize_regrmodels(results)  # lists out one results row per use_model call
```
