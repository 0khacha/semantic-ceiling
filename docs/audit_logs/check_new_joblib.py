import joblib, os
p = r'c:\projects\cancer-paper-mill\models\tfidf_combined.joblib'
obj = joblib.load(p)
clf = obj.get('classifier')
if clf and hasattr(clf, 'get_params'):
    params = clf.get_params()
    print(f'NEW Classifier C: {params.get("C")}')
else:
    print('No classifier or get_params')
