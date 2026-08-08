import joblib, os
p = r'c:\projects\cancer-paper-mill\models\tfidf_combined.joblib.bak_pre_audit_C15'
obj = joblib.load(p)
clf = obj.get('classifier')
if clf and hasattr(clf, 'get_params'):
    params = clf.get_params()
    print(f'Classifier C: {params.get("C")}')
    print(f'Classifier class_weight: {params.get("class_weight")}')
else:
    print('No classifier or get_params')
