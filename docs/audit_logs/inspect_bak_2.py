import joblib, os, datetime
p = r'c:\projects\cancer-paper-mill\models\tfidf_combined.joblib.bak_pre_audit_C15'
obj = joblib.load(p)
print(f'Type: {type(obj)}')
if isinstance(obj, dict):
    print(f'Keys: {list(obj.keys())}')
    if 'model' in obj:
        print(f'Model type: {type(obj["model"])}')
        if hasattr(obj["model"], "get_params"):
            print(f'Model params: {obj["model"].get_params()}')
    if 'pipeline' in obj:
         print(f'Pipeline type: {type(obj["pipeline"])}')
         if hasattr(obj["pipeline"], "get_params"):
            print(f'Pipeline params: {obj["pipeline"].get_params()}')
