import joblib, os, datetime
p = r'c:\projects\cancer-paper-mill\models\tfidf_combined.joblib.bak_pre_audit_C15'
fi_ctime = os.path.getctime(p)
fi_mtime = os.path.getmtime(p)
print('BAK FILE timestamps:')
print(f'  Creation time : {datetime.datetime.fromtimestamp(fi_ctime)}')
print(f'  Modified time : {datetime.datetime.fromtimestamp(fi_mtime)}')
print(f'  Size          : {os.path.getsize(p)} bytes')
obj = joblib.load(p)
print(f'Type: {type(obj)}')
if hasattr(obj, 'get_params'):
    print(f'Params: {obj.get_params()}')
if hasattr(obj, 'named_steps'):
    for name, step in obj.named_steps.items():
        print(f'  Step [{name}]: {type(step).__name__}')
        if hasattr(step, 'get_params'):
            print(f'    Params: {step.get_params()}')
