import os
import json

param_path = os.path.join('flcore/params','coauthor_cs.json')
content = json.loads(open(param_path).read())
print(content)