from prim_search import *


naive = True
print(("PrIM" if naive else "PrIM-Search") + " search for Tensor programs")
for task in poly_tasks:
    if not task[0]:
        continue
    if args.skip_existing and search_param_exists(*task, naive, jsonfile=args.jsonfile):
        continue
    print(task)
    params = search(*task, naive=naive)
    save_search_params(*task, params, naive)
print(("PrIM" if naive else "PrIM-Search") + " search for GPT-J")
for task in gptj_tasks:
    if args.skip_existing and search_param_exists(*task, naive, jsonfile=args.jsonfile):
        continue
    print(task)
    params = search(*task, naive=naive)
    save_search_params(*task, params, naive, jsonfile=args.jsonfile)
