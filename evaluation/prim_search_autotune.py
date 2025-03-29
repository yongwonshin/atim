from prim_search import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--skip_existing", action="store_true", help="Skip tasks where search parameters already exist")
parser.add_argument("--jsonfile", type=str, default="./reproduced/prim_parameters.json")
args = parser.parse_args()


naive = False
print(("PrIM" if naive else "PrIM-Search") + " search for Tensor programs")
for task in poly_tasks:
    if not task[0]:
        continue
    if args.skip_existing and search_param_exists(*task, naive, jsonfile=args.jsonfile):
        continue
    print(task)
    params = search(*task, naive=naive)
    save_search_params(*task, params, naive, jsonfile=args.jsonfile)
print(("PrIM" if naive else "PrIM-Search") + " search for GPT-J")
for task in gptj_tasks:
    if args.skip_existing and search_param_exists(*task, naive, jsonfile=args.jsonfile):
        continue
    print(task)
    params = search(*task, naive=naive)
    save_search_params(*task, params, naive, jsonfile=args.jsonfile)
