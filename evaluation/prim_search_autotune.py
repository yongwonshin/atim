from prim_search import *
from parser_utils import get_tune_parser, args_to_tasks


parser = get_tune_parser()
parser.add_argument("--skip_existing", action="store_true", help="Skip tasks where search parameters already exist")
parser.add_argument("--jsonfile", type=str, default="./reproduced/prim_parameters.json")
args = parser.parse_args()
tasks = args_to_tasks(args)

naive = False
for task in tasks:
    if not task[0]:
        continue
    if args.skip_existing and search_param_exists(*task, naive, jsonfile=args.jsonfile):
        continue
    print(task)
    params = search(*task, naive=naive)
    save_search_params(*task, params, True, jsonfile=args.jsonfile)

