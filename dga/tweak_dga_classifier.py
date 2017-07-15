import commands
import json

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

model_config_filepath = "config/model_config.json"

def model_config(classifier_type, hlayers, nodes, node_type):
	json_data = json.load(open(model_config_filepath))
	classifier_data = json_data[classifier_type]
	classifier_data["hidden_layers"] = []
	for i in range(0, hlayers):
		classifier_data["hidden_layers"].append({"num": nodes[i], "type": node_type[i]})
	open(model_config_filepath, "w").write(json.dumps(json_data, indent=4))

def run_test(t, s, me, nf, bs):
	# run_conf = ("multi", 1000, 100, 10, 128)
	run_conf = (t, s, me, nf, bs)
	json_data = json.load(open(model_config_filepath))[run_conf[0]]
	print(json.dumps(json_data, indent=4))

	print "python dga_classifier.py %s %s %s %s %s" % (run_conf[0], str(run_conf[1]), str(run_conf[2]), str(run_conf[3]), str(run_conf[4]))
	output = commands.getoutput("python dga_classifier.py %s %s %s %s %s" % (run_conf[0],
																			 str(run_conf[1]),
																			 str(run_conf[2]),
																			 str(run_conf[3]),
																			 str(run_conf[4])))

	print output
	endstring = "CLASSIFIER AVERAGE FINAL SCORE: "
	print bcolors.OKGREEN + bcolors.UNDERLINE + "*** tweak_dga_classifier.py DONE ***" + bcolors.ENDC
	auc_score = float(output[output.rfind(endstring)+endstring.__len__():])	
	return auc_score

auc_scores = []
classifier_type = "multi"
run_type = "binary_search"
hlayers = 2
node_type = ["relu", "tanh"]*(hlayers/2)

if run_type == "simple":
	nodes = [250, 500, 750, 1500]
	for n in nodes:
		print bcolors.OKGREEN + "Running Simple with Nodes " + str(n) + bcolors.ENDC
		model_config(classifier_type, hlayers, [500, n], node_type)
		auc_scores.append((n, run_test(classifier_type, 1000, 100, 10, 128)))
elif run_type == "binary_search":
	lo = 0
	hi = 200
	it = 0
	default = 0.968008565699
	while (lo < hi):
		it += 1
		mid = (lo + hi)/2
		print bcolors.OKGREEN + "Running Binary Search with Nodes=" + str(mid) + " with Iteration=" + str(it) + bcolors.ENDC
		model_config(classifier_type, hlayers, [500, mid], node_type)
		val = run_test(classifier_type, 1000, 150, 10, 128)
		auc_scores.append((mid, val))
		if val <= default:
			lo = mid
		elif val > default:
			hi = mid-1

print bcolors.OKBLUE + bcolors.UNDERLINE + str(auc_scores) + bcolors.ENDC