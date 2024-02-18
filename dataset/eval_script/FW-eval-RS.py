#!/usr/bin/python
"""
@author: Thomas Demeester (for more information: thomas.demeester@intec.ugent.be)

This file is part of the FedWeb Greatest Hits collection.
Goal:  
This simple (and not fool-proof) script allows calculating basic TREC FedWeb evaluation measures (nDCG@k, nP@k) for RESOURCE SELECTION
and serves as an example for writing your own evaluation scripts.
"""

usage="""
Usage: following input arguments are required (in this order)
* RUN: location to runfile to be evaluated
* QRELS: location to appropriate qrels file (FW13-QRELS-RS.txt or FW14-QRELS-RS.txt)
* TRECEVAL: location of folder with trec_eval executable (e.g., /usr/local/lib/trec_eval.9.0)
"""

import sys
import subprocess
from os import path
import numpy as np


"""input"""
args = sys.argv
if not len(args)==4 :
    sys.exit()
runfile,qrelsfile,trecevalpath = args[1:]

"""initialize"""
metrics = []
RESULTS = {}

"""nDCG@k"""
treceval_script = path.join(trecevalpath,'trec_eval')
for k in [10,20,100]:
    metr = 'nDCG@%d'%k
    metrics.append(metr)
    out = subprocess.check_output('%s -q -m ndcg_cut.%d %s %s'%(treceval_script,k,qrelsfile,runfile), shell=True).decode('utf-8')
    for line in out.strip().split('\n')[:-1]:
        [m,qID,v] = line.split()
        if not qID in RESULTS:
            RESULTS[qID] = {}
        RESULTS[qID][metr] = v


"""  """

#ratio of precision for predicted search engine, divided by max. precision for that query (from best search engine)
def calc_nPatk(runfile,qrelsfile,kvalue):

    idealscores = {}
    predictedscores = {}
    nPatk = {}
    runfile_query_ids = set()
    with open(runfile,'r') as runID:
        for line in runID:
            parts = line.strip().split()
            qID = parts[0]
            runfile_query_ids.add(qID)
            SEID = parts[2]
            if not qID in predictedscores:
                predictedscores[qID] = {}
            #if float(parts[4]) > 0.5:
            predictedscores[qID][SEID] = float(parts[4])

    with open(qrelsfile,'r') as qrels:
        for line in qrels:
            (qID,zero,SEID,score) = line.strip().split()
            if qID not in runfile_query_ids:
                continue

            if not qID in idealscores:
                idealscores[qID] = {}
            idealscores[qID][SEID] = float(score)
    #watch out: if no results were returned for specific SEID for this query, then it is not in idealscores
    #however: score should be zero, obviously

        for qID in idealscores:
            try:
                for SEID in predictedscores[qID]:
                    if not SEID in idealscores[qID]:
                        idealscores[qID][SEID] = 0.
            except:
                continue
        for qID in sorted(idealscores.keys()): #not predictedscores, because more query results returned than judged queries
            tmp = sorted(predictedscores[qID].items(), key=lambda x: (x[1], x[0]), reverse=True)  # if scores are the same, sort by engine id
            sorted_predictedSEIDs = [t[0] for t in tmp]
            top_predictedSEIDs = sorted_predictedSEIDs[:kvalue]
            tmp = sorted(idealscores[qID].items(), key=lambda x: (x[1], x[0]), reverse=True)
            sorted_idealSEIDs = [t[0] for t in tmp]
            top_idealSEIDs = sorted_idealSEIDs[:kvalue]
            top_predicted_values = [idealscores[qID][SEID] for SEID in top_predictedSEIDs]
            top_ideal_values = [idealscores[qID][SEID] for SEID in top_idealSEIDs]
            nPatk[qID] = np.sum(top_predicted_values)/np.sum(top_ideal_values)


    return nPatk








for k in [1,5]:
    metr = 'nP@%d'%k
    metrics.append(metr)
    nPatk = calc_nPatk(runfile,qrelsfile,k)
    for qID in nPatk:
        RESULTS[qID][metr] = '%.4f'%nPatk[qID]

"""print out"""
print('topic,%s'%','.join([m for m in metrics]))
for qID in sorted(RESULTS.keys()):
    print('%s,%s'%(qID,','.join([RESULTS[qID][metr] for metr in metrics])))
print("There are %d queries in total."%len(RESULTS))
print('%s,%s'%('all',','.join(['%.4f'%np.mean([float(RESULTS[qID][metr]) for qID in RESULTS]) for metr in metrics])))

# except:
#     e = sys.exc_info()[0]
#     print('Exception: \n%s'%e)
#     print(usage)
