import subprocess

for seq in ['magistrale1','magistrale2','magistrale3','magistrale4','magistrale5','magistrale6',\
            'outdoors1','outdoors2','outdoors3','outdoors4','outdoors5','outdoors6','outdoors7','outdoors8']:
    p = subprocess.Popen('python ./evaluation_scripts/evaluate_tumvi.py --batch --seq=%s | grep rmse' % seq, shell = True)
    p.wait()