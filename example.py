import test_YY
import imaging
import numpy as np
import numpy.linalg as la

separation = .5
nballs = 3
pts_per_ball = 5
#get a random Cluster_Pblm object.
pblm = test_YY.stoch_ball_pblm(nballs, pts_per_ball, separation) 


print("---tr value for Lloyd's algorithm results (10 runs)---")
best_value = np.inf
for j in range(10):
    Y = pblm.run_lloyd()
    tr_value = pblm.tr(Y)
    best_value = min(best_value, tr_value)
    grad_tr_norm = la.norm(pblm.gr_tr_projected(Y))
    print("tr(DYY^T): " + str(tr_value) + ", Norm Gradient: " + str(grad_tr_norm))

ts = np.arange(0, 1, .1)
As = (1-ts)
Bs = pblm.Dsize*ts

print("---Running homotpy, and saving with suffix test_pblm---")
print("---Ensure that the directory data exists---")
file_suffix = "test_pblm"
test_YY.gen_path(pblm, As, Bs, file_suffix)

print("---Generating random local minimizers---")
#For a more interesting plot you may want to use more t-values.
num_runs_per_pair = 10
test_YY.gen_many_mins(pblm, As, Bs, num_runs_per_pair, suffix=file_suffix)

fig = imaging.plt.figure()
imaging.single_plot_file(fig, "test_pblm", lloyd = best_value)
