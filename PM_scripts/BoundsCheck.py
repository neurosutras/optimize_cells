import sys, yaml
import numpy as np
from nested.optimize_utils import StorageModelReport as smr
import matplotlib.pyplot as plt

def run_test(file_name, config_file, plot_file):

    report = smr(file_name)
    params = np.array(report.param_names, dtype='U')    
    objectives = np.array(report.objective_names, dtype='U')
    spe_params = report.get_category_att()
    _, best_mod, _, _ = report.get_best_model()
    report.close_file()

    with open(config_file, 'r') as config_data:
        config_data_dict = yaml.full_load(config_data)

    param_bounds = config_data_dict['bounds']
    param_bds_arr = np.empty(shape=(report.N_params, 3))
    param_logbds_arr = np.empty(shape=(report.N_params, 3))

    for kidx, key in enumerate(params):
        param_bds_arr[kidx,:2] = param_bounds[key]

    param_bds_arr[:,2] = param_bds_arr[:,1] - param_bds_arr[:,0] 

    param_logbds_arr[:,:2] = np.log(param_bds_arr[:,:2])
    param_logbds_arr[:,2] = param_logbds_arr[:,1] - param_logbds_arr[:,0] 

    param_par_arr = np.empty(shape=(report.N_objectives+1, report.N_params, 2))

    for i in range(report.N_objectives):
        param_par_arr[i,:,0] = (spe_params[i,:] - param_bds_arr[:,0])/param_bds_arr[:,2]
        param_par_arr[i,:,1] = (np.log(spe_params[i,:]) - param_logbds_arr[:,0])/param_logbds_arr[:,2]

    param_par_arr[-1,:,0] = (best_mod - param_bds_arr[:,0])/param_bds_arr[:,2]
    param_par_arr[-1,:,1] = (np.log(best_mod) - param_logbds_arr[:,0])/param_logbds_arr[:,2]

    fig, subplots = plt.subplots(nrows=report.N_objectives+1, sharex=True, sharey=True, figsize=(report.N_params, 2*(report.N_objectives+1)))
    for i in range(report.N_objectives):
        subplots[i].plot(params, param_par_arr[i,:,0], 'b.')
        subplots[i].plot(params, param_par_arr[i,:,1], 'r.')
        subplots[i].set_title(objectives[i])
        subplots[i].axhline(0.05, ls='--', color='k')
        subplots[i].axhline(0.95, ls='--', color='k')

    subplots[-1].plot(params, param_par_arr[-1,:,0], 'b.')
    subplots[-1].plot(params, param_par_arr[-1,:,1], 'r.')
    subplots[-1].set_title('best')
    subplots[-1].axhline(0.05, ls='--', color='k')
    subplots[-1].axhline(0.95, ls='--', color='k')
    plt.xticks(rotation=90)


    

    fig.tight_layout()
    fig.savefig('{!s}.pdf'.format(plot_file))

#    print(param_par_arr)

    param_dec_arr = np.empty(shape=(report.N_objectives, report.N_params, 2), dtype=np.bool)

        



if __name__=='__main__':

    file_name = sys.argv[1]
    config_file = sys.argv[2]
    plot_file = sys.argv[3]
    run_test(file_name, config_file, plot_file)
