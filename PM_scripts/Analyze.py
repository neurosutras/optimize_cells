import os 
import numpy as np
import itertools as it
from nested.optimize import StorageModelReport as OptRep
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gs
import matplotlib.colors as colors
import pprint

class Analysis():
    def __init__(self, dir_path='.', file_string='history.hdf5', filename='test.pdf'):
        self.dir_path = dir_path 
        self.file_arr = self.get_file_arr(file_string) 
        self.filename = filename
#        self.obj_lst = ['soma R_inp', 'f_I_residuals', 'spike_adaptation_residuals']
#        self.obj_lst = ['soma R_inp', 'f_I_log10_slope', 'ADP']
#        self.obj_lst= ['term_dend R_inp', 'soma R_inp', 'soma R_inp (no h)', 'soma vm_rest', 'soma vm_rest (no h)', 'vm_th', 'ADP', 'fAHP', 
#                        'rebound_firing', 'vm_stability', 'ais_delay', 'slow_depo', 'f_I_residuals']
        if self.file_arr.shape[0] == 1:
            self.obj_lst = np.array(self.file_arr[0,1].objective_names, dtype='U').tolist()
        else:
            self.obj_lst = [
                            'term_dend R_inp',
                            'soma R_inp',
      #                      'soma R_inp (no h)',
                            'soma vm_rest',
      #                      'soma vm_rest (no h)',
                            'vm_th',
      #                      'ADP',
      #                      'fAHP',
      #                      'rebound_firing',
                            'vm_stability',
                            'ais_delay',
                            'slow_depo',
                            'f_I_residuals',
                            ]

        self.model_lst = self.obj_lst
        self.best_mod = True
#        self.best_mod = False 
#        self.obj_vals = self.get_objective_vals(self.obj_lst, self.model_lst)
        self.obj_vals = self.get_hack_objective_vals(self.obj_lst, self.model_lst, norm=False)


#        self.ax_arr = self.get_axes(self.obj_vals.shape)
#        self.plot_objective_errors(self.obj_vals, self.ax_arr)

        self.plot_single_file_matrix()
    

    def get_file_arr(self, file_string):
        self.file_lst = [(os.path.join(self.dir_path, f), f.strip(file_string)) for f in os.listdir(self.dir_path) if f.endswith(file_string)] 
        self.N_fil = len(self.file_lst)
        file_arr = np.empty(shape=(self.N_fil, 3), dtype='O')
        for idx, fil in enumerate(self.file_lst):
            rep = OptRep(file_path=fil[0])
            file_arr[idx,:] = fil, rep, rep.get_best_model() 
        return file_arr

    def get_objective_vals(self, obj_lst, model_lst):
        self.N_objectives = len(obj_lst)
        self.N_models = len(model_lst)
        obj_vals = np.empty(shape=(self.N_fil, self.N_models+self.best_mod, self.N_objectives))
        for idx, rep in enumerate(self.file_arr[:,1]):
            obj_idx = self.get_unsort_subset_idx(rep.objective_names, obj_lst) 



#            spe_idx = self.get_unsort_subset_idx(rep.specialist_arr['specialist'], model_lst, encode=True) 
#            spe_idx = self.get_unsort_subset_idx(rep.objective_names, model_lst, encode=True) 
#            print(spe_idx)
#            mod_idx = rep.specialist_arr['model_pos'][spe_idx]
            obj_vals[idx, :self.N_models, :] = rep.spe_objectives[np.ix_(mod_idx, obj_idx)]

            if self.best_mod:
                bidx = rep.best_pos 
                b_arr = rep.spe_objectives[bidx, obj_idx] if rep.best_spe else rep.sur_objectives[bidx, obj_idx]
                obj_vals[idx, -1, :] = b_arr 

        return obj_vals

    def get_hack_objective_vals(self, obj_lst, model_lst, norm=False):
        self.N_objectives = len(obj_lst)
        self.N_models = len(model_lst)
        obj_vals = np.empty(shape=(self.N_fil, self.N_models+self.best_mod, self.N_objectives))
        att = 'normalized_objectives' if norm else 'objectives'
        for idx, rep in enumerate(self.file_arr[:,1]):
            obj_idx = self.get_unsort_subset_idx(rep.objective_names, obj_lst, encode=True) 
            spec_vals = rep.get_category_att(att=att)
            for iobj, obj in enumerate(obj_idx):
                obj_vals[idx, iobj, :] = spec_vals[obj, obj_idx] 
            if self.best_mod:
                bst = rep.get_best_model()
                obj_vals[idx, -1, :] = bst[-1][obj_idx] 
        return obj_vals


    def plot_single_file_matrix(self, idx=None):

#        mat_lst = list(self.obj_vals.shape[0]) if idx is None else idx
#        N_mat_lst = len(mat_lst)
#        N_rows = np.sqrt(N_mat_lst)
        eps = np.finfo(float).eps
        mat = self.obj_vals[0,:,:]
        N_mod, N_obj = mat.shape

        norm_mat = np.empty(shape=mat.shape)
        for i in range(N_obj):
            max_vec = mat[:,i].max()
            min_vec = mat[:,i].min()
            rng = max_vec - min_vec + eps
            norm_mat[:,i] = (mat[:,i] - min_vec)/rng + eps 

        fig = plt.figure(figsize=(N_mod, N_obj))
        gl = gs.GridSpec(1,1)
        model_cat = self.obj_lst + ['best']
        
        ax = fig.add_subplot(gl[0,0])
        axobj = ax.matshow(norm_mat, norm=colors.LogNorm(vmin=norm_mat.min(), vmax=norm_mat.max()), cmap='jet', aspect='equal')
#        axobj = ax.matshow(mat, norm=colors.LogNorm(vmin=mat.min(), vmax=mat.max()), cmap='jet')
#        axobj = ax.matshow(norm_mat, cmap='jet')


     #   ax.set_yticklabels(labels=['']+model_cat)
        ax.set_xticks(range(N_obj))
        ax.set_yticks(range(N_mod))
        ax.set_yticklabels(labels=model_cat)
        ax.set_xticklabels(labels=self.obj_lst, rotation=90)
        ax.set_xlabel('Objectives')
        ax.set_ylabel('Specialists')
        ax.set_title('Normalized Objective Errors')

        fig.colorbar(axobj, ax=ax)

        plt.tight_layout()
        plt.savefig(self.filename)



    def get_axes(self, val_shape, rows_sharey=True):
        N_exp, N_mod, N_obj = val_shape
        fig = plt.figure(figsize=(0.5*N_exp*N_mod, 2*N_obj)) 
        gl = gs.GridSpec(N_obj, N_exp)
        ax_arr = np.empty(shape=gl.get_geometry(), dtype='O') 
        for axidx, ax in np.ndenumerate(ax_arr):
            ax_arr[axidx] = fig.add_subplot(gl[axidx])     
#            ax_arr[axidx].set_yscale('log')
            ax_arr[axidx].ticklabel_format(axis='y', style='sci', scilimits=(0,1), useMathText=True)
        for i in range(N_obj):
            for ax in ax_arr[i,1:]:
                ax_arr[i,0].get_shared_y_axes().join(ax_arr[i,0], ax)
        for j in range(N_exp):
            for ax in ax_arr[:-1,j]:
                ax_arr[-1,j].get_shared_x_axes().join(ax_arr[-1,j], ax)
                ax.set_xticklabels([])
        return ax_arr

    def plot_objective_errors(self, vals, ax_arr):
    #    model_cat = self.model_lst + ['best'] if self.best_mod else self.model_lst
     #   model_cat = ['somaRinp', 'FI_res', 'spkad_res', 'best'] 
        model_cat = self.obj_lst + ['best']

        for (obidx, exidx), ax in np.ndenumerate(ax_arr):
            ax.plot(model_cat, vals[exidx, :, obidx])
        
        for objidx, obj in enumerate(self.obj_lst):
            ax_arr[objidx,0].set_ylabel(obj)

        for exidx, exp in enumerate(self.file_lst):
            ax_arr[0, exidx].set_title(exp[1])
            ax_arr[-1, exidx].set_xticklabels(labels=model_cat, rotation=90)

        plt.tight_layout()
        plt.savefig(self.filename)

    def get_unsort_subset_idx(self, main_lst, subset, encode=False):
        sorter = np.argsort(main_lst)
        if encode:
            subset = np.array(subset, dtype='S')
        if np.intersect1d(main_lst, subset).size == subset.size:
            idx = sorter[np.searchsorted(main_lst, subset, sorter=sorter)]
        else:
            print('Incompatible query')
        return idx
        
        
if __name__=='__main__':
    Analysis('data/history', file_string='20200521_032449_DG_MC_leak_spiking_cell1000000_PopulationAnnealing_optimization_history.hdf5',  filename='Analyze/Results/NewMorphLS.pdf')
