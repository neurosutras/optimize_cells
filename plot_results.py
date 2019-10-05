__author__ = 'milsteina'
from cell_utils import *
import matplotlib as mpl
import matplotlib.lines as mlines
import scipy.stats as stats
import matplotlib.gridspec as gridspec
from matplotlib import cm
from dentate.synapses import get_syn_mech_param, get_syn_filter_dict

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 12.
# mpl.rcParams['font.size'] = 14.
#mpl.rcParams['font.sans-serif'] = 'Arial'
#mpl.rcParams['font.sans-serif'] = 'Calibri'
mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
mpl.rcParams['text.usetex'] = False
#mpl.rcParams['figure.figsize'] = 6, 4.3
"""
mpl.rcParams['axes.labelsize'] = 'larger'
mpl.rcParams['axes.titlesize'] = 'xx-large'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'x-large'
"""


def plot_Rinp(rec_file_list, sec_types_list=None, features_list=None, features_labels=None, file_labels=None,
              data_dir='data/'):
    """
    Expects each file in list to be generated by parallel_rinp.
    Superimpose features across cells recorded from simulated step current injections to probe input resistance and
    membrane time constant.
    :return:
    """
    orig_fontsize = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 18.
    if isinstance(rec_file_list, str):
        rec_file_list = [rec_file_list]
    if isinstance(sec_types_list, str):
        sec_types_list = [sec_types_list]
    if isinstance(features_list, str):
        features_list = [features_list]
    if isinstance(features_labels, str):
        features_labels = [features_labels]
    if isinstance(file_labels, str):
        file_labels = [file_labels]
    if sec_types_list is None:
        sec_types_list = ['axon', 'apical', 'soma']
    axon_types_list = ['axon', 'ais', 'hillock']
    dend_types_list = ['basal', 'apical', 'trunk', 'tuft']
    if features_list is None:
        features_list = ['Rinp_peak', 'Rinp_baseline', 'Rinp_steady', 'decay_90']
    if features_labels is None:
        features_labels_default_dict = {'Rinp_peak': 'Input resistance - peak (MOhm)', 'Rinp_baseline': 'Baseline Vm (mV)',
                                'Rinp_steady': 'Input resistance - steady-state (MOhm)',
                                'decay_90': 'Membrane time constant (ms)'}
        features_labels_dict = {}
        for feature in features_list:
            if feature in features_labels_default_dict:
                features_labels_dict[feature] = features_labels_default_dict[feature]
            else:
                features_labels_dict[feature] = feature
    else:
        features_labels_dict = {feature: label for (feature, label) in zip(features_list, features_labels)}
    ax_list = []
    for file_index, rec_file in enumerate(rec_file_list):
        feature_dict = {feature: {} for feature in features_list}
        distances_dict = {feature: {} for feature in features_list}
        with h5py.File(data_dir + rec_file + '.hdf5', 'r') as f:
            for item in viewvalues(f['Rinp_data']):
                if ((item.attrs['type'] in sec_types_list) or
                        ('axon' in sec_types_list and item.attrs['type'] in axon_types_list) or
                        ('dendrite' in sec_types_list and item.attrs['type'] in dend_types_list)):
                    if 'axon' in sec_types_list and item.attrs['type'] in axon_types_list:
                        sec_type = 'axon'
                    elif 'dendrite' in sec_types_list and item.attrs['type'] in dend_types_list:
                        sec_type = 'dendrite'
                    else:
                        sec_type = item.attrs['type']
                    for feature in features_list:
                        if sec_type not in distances_dict[feature]:
                            distances_dict[feature][sec_type] = []
                        if item.attrs['type'] in ['basal', 'axon', 'ais', 'hillock']:
                            distances_dict[feature][sec_type].append(item.attrs['soma_distance'] * -1.)
                        else:
                            distances_dict[feature][sec_type].append(item.attrs['soma_distance'])
                        if sec_type not in feature_dict[feature]:
                            feature_dict[feature][sec_type] = []
                        feature_dict[feature][sec_type].append(item.attrs[feature])
        num_colors = 10
        color_x = np.linspace(0., 1., num_colors)
        colors = [cm.Set1(x) for x in color_x]
        for index, feature in enumerate(features_list):
            if len(ax_list) <= index:
                fig, ax = plt.subplots(1)
                ax_list.append(ax)
            for i, sec_type in enumerate(sec_types_list):
                if file_labels is None:
                    label = sec_type
                elif len(sec_types_list) == 1:
                    label = file_labels[file_index]
                else:
                    label = file_labels[file_index] + ': ' + sec_type
                ax_list[index].scatter(distances_dict[feature][sec_type], feature_dict[feature][sec_type],
                            label=label, color = colors[file_index*len(sec_types_list) + i], alpha=0.5)
            ax_list[index].set_xlabel('Distance to soma')
            ax_list[index].set_ylabel(features_labels_dict[feature])
            ax_list[index].legend(loc='best', scatterpoints = 1, frameon=False, framealpha=0.5)
            clean_axes(ax_list[index])
    plt.show()
    plt.close()
    mpl.rcParams['font.size'] = orig_fontsize


def plot_Rinp_curr_inj(rec_file, rec_num):
    """

    :param rec_file: str
    :return:
    """
    with h5py.File(data_dir + rec_file + '.hdf5', 'r') as f:
        y_vals = f[str(rec_num)]['stim']['0'][:]
        x_vals = f[str(rec_num)]['time']
        fig, axes = plt.subplots(1)
        #axes.scatter(x_vals, y_vals)
        axes.plot(x_vals, y_vals)
        axes.set_xlabel('Time (ms')
        axes.set_ylabel('Injected current (nA)')
        clean_axes(axes)
        fig.tight_layout()
    plt.show()
    plt.close()


def plot_Rinp_avg_waves(rec_file_list, sec_types_list=None, file_labels=None):
    """
    Expects each file in list to be generated by parallel_rinp.
    avg_waves contains voltage waves averaged across similar sec_types (dendritic sections are divided into proximal
    and distal) recorded from simulated step current injections to probe input resistance and membrane time constant.
    :param rec_file_list: str or list of str
    :param sec_types_list:  str or list of str
    :param file_labels:  str or list of str
    """
    orig_fontsize = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 18.
    if isinstance(rec_file_list, str):
        rec_file_list = [rec_file_list]
    if isinstance(sec_types_list, str):
        sec_types_list = [sec_types_list]
    if isinstance(file_labels, str):
        file_labels = [file_labels]
    if sec_types_list is None:
        sec_types_list = ['soma', 'prox_apical', 'dist_apical']
    num_colors = 10
    color_x = np.linspace(0., 1., num_colors)
    colors = [cm.Set1(x) for x in color_x]
    ax_list = []
    for i, item in enumerate(sec_types_list):
        if len(ax_list) <= i:
            fig, ax = plt.subplots(1)
            ax_list.append(ax)
        for file_index, rec_file in enumerate(rec_file_list):
            with h5py.File(data_dir + rec_file + '.hdf5', 'r') as f:
                if item in f['avg_waves']:
                    if file_labels is not None:
                        label = file_labels[file_index]
                        ax_list[i].plot(f['avg_waves']['time'], f['avg_waves'][item], label=label,
                                        color=colors[file_index])
                    else:
                        ax_list[i].plot(f['avg_waves']['time'], f['avg_waves'][item], color=colors[file_index])
                ax_list[i].set_xlabel('Time (ms)')
                ax_list[i].set_ylabel('Voltage (mV)')
                if file_labels is not None:
                    ax_list[i].legend(loc='best', scatterpoints = 1, frameon=False, framealpha=0.5)
                ax_list[i].set_title(item)
                clean_axes(ax_list[i])
    plt.show()
    plt.close()
    mpl.rcParams['font.size'] = orig_fontsize


def plot_superimpose_conditions(rec_filename, legend=False):
    """
    File contains simulation results from iterating through some changes in parameters or stimulation conditions.
    This function produces one plot per recorded vector. Each plot superimposes the recordings from each of the
    simulation iterations.
    :param rec_filename: str
    :param legend: bool
    """
    f = h5py.File(data_dir+rec_filename+'.hdf5', 'r')
    rec_ids = []
    sim_ids = []
    for sim in viewvalues(f):
        if 'description' in sim.attrs and not sim.attrs['description'] in sim_ids:
            sim_ids.append(sim.attrs['description'])
        for rec in viewvalues(sim['rec']):
            if 'description' in rec.attrs:
                rec_id = rec.attrs['description']
            else:
                rec_id = rec.attrs['type']+str(rec.attrs['index'])
            if not rec_id in (id['id'] for id in rec_ids):
                rec_ids.append({'id': rec_id, 'ylabel': rec.attrs['ylabel']+' ('+rec.attrs['units']+')'})
    fig, axes = plt.subplots(1, max(2, len(rec_ids)))
    for i in range(len(rec_ids)):
        axes[i].set_xlabel('Time (ms)')
        axes[i].set_ylabel(rec_ids[i]['ylabel'])
        axes[i].set_title(rec_ids[i]['id'])
    for sim in viewvalues(f):
        if 'description' in sim.attrs:
            sim_id = sim.attrs['description']
        else:
            sim_id = ''
        tvec = sim['time']
        for rec in viewvalues(sim['rec']):
            if ('description' in rec.attrs):
                rec_id = rec.attrs['description']
            else:
                rec_id = rec.attrs['type']+str(rec.attrs['index'])
            i = [index for index, id in enumerate(rec_ids) if id['id'] == rec_id][0]
            axes[i].plot(tvec[:], rec[:], label=sim_id)
    if legend:
        for i in range(len(rec_ids)):
            axes[i].legend(loc='best', framealpha=0.5, frameon=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.1)
    plt.show()
    plt.close()
    f.close()


def plot_synaptic_parameter(rec_file_list, description_list=None):
    """
    Expects each file in list to be generated by optimize_EPSP_amp.
    Files contain one group for each type of dendritic section. Groups contain distances from soma and values for all
    measured synaptic parameters. Produces one column of plots per sec_type, one row of plots per parameter, and
    superimposes data from each rec_file.
    :param rec_file_list: list of str
    :param description_list: list of str
    """
    if not type(rec_file_list) == list:
        rec_file_list = [rec_file_list]
    if description_list is None:
        description_list = [" " for rec in rec_file_list]
    with h5py.File(data_dir+rec_file_list[0]+'.hdf5', 'r') as f:
        param_list = [dataset for dataset in next(iter(viewvalues(f))) if not dataset == 'distances']
        fig, axes = plt.subplots(max(2,len(param_list)), max(2, len(f)))
    colors = ['k', 'r', 'c', 'y', 'm', 'g', 'b']
    for index, rec_filename in enumerate(rec_file_list):
        with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
            for i, sec_type in enumerate(f):
                for j, dataset in enumerate(param_list):
                    axes[j][i].scatter(f[sec_type]['distances'][:], f[sec_type][dataset][:],
                                       label=description_list[index], color=colors[index])
                    axes[j][i].set_title(sec_type+' synapses')
                    axes[j][i].set_xlabel('Distance to soma (um)')
                    axes[j][i].set_ylabel(f.attrs['syn_type']+': '+dataset+'\n'+f.attrs[dataset])
    plt.legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5)
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.09, right=0.98, top=0.95, bottom=0.05)
    plt.show()
    plt.close()


def plot_synaptic_parameter_GC(rec_file_list, param_names=None, description_list=None):
    """
    Expects each file in list to be generated by optimize_EPSP_amp.
    Files contain one group for each type of dendritic section. Groups contain distances from soma and values for all
    measured synaptic parameters. Produces one column of plots per sec_type, one row of plots per parameter, and
    superimposes data from each rec_file.
    :param rec_file_list: list of str
    :param description_list: list of str
    """
    if not type(rec_file_list) == list:
        rec_file_list = [rec_file_list]
    default_input_locs = ['apical']
    # default_rec_locs = ['soma']
    with h5py.File(data_dir+rec_file_list[0]+'.hdf5', 'r') as f:
        if param_names is None:
            param_names = [param_name for param_name in next(iter(viewvalues(f))).attrs if param_name not in ['input_loc', 'equilibrate', 'duration']]
        temp_input_locs = []
        temp_rec_locs = []
        for sim in viewvalues(f):
            input_loc = sim.attrs['input_loc']
            if not input_loc in temp_input_locs:
                temp_input_locs.append(input_loc)
    # enforce the default order of input and recording locations for plotting, but allow for adding or subtracting
    # sec_types
    input_locs = [input_loc for input_loc in default_input_locs if input_loc in temp_input_locs]+\
                 [input_loc for input_loc in temp_input_locs if not input_loc in default_input_locs]
    distances_soma = {}
    #distances_dend = {}
    param_vals = {}
    for param_name in param_names:
        param_vals[param_name] = {}
    for index, rec_filename in enumerate(rec_file_list):
        for input_loc in input_locs:
            distances_soma[input_loc] = {}
            #distances_dend[input_loc] = {}
            for param_name in param_names:
                param_vals[param_name][input_loc] = {}
        with h5py.File(data_dir+rec_filename+'.hdf5', 'r') as f:
            for sim in viewvalues(f):
                input_loc = sim.attrs['input_loc']
                is_terminal = str(sim['rec']['2'].attrs['is_terminal'])
                if is_terminal not in distances_soma[input_loc].keys():
                    distances_soma[input_loc][is_terminal] = []
                    #distances_dend[input_loc][is_terminal] = []
                distances_soma[input_loc][is_terminal].append(sim['rec']['2'].attrs['soma_distance'])
                #distances_dend[input_loc][is_terminal].append(sim['rec']['2'].attrs['soma_distance'] -
                                                                        #sim['rec']['1'].attrs['soma_distance'])
                for param_name in param_names:
                    if is_terminal not in param_vals[param_name][input_loc]:
                        param_vals[param_name][input_loc][is_terminal] = []
                    param_vals[param_name][input_loc][is_terminal].append(sim.attrs[param_name])
            fig, axes = plt.subplots(max(2, len(input_locs)), max(2, len(param_names)))
            # fig, axes = plt.subplots(max(2, len(input_locs)*2), max(2, len(param_names)))
            colors = ['k', 'r', 'c', 'y', 'm', 'g', 'b']
            for i, input_loc in enumerate(input_locs):
                for j, param_name in enumerate(param_names):
                    string_keys = list(distances_soma[input_loc].keys())
                    terminal_keys = [int(key) for key in string_keys]
                    terminal_keys.sort()
                    terminal_labels = []
                    for key in terminal_keys:
                        if key == 0:
                            terminal_labels.append('not term.')
                        else:
                            terminal_labels.append('terminal')
                    for ind, is_terminal in enumerate([str(key) for key in terminal_keys]):
                        axes[i][j].scatter(distances_soma[input_loc][is_terminal], param_vals[param_name][input_loc][is_terminal],
                                           color=colors[ind], label=terminal_labels[ind])
                        #axes[i + len(input_locs)][j].scatter(distances_dend[input_loc][is_terminal],
                                            # param_vals[param_name][input_loc][is_terminal], color=colors[ind], label=terminal_labels[ind])
                    axes[i][j].legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5)
                    #axes[i + len(input_locs)][j].legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5)
                    axes[i][j].set_xlabel('Distance from Soma (um)')
                    #axes[i + len(input_locs)][j].set_xlabel('Distance from Dendritic Origin (um)')
                    axes[0][j].set_title('Parameter: ' + param_name, fontsize=mpl.rcParams['font.size'])
                axes[i][0].set_ylabel('Synapse Location: '+input_loc+'\n'+param_name)
                #axes[i + len(input_locs)][0].set_ylabel('Synapse Location: ' + input_loc + '\n'+param_name)

    fig.subplots_adjust(hspace=0.25, wspace=0.3, left=0.07, right=0.98, top=0.94, bottom=0.1)
    clean_axes(axes.flatten())
    plt.show()
    plt.close()


def plot_sum_mech_param_distribution(cell, mech_param_list, scale_factor=10000., param_label=None,
                                 ylabel='Conductance density', yunits='pS/um2', svg_title=None):
    """
    Takes a cell as input rather than a file. No simulation is required, this method just takes a fully specified cell
    and plots the relationship between distance and the specified mechanism parameter for all dendritic segments. Used
    while debugging specification of mechanism parameters.
    :param cell: :class:'HocCell'
    :param mech_param_list: list of tuple of str
    :param scale_factor: float
    :param ylabel: str
    :param yunits: str
    :param svg_title: str
    """
    colors = ['k', 'r', 'c', 'y', 'm', 'g', 'b']
    dend_types = ['basal', 'trunk', 'apical', 'tuft']

    if svg_title is not None:
        remember_font_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = 20
    fig, axes = plt.subplots(1)
    for i, sec_type in enumerate(dend_types):
        distances = []
        param_vals = []
        for branch in cell.get_nodes_of_subtype(sec_type):
            for seg in branch.sec:
                this_param_val = 0.
                this_distance = None
                for mech_name, param_name in mech_param_list:
                    if hasattr(seg, mech_name):
                        if this_distance is None:
                            this_distance = cell.get_distance_to_node(cell.tree.root, branch, seg.x)
                            if sec_type == 'basal':
                                this_distance *= -1
                        this_param_val += getattr(getattr(seg, mech_name), param_name) * scale_factor
                if this_distance is not None:
                    distances.append(this_distance)
                    param_vals.append(this_param_val)
        if param_vals:
            axes.scatter(distances, param_vals, color=colors[i], label=sec_type)
            if maxval is None:
                maxval = max(param_vals)
            else:
                maxval = max(maxval, max(param_vals))
            if minval is None:
                minval = min(param_vals)
            else:
                minval = min(minval, min(param_vals))
    axes.set_xlabel('Distance to soma (um)')
    axes.set_xlim(-200., 525.)
    axes.set_xticks([-150., 0., 150., 300., 450.])
    axes.set_ylabel(ylabel+' ('+yunits+')')
    buffer = 0.1 * (maxval - minval)
    axes.set_ylim(minval-buffer, maxval+buffer)
    if param_label is not None:
        plt.title(param_label, fontsize=mpl.rcParams['font.size'])
    plt.legend(loc='best', scatterpoints=1, frameon=False, framealpha=0.5, fontsize=mpl.rcParams['font.size'])
    clean_axes(axes)
    axes.tick_params(direction='out')
    if not svg_title is None:
        if param_label is not None:
            svg_title = svg_title+' - '+param_label+'.svg'
        else:
            mech_name, param_name = mech_param_list[0]
            svg_title = svg_title+' - '+mech_name+'_'+param_name+' distribution.svg'
        fig.set_size_inches(5.27, 4.37)
        fig.savefig(data_dir + svg_title, format='svg', transparent=True)
    plt.show()
    plt.close()
    if svg_title is not None:
        mpl.rcParams['font.size'] = remember_font_size


def plot_absolute_energy(storage):
    fig, axes = plt.subplots(1)
    colors = list(cm.rainbow(np.linspace(0, 1, len(storage.history))))
    this_attr = 'objectives'
    for j, population in enumerate(storage.history):
        axes.scatter([indiv.rank for indiv in population],
                    [np.sum(getattr(indiv, this_attr)) for indiv in population],
                    c=colors[j], alpha=0.05)
        axes.scatter([indiv.rank for indiv in storage.survivors[j]],
                    [np.sum(getattr(indiv, this_attr)) for indiv in storage.survivors[j]], c=colors[j], alpha=0.5)
    axes.set_xlabel('Rank')
    axes.set_ylabel('Summed Objectives')


def plot_best_norm_features_boxplot(storage, target_val, target_range):
    """

    :return:
    """
    #Ensure that f_I_slope is in target_val with a value of 53.
    fig, axes = plt.subplots(1)
    labels = list(target_val.keys())
    # y_values = range(len(y_labels))
    final_survivors = storage.survivors[-1]
    norm_feature_vals = {}
    colors = list(cm.rainbow(np.linspace(0, 1, len(labels))))
    for survivor in final_survivors:
        for i, feature in enumerate(storage.feature_names):
            if feature in target_val:
                if feature not in norm_feature_vals:
                    norm_feature_vals[feature] = []
                if (feature == 'slow_depo' and getattr(survivor, 'features')[i] < target_val[feature]) or \
                        (feature == 'AHP' and getattr(survivor, 'features')[i] < target_val[feature]):
                    normalized_val = 0.
                else:
                    normalized_val = (getattr(survivor, 'features')[i] - target_val[feature]) / target_range[feature]
                norm_feature_vals[feature].append(normalized_val)
    x_values_list = [norm_feature_vals[feature] for feature in labels]
    """
    for i, y_value in enumerate(y_values):
        axes.scatter(x_values_list[i], y_value * np.ones(len(x_values_list[i])), c=colors[i])
    """
    bplot = axes.boxplot(x_values_list, vert=False, labels=labels, patch_artist=True)
    colors = list(cm.rainbow(np.linspace(0, 1, len(labels))))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    axes.set_xlabel('Normalized Features')
    #axes.set_yticks(y_values)
    #axes.set_yticklabels(y_labels)
    axes.set_title('Best Parameter Sets')
    clean_axes(axes)
    fig.tight_layout()
    plt.show()
    plt.close()


def plot_best_norm_features_scatter(storage, target_val, target_range):
    """

    :return:
    """
    #Ensure that f_I_slope is in target_val with a value of 53.
    """
    {'ADP': 2.0, 'AHP': 0.8, 'ais_delay': 0.02, 'dend R_inp': 75.0, 'dend_amp': 0.06, 'rebound_firing': 0.2,
     'slow_depo': 2.0, 'soma R_inp': 59.0, 'soma_peak': 8.0, 'spont_firing': 0.2, 'v_th': -9.600000000000001,
     'vm_stability': 2.0, 'f_I_slope': 10.6}
    """
    orig_fontsize = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 16.
    fig, axes = plt.subplots(1)
    y_labels = list(target_val.keys())
    y_values = list(range(len(y_labels)))
    final_survivors = storage.survivors[-1]
    norm_feature_vals = {}
    colors = list(cm.rainbow(np.linspace(0, 1, len(y_labels))))
    for survivor in final_survivors:
        for i, feature in enumerate(storage.feature_names):
            if feature in target_val:
                if feature not in norm_feature_vals:
                    norm_feature_vals[feature] = []
                if (feature == 'slow_depo' and getattr(survivor, 'features')[i] < target_val[feature]) or \
                        (feature == 'AHP' and getattr(survivor, 'features')[i] < target_val[feature]):
                    normalized_val = 0.
                else:
                    normalized_val = (getattr(survivor, 'features')[i] - target_val[feature]) / target_range[feature]
                norm_feature_vals[feature].append(normalized_val)
    x_values_list = [norm_feature_vals[feature] for feature in y_labels]
    for i, y_value in enumerate(y_values):
        axes.scatter(x_values_list[i], y_value * np.ones(len(x_values_list[i])), c=colors[i], alpha=0.4)
    axes.set_xlabel('Normalized Features')
    axes.set_xlim(-2.5, 2.5)
    axes.set_yticks(y_values)
    axes.set_yticklabels(y_labels)
    axes.set_title('Best Parameter Sets', fontsize=mpl.rcParams['font.size'] + 2)
    clean_axes(axes)
    fig.tight_layout()
    plt.show()
    plt.close()
    mpl.rcParams['font.size'] = orig_fontsize


def plot_exported_DG_GC_spiking_features(file_path):
    """
    
    :param file_path: str (path)
    """
    orig_fontsize = mpl.rcParams['font.size']
    if not os.path.isfile(file_path):
        raise IOError('plot_exported_DG_GC_spiking_features: invalid file path: %s' % file_path)
    with h5py.File(file_path, 'r') as f:
        group_name = 'f_I'
        if group_name not in f:
            raise AttributeError('plot_exported_DG_GC_spiking_features: provided file path: %s does not contain a '
                                 'required group: %s' % (file_path, group_name))
        group = f[group_name]
        fig1, axes1 = plt.subplots()
        i_relative_amp = group['i_relative_amp'][:]
        rate = group['rate'][:]
        exp_rate = group['exp_rate'][:]
        axes1.scatter(i_relative_amp, rate, label='Model', c='r', linewidth=0, alpha=0.5)
        axes1.plot(i_relative_amp, rate, c='r', alpha=0.5)
        axes1.scatter(i_relative_amp, exp_rate, label='Experiment', c='grey', linewidth=0, alpha=0.5)
        axes1.plot(i_relative_amp, exp_rate, c='grey', alpha=0.5)
        axes1.legend(loc='best', frameon=False, framealpha=0.5)
        axes1.set_xlabel('Amplitude of current injection\nrelative to rheobase (nA)')
        axes1.set_ylabel('Firing rate (Hz)')
        axes1.set_ylim(0., axes1.get_ylim()[1])
        axes1.set_xlim(0., axes1.get_xlim()[1])
        axes1.set_title('f-I', fontsize=mpl.rcParams['font.size'])
        clean_axes(axes1)
        fig1.tight_layout()
        fig1.show()

        group_name = 'spike_adaptation'
        if group_name not in f:
            raise AttributeError('plot_exported_DG_GC_spiking_features: provided file path: %s does not contain a '
                                 'required group: %s' % (file_path, group_name))
        group = f[group_name]
        fig2, axes2 = plt.subplots()
        model_ISI_array = group['model_ISI_array'][:]
        exp_ISI_array = group['exp_ISI_array'][:]
        ISI_num = list(range(1, len(exp_ISI_array) + 1))
        axes2.scatter(ISI_num, model_ISI_array, label='Model', c='r', linewidth=0, alpha=0.5)
        axes2.plot(ISI_num, model_ISI_array, c='r', alpha=0.5)
        axes2.scatter(ISI_num, exp_ISI_array, label='Experiment', c='k', linewidth=0, alpha=0.5)
        axes2.plot(ISI_num, exp_ISI_array, c='k', alpha=0.5)
        axes2.legend(loc='best', frameon=False, framealpha=0.5)
        axes2.set_xlabel('ISI number')
        axes2.set_ylabel('Inter-spike interval (ms)')
        axes2.set_ylim(0., axes2.get_ylim()[1])
        axes2.set_title('Spike rate adaptation', fontsize=mpl.rcParams['font.size'])
        clean_axes(axes2)
        fig2.tight_layout()
        fig2.show()
    mpl.rcParams['font.size'] = orig_fontsize


def plot_exported_DG_MC_spiking_features(file_path):
    """

    :param file_path: str (path)
    """
    orig_fontsize = mpl.rcParams['font.size']
    if not os.path.isfile(file_path):
        raise IOError('plot_exported_DG_MC_spiking_features: invalid file path: %s' % file_path)
    with h5py.File(file_path, 'r') as f:
        group_name = 'f_I'
        if group_name not in f:
            raise AttributeError('plot_exported_DG_MC_spiking_features: provided file path: %s does not contain a '
                                 'required group: %s' % (file_path, group_name))
        group = f[group_name]
        fig1, axes1 = plt.subplots()
        i_relative_amp = group['i_relative_amp'][:]
        rate = group['rate'][:]
        exp_rate = group['exp_rate'][:]
        axes1.scatter(i_relative_amp, rate, label='Model', c='r', linewidth=0, alpha=0.5)
        axes1.plot(i_relative_amp, rate, c='r', alpha=0.5)
        axes1.scatter(i_relative_amp, exp_rate, label='Experiment', c='grey', linewidth=0, alpha=0.5)
        axes1.plot(i_relative_amp, exp_rate, c='grey', alpha=0.5)
        axes1.legend(loc='best', frameon=False, framealpha=0.5)
        axes1.set_xlabel('Amplitude of current injection\nrelative to rheobase (nA)')
        axes1.set_ylabel('Firing rate (Hz)')
        axes1.set_ylim(0., axes1.get_ylim()[1])
        axes1.set_xlim(0., axes1.get_xlim()[1])
        axes1.set_title('f-I', fontsize=mpl.rcParams['font.size'])
        clean_axes(axes1)
        fig1.tight_layout()
        fig1.show()

        fig2, axes2 = plt.subplots()
        model_adi_array = group['adi'][:]
        exp_adi_array = group['exp_adi'][:]
        axes2.scatter(i_relative_amp, model_adi_array, label='Model', c='r', linewidth=0, alpha=0.5)
        axes2.plot(i_relative_amp, model_adi_array, c='r', alpha=0.5)
        axes2.scatter(i_relative_amp, exp_adi_array, label='Experiment', c='k', linewidth=0, alpha=0.5)
        axes2.plot(i_relative_amp, exp_adi_array, c='k', alpha=0.5)
        axes2.legend(loc='best', frameon=False, framealpha=0.5)
        axes1.set_xlabel('Amplitude of current injection\nrelative to rheobase (nA)')
        axes2.set_ylabel('Spike adaptation (%)\n(Last ISI/First ISI)')
        axes2.set_ylim(0., axes2.get_ylim()[1])
        axes2.set_title('Spike rate adaptation', fontsize=mpl.rcParams['font.size'])
        clean_axes(axes2)
        fig2.tight_layout()
        fig2.show()
    mpl.rcParams['font.size'] = orig_fontsize


def plot_exported_DG_GC_synaptic_integration_features(file_path):
    """

    :param file_path: str (path)
    """
    orig_fontsize = mpl.rcParams['font.size']
    if not os.path.isfile(file_path):
        raise IOError('plot_exported_DG_GC_synaptic_integration_features: invalid file path: %s' % file_path)
    from matplotlib import cm
    with h5py.File(file_path, 'r') as f:
        group_name = 'mean_unitary_EPSP_traces'
        if group_name not in f:
            raise AttributeError('plot_exported_DG_GC_synaptic_integration_features: provided file path: %s does not '
                                 'contain a required group: %s' % (file_path, group_name))
        t = f[group_name]['time'][:]
        data_group = f[group_name]['data']

        for syn_group in data_group:
            syn_conditions = list(data_group[syn_group].keys())
            ordered_syn_conditions = ['control'] + [syn_condition for syn_condition in syn_conditions
                                                    if syn_condition not in ['control']]
            rec_names = list(data_group[syn_group]['control'].keys())
            if 'soma' in rec_names:
                ordered_rec_names = ['soma'] + [rec_name for rec_name in rec_names if rec_name not in ['soma']]
            else:
                ordered_rec_names = rec_names

            fig, axes = plt.subplots(1, len(rec_names), sharey=True)
            colors = list(cm.Paired(np.linspace(0, 1, len(syn_conditions))))
            if len(rec_names) == 1:
                axes = [axes]
            for i, rec_name in enumerate(ordered_rec_names):
                for j, syn_condition in enumerate(ordered_syn_conditions):
                    axes[i].plot(t, data_group[syn_group][syn_condition][rec_name][:], label=syn_condition,
                                 color=colors[j])
                axes[i].set_title(rec_name + ' Vm', fontsize=mpl.rcParams['font.size'])
                axes[i].set_xlabel('Time (ms)')
            axes[0].set_ylabel('Unitary EPSP amplitude (mV)')
            axes[0].legend(loc='best', frameon=False, framealpha=0.5)
            clean_axes(axes)
            fig.suptitle('Branch: %s' % syn_group, fontsize=mpl.rcParams['font.size'])
            fig.tight_layout()
            fig.subplots_adjust(top=0.875)
            fig.show()

        group_name = 'compound_EPSP_summary'
        if group_name not in f:
            raise AttributeError('plot_exported_DG_GC_synaptic_integration_features: provided file path: %s does not '
                                 'contain a required group: %s' % (file_path, group_name))
        group = f[group_name]
        t = group['time'][:]
        syn_conditions = list(next(iter(viewvalues(group['traces']))).keys())
        ordered_syn_conditions = ['expected_control', 'control']
        for syn_condition in [syn_condition for syn_condition in syn_conditions
                              if syn_condition != 'control' and 'expected' not in syn_condition]:
            ordered_syn_conditions.extend(['expected_' + syn_condition, syn_condition])

        for branch_name in group['traces']:
            for rec_name in rec_names:
                fig, axes = plt.subplots(1, len(syn_conditions), sharey=True)
                fig.suptitle('Branch: %s\nRecording loc: %s' % (branch_name, rec_name),
                             fontsize=mpl.rcParams['font.size'])
                for i, syn_condition in enumerate(ordered_syn_conditions):
                    for num_syns in group['traces'][branch_name][syn_condition]:
                        axes[i].plot(t, group['traces'][branch_name][syn_condition][num_syns][rec_name][:], c='k')
                    axes[i].set_xlabel('Time (ms)')
                    axes[i].set_title(syn_condition, fontsize=mpl.rcParams['font.size'])
                axes[0].set_ylabel('Compound EPSP amplitude (mV)')
                clean_axes(axes)
                fig.tight_layout()
                fig.subplots_adjust(top=0.85)
                fig.show()

        data_group = group['soma_compound_EPSP_amp']
        branch_names = list(data_group.keys())
        fig, axes = plt.subplots(1, len(branch_names), sharey=True, sharex=True)
        if len(branch_names) == 1:
            axes = [axes]
        syn_conditions = list(next(iter(viewvalues(data_group))).keys())
        ordered_syn_conditions = ['control'] + [syn_condition for syn_condition in syn_conditions
                                                if syn_condition != 'control' and 'expected' not in syn_condition]
        colors = list(cm.Paired(np.linspace(0, 1, len(syn_conditions))))
        rec_name = 'soma'
        for i, branch_name in enumerate(branch_names):
            expected_max = 0.
            for j, syn_condition in enumerate(ordered_syn_conditions):
                expected_key = 'expected_' + syn_condition
                expected_max = max(expected_max, np.max(data_group[branch_name][expected_key][:]))
                axes[i].plot(data_group[branch_name][expected_key][:], data_group[branch_name][syn_condition][:],
                             c=colors[j], label=syn_condition)
            axes[i].set_title('Branch: %s\nRecording loc: %s' % (branch_name, rec_name),
                              fontsize=mpl.rcParams['font.size'])
            axes[i].set_xlabel('Expected EPSP amp (mV)')
            diagonal = np.linspace(0., expected_max, 10)
            axes[i].plot(diagonal, diagonal, c='lightgrey', linestyle='--')
        axes[0].set_ylabel('Actual EPSP amp (mV)')
        axes[0].legend(loc='best', frameon=False, framealpha=0.5)
        clean_axes(axes)
        fig.tight_layout()
        fig.show()
    mpl.rcParams['font.size'] = orig_fontsize


def plot_sim_from_file(file_path, group_name='sim_output'):
    """

    :param file_path: str (path)
    :param group_name: str
    """
    orig_fontsize = mpl.rcParams['font.size']
    if not os.path.isfile(file_path):
        raise IOError('plot_sim_from_file: invalid file path: %s' % file_path)
    with h5py.File(file_path, 'r') as f:
        if group_name not in f:
            raise AttributeError('plot_sim_from_file: provided file path: %s does not contain required top-level group '
                                 'with name: %s' % (file_path, group_name))
        for trial in viewvalues(f[group_name]):
            fig, axes = plt.subplots()
            for name, rec in viewitems(trial['recs']):
                description = get_h5py_attr(rec.attrs, 'description')
                sec_type = get_h5py_attr(rec.attrs, 'type')
                node_name = '%s%i' % (sec_type, rec.attrs['index'])
                label = '%s: %s(%.2f) %s' % (name, node_name, rec.attrs['loc'], description)
                axes.plot(trial['time'], rec, label=label)
                axes.set_xlabel('Time (ms)')
                ylabel = get_h5py_attr(rec.attrs, 'ylabel')
                units = get_h5py_attr(rec.attrs, 'units')
                axes.set_ylabel('%s (%s)' % (ylabel, units))
            axes.legend(loc='best', frameon=False, framealpha=0.5)
            title = None
            if 'title' in trial.attrs:
                title = get_h5py_attr(trial.attrs, 'title')
            if 'description' in trial.attrs:
                description = get_h5py_attr(trial.attrs, 'description')
                if title is not None:
                    title = title + '; ' + description
                else:
                    title = description
            if title is not None:
                axes.set_title(title, fontsize=mpl.rcParams['font.size'])
            clean_axes(axes)
            fig.tight_layout()
            fig.show()
    mpl.rcParams['font.size'] = orig_fontsize


def plot_na_gradient_params(x_dict):
    """

    :param x_dict: dict
    :return:
    """
    orig_fontsize = mpl.rcParams['font.size']
    mpl.rcParams['font.size'] = 20.
    fig, axes = plt.subplots(1)
    x_labels = ['axon', 'AIS', 'soma', 'dend']
    x_values = list(range(len(x_labels)))
    colors = ['b', 'c', 'g', 'r']
    y_values = [x_dict['axon.gbar_nax'], x_dict['ais.gbar_nax'], x_dict['soma.gbar_nas'], x_dict['dend.gbar_nas']]
    for i in x_values:
        axes.scatter(x_values[i], y_values[i], c=colors[i])
    # axes.set_ylim(-2.5, 2.5)
    axes.set_xticks(x_values)
    axes.set_xticklabels(x_labels)
    axes.set_ylabel('gmax_na')
    clean_axes(axes)
    fig.tight_layout()
    plt.show()
    plt.close()
    mpl.rcParams['font.size'] = orig_fontsize


def plot_NMDAR_g_V(Kd=9.98, gamma=0.101, mg=1., vshift=0., label='original', axes=None, show=True):
    """

    :param Kd: float
    :param gamma: float
    :param mg:  float
    :param vshift: float
    :param label: str
    :param axes: :class:'Axes'
    :param show: bool
    """
    v = np.arange(-100., 50., 1.)
    B = 1. / (1. + np.exp(gamma * (-(v-vshift))) * (mg / Kd))
    # B /= np.max(B)
    if axes is None:
        fig, axes = plt.subplots(1)
    axes.plot(v, B, label=label)
    axes.set_ylabel('Normalized conductance')
    axes.set_xlabel('Voltage (mV)')
    axes.set_title('NMDAR g-V')
    clean_axes(axes)
    if show:
        plt.legend(loc='best', frameon=False, framealpha=0.5)
        plt.show()
    else:
        return axes
