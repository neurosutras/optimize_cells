__author__ = 'Aaron D. Milstein, Grace Ng and Prannath Moolchand'
from nested.utils import *
from dentate.cells import *
from dentate.synapses import *


def time2index(tvec, start, stop):
    """
    When using adaptive time step (cvode), indices corresponding to specific time points cannot be calculated from a
    fixed dt. This method returns the indices closest to the duration bounded by the specified time points.
    :param tvec: :class:'numpy.array'
    :param start: float
    :param stop: float
    :return: tuple of int
    """
    left = np.where(tvec >= start)[0]
    if np.any(left):  # at least one value was found
        left = left[0]
    else:
        right = len(tvec) - 1  # just take the last two indices
        left = right - 1
        return left, right
    if tvec[left] >= stop:
        right = left
        left -= 1
        return left, right
    right = np.where(tvec <= stop)[0][-1]
    if right == left:
        left -= 1
    return left, right


def interp(t, x, duration, dt=0.025):
    """
    Interpolates the arrays t and x from t=0 to t=duration with resolution dt.
    :param t: array
    :param x: array
    :param duration: float
    :param dt: float
    :return: array, array
    """
    interp_t = np.arange(0., duration, dt)
    interp_x = np.interp(interp_t, t, x)
    return interp_t, interp_x


def get_R_inp(t, vm, start, stop, amp, dt=0.025):
    """
    Calculate peak and steady-state input resistance from a step current injection.
    :param t: array
    :param vm: array
    :param start: float
    :param stop: float
    :param amp: float
    :param dt: float
    :return: tuple of float
    """
    interp_t, interp_vm = interp(t, vm, stop, dt)
    left = int((start-3.) / dt)
    right = left + int(2. / dt)
    baseline = np.mean(interp_vm[left:right])
    interp_vm = np.abs(interp_vm - baseline)
    start_index = int(start / dt)
    peak = np.max(interp_vm[start_index:])
    left = int((stop-3.) / dt)
    right = left + int(2. / dt)
    plateau = np.mean(interp_vm[left:right])
    return baseline, peak/abs(amp), plateau/abs(amp)


def get_event_amp(t, rec, start, stop, dt=0.025):
    """
    Calculate peak amplitude of an event within the specified recording window.
    :param t: array
    :param rec: array
    :param start: float
    :param stop: float
    :param dt: float
    :return: float
    """
    interp_t, interp_rec = interp(t, rec, stop, dt)
    start_index = int(start / dt)
    left = start_index - int(3. / dt)
    right = left + int(2. / dt)
    baseline = np.mean(interp_rec[left:right])
    interp_rec -= baseline
    abs_interp_rec = np.abs(interp_rec)

    peak_index = np.argmax(abs_interp_rec[start_index:])
    peak_val = interp_rec[start_index + peak_index]

    return peak_val


def model_exp_rise_decay(t, tau_rise, tau_decay):
    shape = np.exp(-t/tau_decay)-np.exp(-t/tau_rise)
    return shape/np.max(shape)


def model_exp_rise(t, tau):
    return 1-np.exp(-t/tau)


def model_exp_decay(t, tau):
    return np.exp(-t/tau)


def model_scaled_exp(t, A, tau, A0=0):
    return A*np.exp(t/tau)+A0


def get_expected_spine_index_map(sim_file):
    """
    There is a bug with HDF5 when reading from a file too often within a session. Instead of constantly reading from the
    HDF5 file directly and searching for content by spine_index or path_index, the number of calls to the sim_file can
    be reduced by creating a mapping from spine_index or path_index to HDF5 group key. It is possible for a spine to
    have more than one entry in an expected_file, with branch recordings in different locations and therefore different
    expected EPSP waveforms, so it is necessary to also distinguish those entries by path_index.

    :param sim_file: :class:'h5py.File'
    :return: dict
    """
    index_map = {}
    for key, sim in viewitems(sim_file):
        path_index = sim.attrs['path_index']
        spine_index = sim.attrs['spine_index']
        if path_index not in index_map:
            index_map[path_index] = {}
        index_map[path_index][spine_index] = key
    return index_map


def get_spine_group_info(sim_filename, verbose=1):
    """
    Given a processed output file generated by export_nmdar_cooperativity, this method returns a dict that has
    separated each group of stimulated spines by dendritic sec_type, and sorted by distance from soma. For ease of
    inspection so that the appropriate path_index can be chosen for plotting expected and actual summation traces.
    :param sim_filename: str
    :return: dict
    """
    spine_group_info = {}
    with h5py.File(data_dir+sim_filename+'.hdf5', 'r') as f:
        for path_index in f:
            sim = f[path_index]
            path_type = sim.attrs['path_type']
            path_category = sim.attrs['path_category']
            if path_type not in spine_group_info:
                spine_group_info[path_type] = {}
            if path_category not in spine_group_info[path_type]:
                spine_group_info[path_type][path_category] = {'path_indexes': [], 'distances': []}
            if path_index not in spine_group_info[path_type][path_category]['path_indexes']:
                spine_group_info[path_type][path_category]['path_indexes'].append(path_index)
                if path_type == 'apical':
                    # for obliques, sort by the distance of the branch origin from the soma
                    distance = sim.attrs['origin_distance']
                else:
                    distance = sim.attrs['soma_distance']
                spine_group_info[path_type][path_category]['distances'].append(distance)
    for path_type in spine_group_info:
        for path_category in spine_group_info[path_type]:
            indexes = list(range(len(spine_group_info[path_type][path_category]['distances'])))
            indexes.sort(key=spine_group_info[path_type][path_category]['distances'].__getitem__)
            spine_group_info[path_type][path_category]['distances'] = \
                list(map(spine_group_info[path_type][path_category]['distances'].__getitem__, indexes))
            spine_group_info[path_type][path_category]['path_indexes'] = \
                list(map(spine_group_info[path_type][path_category]['path_indexes'].__getitem__, indexes))
        if verbose:
            for path_category in spine_group_info[path_type]:
                print(path_type, '-', path_category)
                for i, distance in enumerate(spine_group_info[path_type][path_category]['distances']):
                    print(spine_group_info[path_type][path_category]['path_indexes'][i], distance)
    return spine_group_info


def get_expected_EPSP(sim_file, group_index, equilibrate, duration, dt=0.02):
    """
    Given an output file generated by parallel_clustered_branch_cooperativity or build_expected_EPSP_reference, this
    method returns a dict of numpy arrays, each containing the depolarization-rectified expected EPSP for each
    recording site resulting from stimulating a single spine.
    :param sim_file: :class:'h5py.File'
    :param group_index: int
    :param equilibrate: float
    :param duration: float
    :param dt: float
    :return: dict of :class:'numpy.array'
    """
    sim = sim_file[str(group_index)]
    t = sim['time'][:]
    interp_t = np.arange(0., duration, dt)
    left, right = time2index(interp_t, equilibrate-3., equilibrate-1.)
    start, stop = time2index(interp_t, equilibrate-2., duration)
    trace_dict = {}
    for rec in viewvalues(sim['rec']):
        location = rec.attrs['description']
        vm = rec[:]
        interp_vm = np.interp(interp_t, t, vm)
        baseline = np.average(interp_vm[left:right])
        interp_vm -= baseline
        interp_vm = interp_vm[start:stop]
        """
        rectified = np.zeros(len(interp_vm))
        rectified[np.where(interp_vm>0.)[0]] += interp_vm[np.where(interp_vm>0.)[0]]
        trace_dict[location] = rectified
        """
        peak = np.max(interp_vm)
        peak_index = np.where(interp_vm == peak)[0][0]
        zero_index = np.where(interp_vm[peak_index:] <= 0.)[0]
        if np.any(zero_index):
            interp_vm[peak_index+zero_index[0]:] = 0.
        trace_dict[location] = interp_vm
    interp_t = interp_t[start:stop]
    interp_t -= interp_t[0] + 2.
    trace_dict['time'] = interp_t
    return trace_dict


def get_expected_vs_actual(expected_sim_file, actual_sim_file, expected_index_map, sorted_actual_sim_keys,
                           interval=0.3, dt=0.02):
    """
    Given an output file generated by parallel_clustered_branch_cooperativity, and an output file generated by
    parallel_branch_cooperativity, this method returns a dict of lists, each containing an input-output function
    relating expected to actual peak depolarization for each recording site from stimulating a group of spines on a
    single branch or path. The variable expected_index_map contains a dictionary that converts an integer spine_index to
    a string group_index to locate the expected EPSP for a given spine in the expected_sim_file. The variable
    sorted_actual_sim_keys contains the indexes of the simulations in the actual_sim_file corresponding to the branch or
    path, ordered by number of stimulated spines. These variables must be pre-computed.
    :param expected_sim_file: :class:'h5py.File'
    :param actual_sim_file: :class:'h5py.File'
    :param expected_index_map: dict
    :param sorted_actual_sim_keys: list of str
    :param interval: float
    :return: dict of list
    """
    equilibrate = actual_sim_file[sorted_actual_sim_keys[0]].attrs['equilibrate']
    duration = actual_sim_file[sorted_actual_sim_keys[0]].attrs['duration']
    actual = {}
    for sim in [actual_sim_file[key] for key in sorted_actual_sim_keys]:
        t = sim['time'][:]
        interp_t = np.arange(0., duration, dt)
        left, right = time2index(interp_t, equilibrate-3., equilibrate-1.)
        start, stop = time2index(interp_t, equilibrate-2., duration)
        for rec in viewvalues(sim['rec']):
            location = rec.attrs['description']
            if not location in actual:
                actual[location] = []
            vm = rec[:]
            interp_vm = np.interp(interp_t, t, vm)
            baseline = np.average(interp_vm[left:right])
            interp_vm -= baseline
            interp_vm = interp_vm[start:stop]
            actual[location].append(np.max(interp_vm))
    spine_list = sim.attrs['syn_indexes']
    interp_t = interp_t[start:stop]
    interp_t -= interp_t[0] + 2.
    expected = {}
    summed_traces = {}
    equilibrate = next(iter(viewvalues(expected_sim_file))).attrs['equilibrate']
    duration = next(iter(viewvalues(expected_sim_file))).attrs['duration']
    for i, spine_index in enumerate(spine_list):
        group_index = expected_index_map[spine_index]
        trace_dict = get_expected_EPSP(expected_sim_file, group_index, equilibrate, duration, dt)
        t = trace_dict['time']
        left, right = time2index(interp_t, -2.+i*interval, interp_t[-1])
        right = min(right, left+len(t))
        for location in [location for location in trace_dict if not location == 'time']:
            trace = trace_dict[location]
            if not location in expected:
                expected[location] = []
                summed_traces[location] = np.zeros(len(interp_t))
            summed_traces[location][left:right] += trace[:right-left]
            expected[location].append(np.max(summed_traces[location]))
    return expected, actual


def export_nmdar_cooperativity(expected_filename, actual_filename, description="", output_filename=None):
    """
    Expects expected and actual files to be generated by parallel_clustered_ or
    parallel_distributed_branch_cooperativity. Files contain simultaneous voltage recordings from 3 locations (soma,
    trunk, dendrite origin) during synchronous stimulation of branches, and an NMDAR conductance recording from a single
    spine in each group. Spines are distributed across 4 dendritic sec_types (basal, trunk, apical, tuft).
    Generates a processed output file containing expected vs. actual data and metadata for each group of spines.
    Can be used to generate plots of supralinearity, NMDAR conductance, or average across conditions, etc.
    :param expected_filename: str
    :param actual_filename: str
    :param description: str
    :param output_filename: str
    """
    sim_key_dict = {}
    with h5py.File(data_dir+actual_filename+'.hdf5', 'r') as actual_file:
        for key, sim in viewitems(actual_file):
            path_index = sim.attrs['path_index']
            if path_index not in sim_key_dict:
                sim_key_dict[path_index] = []
            sim_key_dict[path_index].append(key)
        with h5py.File(data_dir+expected_filename+'.hdf5', 'r') as expected_file:
            expected_index_map = get_expected_spine_index_map(expected_file)
            with h5py.File(data_dir+output_filename+'.hdf5', 'w') as output_file:
                output_file.attrs['description'] = description
                for path_index in sim_key_dict:
                    path_group = output_file.create_group(str(path_index))
                    sim_keys = sim_key_dict[path_index]
                    sim_keys.sort(key=lambda x: len(actual_file[x].attrs['syn_indexes']))
                    sim = actual_file[sim_keys[0]]
                    path_type = sim.attrs['path_type']
                    path_category = sim.attrs['path_category']
                    soma_distance = sim['rec']['4'].attrs['soma_distance']
                    branch_distance = sim['rec']['4'].attrs['branch_distance']
                    origin_distance = soma_distance - branch_distance
                    path_group.attrs['path_type'] = path_type
                    path_group.attrs['path_category'] = path_category
                    path_group.attrs['soma_distance'] = soma_distance
                    path_group.attrs['branch_distance'] = branch_distance
                    path_group.attrs['origin_distance'] = origin_distance
                    expected_dict, actual_dict = get_expected_vs_actual(expected_file, actual_file,
                                                                        expected_index_map[path_index], sim_keys)
                    for rec in viewvalues(sim['rec']):
                        location = rec.attrs['description']
                        rec_group = path_group.create_group(location)
                        rec_group.create_dataset('expected', compression='gzip', compression_opts=9,
                                                           data=expected_dict[location])
                        rec_group.create_dataset('actual', compression='gzip', compression_opts=9,
                                                           data=actual_dict[location])


def sliding_window(unsorted_x, y=None, bin_size=60., window_size=3, start=-60., end=7560.):
    """
    An ad hoc function used to compute sliding window density and average value in window, if a y array is provided.
    :param unsorted_x: array
    :param y: array
    :return: bin_center, density, rolling_mean: array, array, array
    """
    indexes = list(range(len(unsorted_x)))
    indexes.sort(key=unsorted_x.__getitem__)
    sorted_x = list(map(unsorted_x.__getitem__, indexes))
    if y is not None:
        sorted_y = list(map(y.__getitem__, indexes))
    window_dur = bin_size * window_size
    bin_centers = np.arange(start+window_dur/2., end-window_dur/2.+bin_size, bin_size)
    density = np.zeros(len(bin_centers))
    rolling_mean = np.zeros(len(bin_centers))
    x0 = 0
    x1 = 0
    for i, bin in enumerate(bin_centers):
        while sorted_x[x0] < bin - window_dur / 2.:
            x0 += 1
            # x1 += 1
        while sorted_x[x1] < bin + window_dur / 2.:
            x1 += 1
        density[i] = (x1 - x0) / window_dur * 1000.
        if y is not None:
            rolling_mean[i] = np.mean(sorted_y[x0:x1])
    return bin_centers, density, rolling_mean


def flush_engine_buffer(result):
    """
    Once an async_result is ready, print the contents of its stdout buffer.
    :param result: :class:'ASyncResult
    """
    for stdout in result.stdout:
        if stdout:
            for line in stdout.splitlines():
                print(line)
    sys.stdout.flush()


def offset_vm(rec_name, context=None, vm_target=None, i_inc=0.005, vm_tol=0.5, i_history=None, dynamic=False,
              cvode=None):
    """

    :param rec_name: str
    :param context: :class:'Context'
    :param vm_target: float
    :param i_inc: float (nA)
    :param vm_tol: float (mV)
    :param i_history: defaultdict of dict
    :param dynamic: bool; whether to use a gradient-based approach to determine i_inc
    :param cvode: bool; whether to use adaptive time step
    """
    if context is None:
        raise RuntimeError('offset_vm: pid: %i; missing required Context object' % os.getpid())
    sim = context.sim
    if not sim.has_rec(rec_name):
        raise RuntimeError('offset_vm: pid: %i; no recording with name: %s' % (os.getpid(), rec_name))
    if not sim.has_stim('holding'):
        raise RuntimeError('offset_vm: pid: %i; missing required stimulus with name: \'holding\'' % os.getpid())

    if vm_target is None:
        vm_target = context.v_init
    if sim.has_stim('step'):
        sim.modify_stim('step', amp=0.)
    rec_dict = sim.get_rec(rec_name)
    node = rec_dict['node']
    loc = rec_dict['loc']
    rec = rec_dict['vec']

    equilibrate = context.equilibrate
    dt = context.dt
    duration = equilibrate

    if i_history is not None:
        if vm_target not in i_history[rec_name]:
            i_amp = 0.
            i_history[rec_name][vm_target] = i_amp
        else:
            i_amp = i_history[rec_name][vm_target]
    else:
        i_amp = 0.

    sim.modify_stim('holding', node=node, loc=loc, amp=i_amp)
    sim.backup_state()
    if cvode is None:
        cvode = True
    sim.set_state(dt=dt, tstop=duration, cvode=cvode)
    sim.run(vm_target)
    t = np.arange(0., duration, dt)
    vm = np.interp(t, sim.tvec, rec)
    vm_rest = np.mean(vm[int((duration - 3.) / dt):int((duration - 1.) / dt)])
    vm_before = vm_rest
    if sim.verbose:
        print('offset_vm: pid: %i; %s; vm_rest: %.1f, vm_target: %.1f' % (os.getpid(), rec_name, vm_rest, vm_target))

    if dynamic is True:
        dyn_i_inc = i_inc
        while not vm_target - vm_tol <= vm_rest <= vm_target + vm_tol:
            prev_i_inc = dyn_i_inc
            i_amp += dyn_i_inc
            if dyn_i_inc > 0:
                delta_str = 'increased'
            else:
                delta_str = 'decreased'
            prev_vm_rest = vm_rest
            sim.modify_stim('holding', amp=i_amp)
            sim.run(vm_target)
            vm = np.interp(t, sim.tvec, rec)
            vm_rest = np.mean(vm[int((duration - 3.) / dt):int((duration - 1.) / dt)])
            vm_inc = vm_rest - prev_vm_rest
            vm_diff = vm_target - vm_rest
            i_inc_mult = vm_diff / vm_inc
            dyn_i_inc = i_inc_mult * prev_i_inc
            if sim.verbose:
                print('offset_vm: pid: %i; %s; %s i_holding to %.3f nA; vm_rest: %.1f; i_inc_mult: %.1f; ' \
                      'i_inc: %.5f' % (os.getpid(), rec_name, delta_str, i_amp, vm_rest, i_inc_mult, dyn_i_inc))
    else:
        if vm_rest > vm_target:
            i_inc *= -1.
            delta_str = 'decreased'
            while vm_rest > vm_target - vm_tol:
                i_amp += i_inc
                sim.modify_stim('holding', amp=i_amp)
                sim.run(vm_target)
                vm = np.interp(t, sim.tvec, rec)
                vm_rest = np.mean(vm[int((duration - 3.) / dt):int((duration - 1.) / dt)])
                if sim.verbose:
                    print('offset_vm: pid: %i; %s; %s i_holding to %.3f nA; vm_rest: %.1f' % \
                          (os.getpid(), rec_name, delta_str, i_amp, vm_rest))
        if i_inc < 0.:
            i_inc *= -1.
        delta_str = 'increased'
        prev_vm_rest = vm_rest
        while vm_rest < vm_target:
            prev_vm_rest = vm_rest
            i_amp += i_inc
            sim.modify_stim('holding', amp=i_amp)
            sim.run(vm_target)
            vm = np.interp(t, sim.tvec, rec)
            vm_rest = np.mean(vm[int((duration - 3.) / dt):int((duration - 1.) / dt)])
            if sim.verbose:
                print('offset_vm: pid: %i; %s; %s i_holding to %.3f nA; vm_rest: %.1f' % \
                      (os.getpid(), rec_name, delta_str, i_amp, vm_rest))

        if abs(vm_rest - vm_target) > abs(prev_vm_rest - vm_target):
            i_amp -= i_inc
            vm_rest = prev_vm_rest
        sim.modify_stim('holding', amp=i_amp)

        if sim.verbose:
            print('offset_vm: pid: %i; %s; vm_rest: %.1f, vm_target: %.1f' % (os.getpid(), rec_name, vm_rest, vm_target))

    if i_history is not None:
        i_history[rec_name][vm_target] = i_amp
    sim.restore_state()
    vm_after = vm_rest
    return vm_before, vm_after, i_amp


def get_spike_adaptation_indexes(spike_times):
    """
    Spike rate adaptation refers to changes in inter-spike intervals during a spike train. Larger values indicate
    larger increases in inter-spike intervals.
    :param spike_times: list of float
    :return: array
    """
    if len(spike_times) < 3:
        return None
    isi = np.diff(spike_times)
    adi = []
    for i in range(len(isi) - 1):
        adi.append((isi[i + 1] - isi[i]) / (isi[i + 1] + isi[i]))
    return np.array(adi)


def get_thickest_dend_branch(cell, distance_target=None, sec_type='apical', distance_tolerance=50., terminal=None):
    """
    Get the thickest apical dendrite with a segment closest to a target distance from the soma.
    :param cell: "class:'BiophysCell'
    :param distance_target: float (um)
    :param sec_type: str
    :param distance_tolerance: float (um)
    :param terminal: bool
    :return: node, loc: :class:'SHocNode', float
    """
    candidate_distances = {}
    candidate_diams = {}
    candidate_locs = {}
    if sec_type not in cell.nodes:
        raise RuntimeError('get_thickest_dend_branch: pid: %i; %s cell %i: cannot find branch to satisfy '
                           'provided filter' % (os.getpid(), cell.pop_name, cell.gid))
    for branch in cell.nodes[sec_type]:
        if (terminal is None) or (terminal == is_terminal(branch)):
            for seg in branch.sec:
                loc = seg.x
                distance = get_distance_to_node(cell, cell.tree.root, branch, loc)
                if distance_target is None or distance_target < distance < distance_target + distance_tolerance:
                    abs_distance_to_target = abs(distance - distance_target)
                    if branch not in candidate_distances:
                        candidate_distances[branch] = abs_distance_to_target
                        candidate_diams[branch] = branch.sec(loc).diam
                        candidate_locs[branch] = loc
                    elif abs_distance_to_target < candidate_distances[branch]:
                        candidate_distances[branch] = abs_distance_to_target
                        candidate_diams[branch] = branch.sec(loc).diam
                        candidate_locs[branch] = loc

    if len(candidate_distances) == 0:
        raise RuntimeError('get_thickest_dend_branch: pid: %i; %s cell %i: cannot find branch to satisfy '
                           'provided filter' % (os.getpid(), cell.pop_name, cell.gid))
    elif len(candidate_distances) == 1:
        best_branch = next(iter(candidate_distances))
        best_loc = candidate_locs[branch]
    else:
        best_branch = None
        for branch in candidate_diams:
            if best_branch is None:
                best_branch = branch
            elif candidate_diams[branch] > candidate_diams[best_branch]:
                best_branch = branch
        best_loc = candidate_locs[best_branch]
    return best_branch, best_loc


def get_dend_segments(cell, ref_seg=None, sec_type='apical', soma_near=False, term_near=True, middle=True,
                      extra_locs=[], all_seg=False, dist_bounds=None, soma_dend=True, term_dend=True):
    """
    Get the patches along a dendritic branch 
    To get nodes at dendritic junctions, pass 0 and/or 1 in extra_locs

    :param cell: "class:'BiophysCell'
    :param sec_type: str
    :param soma_near: bool (include the dendritic segment nearest to the soma)
    :param thick_side: bool (include segment on the thicker side at a dendrite-dendrite junction)
    :param thin_side: bool (include segment on the thinner side at a dendrite-dendrite junction)
    :param middle: bool (include middle segment of the dendrite section) 
    :param extra_locs: list of floats (arbitrary segments)
    :return: node, loc: :class:'SHocNode', float
    """

    if sec_type not in cell.nodes:
        raise RuntimeError('get_dend_branch_patch: pid: {:d}; {!s} cell {:d}: {!s} does not exist'.format(os.getpid(), cell.pop_name, cell.gid, sec_type))

    # Populate dendrite list
    section_list=[]
    soma=cell.nodes['soma'][0]

    if ref_seg is None:
        for i in soma.children:
            if i.get_type() == sec_type:
                dend = i
                section_list.append(dend)
                break
        
        while len(section_list[-1].children):
            dendt = section_list[-1].children[0]
            section_list.append(dendt)
    else:
        par_lst = []
        chl_lst = []
        if ref_seg.parent is not None:
            if ref_seg.parent.type== 'apical': par_lst.append(ref_seg.parent)
        if len(par_lst):
            while par_lst[-1].parent.type == sec_type:
                par_lst.append(par_lst[-1].parent)    

        if len(ref_seg.children): chl_lst.append(ref_seg.children[0]) 
        if len(chl_lst):
            while len(chl_lst[-1].children):
                chl_lst.append(chl_lst[-1].children[0]) 

        if len(par_lst): par_lst.reverse()
        section_list = par_lst + [ref_seg] + chl_lst

    n_seg_list = [i.sec.nseg for i in section_list]

    N_dend = len(section_list)

    # Create segment indices
    soma_junc = True if 0 in extra_locs else False
    term_junc = True if 1 in extra_locs else False

    # Setting terminal centric scheme
    san_locs = [i for i in extra_locs if 0 < i <= 1] if soma_junc and term_junc else [i for i in extra_locs if 0 <= i <= 1]
    if middle: san_locs.append(0.5)

    san_locs = np.array(san_locs)

    seg_idx_list = [] 
    seg_idx_arr = np.empty(shape=(N_dend,2), dtype='O')

    # Add most proximal segment on section
    idx_lst = [1] if soma_near else []
    for i in range(N_dend):
        seg_idx_arr[i,0] = np.rint(san_locs*(n_seg_list[i]+1)).astype(np.int)
        seg_idx_arr[i,1] = [] + idx_lst 

    # Add most distal segment on section
    if term_near:
        for idx, nseg in enumerate(n_seg_list):
            seg_idx_arr[idx,1].append(nseg)
            
    # Special conditions for most proximal and most distal dendrites
    if soma_dend: seg_idx_arr[0,1].append(1)        
    if term_dend: seg_idx_arr[-1,1].append(n_seg_list[-1])

    N_segs = 0
    seg_lst = []
    tot_len = 0
    for i, obj in enumerate(section_list):
        tmp_lst = [j for j in obj.sec.allseg()]
        all_idx = np.union1d(seg_idx_arr[i,0], seg_idx_arr[i,1]).astype(np.int)
        if all_seg: all_idx = np.union1d(all_idx, np.arange(1, obj.sec.nseg+1))
        for k in all_idx: 
            frac_pos = tmp_lst[k].x
            seg_lst.append((obj, frac_pos, frac_pos*obj.sec.L + tot_len)) 
        tot_len += obj.sec.L

    node_loc_arr = np.array(seg_lst, dtype='O')

    if dist_bounds is not None:
        bds_idx = np.searchsorted(node_loc_arr[:,2], dist_bounds, side='right')
        bds_idx_lst = [i for i in range(*bds_idx)]
        if term_dend: bds_idx_lst.append(-1)
        node_loc_arr = node_loc_arr[np.array(bds_idx_lst), :] 

    return node_loc_arr 
   

def get_distal_most_terminal_branch(cell, distance_target=None, sec_type='apical'):
    """
    Get a terminal branch with a branch origin greater than a target distance from the soma.
    :param cell: "class:'BiophysCell'
    :param distance_target: float (um)
    :param sec_type: str
    :return: node: :class:'SHocNode'
    """
    candidate_branches = []
    candidate_distances = []
    if sec_type not in cell.nodes:
        raise RuntimeError('get_distal_most_terminal_branch: pid: %i; %s cell %i: cannot find branch to satisfy '
                           'provided filter' % (os.getpid(), cell.pop_name, cell.gid))
    for branch in cell.nodes[sec_type]:
        if is_terminal(branch):
            distance = get_distance_to_node(cell, cell.tree.root, branch, 0.)
            if distance_target is None or distance > distance_target:
                candidate_branches.append(branch)
                candidate_distances.append(get_distance_to_node(cell, cell.tree.root, branch, 1.))
    indexes = list(range(len(candidate_distances)))
    if len(indexes) == 0:
        raise RuntimeError('get_distal_most_terminal_branch: pid: %i; %s cell %i: cannot find branch to satisfy '
                           'provided filter' % (os.getpid(), cell.pop_name, cell.gid))
    elif len(indexes) == 1:
        index = 0
    else:
        index = np.argmax(candidate_distances)
    return candidate_branches[index]


def reset_biophysics(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('reset_biophysics: missing required Context object')
    init_biophysics(context.cell, reset_cable=False, reset_mech_dict=True, correct_g_pas=context.correct_for_spines,
                    env=context.env)


def reset_syn_mechanisms(x, context=None):
    """

    :param x: array
    :param context: :class:'Context'
    """
    if context is None:
        raise RuntimeError('reset_syn_mechanisms: missing required Context object')
    init_syn_mech_attrs(context.cell, env=context.env, reset_mech_dict=True, update_targets=True)


def log10_fit(x, slope, offset):
    """
    Use with scipy.optimize.curve_fit to obtain least-squares estimate of parameters of a log10 fit to data.
    :param x: float or array
    :param slope: float
    :param offset: float
    :return: float or array
    """
    return slope * np.log10(x) + offset


def inverse_log10_fit(y, slope, offset):
    """
    Obtain the x values for target y, given parameters of a log10 fit to data.
    :param y: float or array
    :param slope: float
    :param offset: float
    :return: float or array
    """
    return 10. ** ((y - offset) / slope)


def check_for_pause_in_spiking(spike_times, duration):
    """

    :param spike_times: array of float
    :param duration: float
    :return: bool
    """
    filtered_spike_times = np.array(spike_times)
    indexes = np.where((filtered_spike_times > 0.) & (filtered_spike_times < duration))[0]
    if len(indexes) >= 3:
        filtered_spike_times = filtered_spike_times[indexes]
        ISI_array = np.diff(filtered_spike_times)
        max_ISI = np.max(ISI_array)
        pause_dur = duration - filtered_spike_times[-1]
        if pause_dur > 2. * max_ISI:
            return True
    return False