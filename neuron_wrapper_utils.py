"""
Tools for pulling individual neurons out of the dentate network simulation environment for single-cell tuning.
"""
__author__ = 'Ivan Raikov, Grace Ng, Aaron D. Milstein'
import sys, os
import os.path
import click
import itertools
from collections import defaultdict
from datetime import datetime
import numpy as np
from mpi4py import MPI  # Must come before importing NEURON
import h5py
from neuron import h
from neuroh5.h5py_io_utils import *
from dentate.env import Env
from dentate.cells import *
from dentate.neuron_utils import *
from nested.utils import *


context = Context()


def make_hoc_cell(env, gid, population):
    """

    :param env:
    :param gid:
    :param population:
    :return:
    """
    datasetPath = os.path.join(env.datasetPrefix, env.datasetName)
    popName = population
    templateName = env.celltypes[popName]['template']
    h.find_template(env.pc, h.templatePaths, templateName)
    dataFilePath = os.path.join(datasetPath, env.modelConfig['Cell Data'])
    context.dataFilePath = dataFilePath
    templateName = env.celltypes[popName]['template']
    templateClass = eval('h.%s' % templateName)

    if env.cellAttributeInfo.has_key(popName) and env.cellAttributeInfo[popName].has_key('Trees'):
        tree = select_tree_attributes(gid, env.comm, dataFilePath, popName)
        i = h.numCells
        hoc_cell = make_neurotree_cell(templateClass, neurotree_dict=tree, gid=gid, local_id=i,
                                             dataset_path=datasetPath)
        h.numCells = h.numCells + 1
    else:
        raise Exception('make_hoc_cell: data file: %s does not contain morphology for population: %s, gid: %i' %
                        dataFilePath, popName, gid)
    return hoc_cell


def configure_env(env, hoc_lib_path):
    """

    :param env:
    """
    h.load_file("nrngui.hoc")
    h.load_file("loadbal.hoc")
    h('objref fi_status, fi_checksimtime, pc, nclist, nc, nil')
    h('strdef datasetPath')
    h('numCells = 0')
    h('totalNumCells = 0')
    h.nclist = h.List()
    datasetPath = os.path.join(env.datasetPrefix, env.datasetName)
    h.datasetPath = datasetPath
    h.pc = h.ParallelContext()
    env.pc = h.pc
    ## polymorphic value template
    h.load_file(hoc_lib_path + "/templates/Value.hoc")
    ## randomstream template
    h.load_file(hoc_lib_path + "/templates/ranstream.hoc")
    ## stimulus cell template
    h.load_file(hoc_lib_path + "/templates/StimCell.hoc")
    h.xopen(hoc_lib_path + "/lib.hoc")
    h('objref templatePaths, templatePathValue')
    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePathValue = h.Value(1, path)
        h.templatePaths.append(h.templatePathValue)


def init_env(config_file, template_paths, hoc_lib_path, dataset_prefix=None, results_path=None, verbose=False,
             **kwargs):
    """

    :param config_file:
    :param template_paths:
    :param hoc_lib_path:
    :param dataset_prefix:
    :param results_path:
    :param verbose: bool
    :param kwargs:
    :return:
    """
    np.seterr(all='raise')
    comm = MPI.COMM_WORLD
    env = Env(comm, config_file, template_paths, dataset_prefix, results_path, verbose, **kwargs)
    configure_env(env, hoc_lib_path)
    return env


def get_hoc_cell_wrapper(env, gid, pop_name):
    """

    :param env:
    :param gid:
    :param pop_name:
    :return:
    """
    hoc_cell = make_hoc_cell(env, gid, pop_name)
    #cell = HocCell(existing_hoc_cell=hoc_cell)
    # cell.load_morphology()
    cell = HocCell(gid=0, population='GC', hoc_cell=hoc_cell)
    cell_attr_index_map = get_cell_attributes_index_map(env.comm, context.dataFilePath, 'GC', 'Synapse Attributes')
    cell_attr_dict = select_cell_attributes(gid, env.comm, context.dataFilePath, cell_attr_index_map, 'GC', 'Synapse Attributes')

    #Need to add dictionary info to synapse_attributes of each node?
    context.update(locals())
    return cell


@click.command()
@click.option("--gid", required=True, type=int, default=0)
@click.option("--pop-name", required=True, type=str, default='GC')
@click.option("--config-file", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='../dentate/config/Small_Scale_Control_log_normal_weights.yaml')
@click.option("--template-paths", type=str, default='../dgc/Mateos-Aparicio2014:../dentate/templates')
@click.option("--hoc-lib-path", type=str, default='../dentate')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='/mnt/s') #'../dentate'
@click.option("--results-path", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='data')
@click.option('--verbose', '-v', is_flag=True)
def main(gid, pop_name, config_file, template_paths, hoc_lib_path, dataset_prefix, results_path, verbose):
    """

    :param gid:
    :param pop_name:
    :param config_file:
    :param template_paths:
    :param hoc_lib_path:
    :param dataset_prefix:
    :param results_path:
    :param verbose
    """
    env = init_env(config_file=config_file, template_paths=template_paths, hoc_lib_path=hoc_lib_path,
                   dataset_prefix=dataset_prefix, results_path=results_path, verbose=verbose)
    cell = get_hoc_cell_wrapper(env, gid, pop_name)
    context.update(locals())


if __name__ == '__main__':
    main(args=sys.argv[(sys.argv.index(os.path.basename(__file__)) + 1):])
