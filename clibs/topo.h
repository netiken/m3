/*
 * This file defines the common constants and extern variables
 * for all topologies
 */

#ifndef TOPO_H
#define TOPO_H
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <stdbool.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define UPLINK 0
#define DOWNLINK 1

// topo
#define XGFT 1 // extended fat tree
#define PL 2   // parking lot

// max-min fair rate for topo
#define PL_TWO_LAYER 1
#define PL_ONE_LAYER 2

// routing
#define PL_ECMP_ROUTING 8

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#define MAX_NODE 16 // Max Number of Nodes in the network
#define MAX_LINK 15
#define MAX_RACK 8
#define MAX_PATH_LEN 8

#define MAX_NFLOW 50000
#define MAX_NFLOWS_PER_SD 50000
#define MAX_NPATH_PER_SD 1
#define MAX_NLINK_PER_SD MAX_NPATH_PER_SD *MAX_PATH_LEN
#define MAX_SD MAX_RACK *(MAX_RACK - 1)

// parking lot
int pl_l;
int pl_BW[2];

// common
int totNode; /**< Total number of nodes, including processing and switch elements. */
int totPE;   /**< Total number of processing elements */
int totLink; /**< Total number of processing elements */
int totRack; /**< Total number of processing elements */
int n_pe_per_rack;
int nsd_active;

int linkid_to_load[MAX_LINK][2];
double linkid_to_bw[MAX_LINK][2];
double linkid_to_bw_ori[MAX_LINK][2];
int linkid_to_fanout[MAX_LINK];
int linkid_to_nsd[MAX_LINK][2];
int linkid_to_sdid[MAX_LINK][MAX_SD][2];

int sd_to_sdid[MAX_RACK][MAX_RACK];
int sdid_to_nlink[MAX_SD][2];
int sdid_to_linkid[MAX_SD][MAX_NLINK_PER_SD][2];
int sdid_to_nflow[MAX_SD];
int sdid_to_flowid[MAX_SD][MAX_NFLOWS_PER_SD];
int sdid_status[MAX_SD];
int sdid_active[MAX_SD];

double rate_limit_per_link[MAX_LINK][2];
double final_flow_vector[MAX_NFLOW]; // only used in  NON_LP algorithm

void pl_topology_init_two_layer(int h, long long int *bw);
static void pl_compute_baseL_two_layer(int l);
void pl_build_topology_two_layer(int l, long long int *BW);
void pl_routing_init_two_layer();
void pl_reset_topology_two_layer();

void pl_topology_init_one_layer(int h, long long int *bw);
static void pl_compute_baseL_one_layer(int l);
void pl_build_topology_one_layer(int l, long long int *BW);
void pl_routing_init_one_layer();
void pl_reset_topology_one_layer();

// max-min fair rate
void pl_ppf_from_array(unsigned int traffic_count, int *src, int *dst, int *iteration_count, double *exec_time);

// utils
double timediff(struct timeval start, struct timeval end);
