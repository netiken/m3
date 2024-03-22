#include "topo.h"

static void pl_compute_baseL_two_layer(int l)
{
  totPE = l;
  totNode = l * 2;
  totLink = l * 2 - 1;
  totRack = l;
  n_pe_per_rack = 1;
}
static void pl_compute_baseL_one_layer(int l)
{
  totPE = l;
  totNode = l;
  totLink = l - 1;
  totRack = l;
  n_pe_per_rack = 1;
}

void pl_routing_init_two_layer()
{

  int sd_id, link_idx, link_id;
  int downlink_indicator;

  for (int src_rack = 0; src_rack < totRack; src_rack++)
  {
    for (int dst_rack = 0; dst_rack < totRack; dst_rack++)
    {
      if (src_rack == dst_rack)
        continue;
      sd_id = sd_to_sdid[src_rack][dst_rack];
      if (src_rack < dst_rack)
      {
        downlink_indicator = 0;
      }
      else
      {
        downlink_indicator = 1;
      }
      // add host links
      for (link_id = src_rack * n_pe_per_rack; link_id < src_rack * n_pe_per_rack + n_pe_per_rack; link_id++)
      {
        sdid_to_linkid[sd_id][sdid_to_nlink[sd_id][UPLINK]][UPLINK] = link_id;
        sdid_to_nlink[sd_id][UPLINK]++;

        linkid_to_sdid[link_id][linkid_to_nsd[link_id][UPLINK]][UPLINK] = sd_id;
        linkid_to_nsd[link_id][UPLINK]++;
      }
      for (link_id = dst_rack * n_pe_per_rack; link_id < dst_rack * n_pe_per_rack + n_pe_per_rack; link_id++)
      {
        sdid_to_linkid[sd_id][sdid_to_nlink[sd_id][DOWNLINK]][DOWNLINK] = link_id;
        sdid_to_nlink[sd_id][DOWNLINK]++;

        linkid_to_sdid[link_id][linkid_to_nsd[link_id][DOWNLINK]][DOWNLINK] = sd_id;
        linkid_to_nsd[link_id][DOWNLINK]++;
      }
      // add internal links
      for (link_idx = 0; link_idx < abs(dst_rack - src_rack); link_idx++)
      {
        link_id = totPE + min(src_rack, dst_rack) + link_idx;
        sdid_to_linkid[sd_id][sdid_to_nlink[sd_id][downlink_indicator]][downlink_indicator] = link_id;
        sdid_to_nlink[sd_id][downlink_indicator]++;

        linkid_to_sdid[link_id][linkid_to_nsd[link_id][downlink_indicator]][downlink_indicator] = sd_id;
        linkid_to_nsd[link_id][downlink_indicator]++;
      }
    }
  }
}
void pl_routing_init_one_layer()
{
  int sd_id, link_idx, link_id;
  int downlink_indicator;

  for (int src_rack = 0; src_rack < totRack; src_rack++)
  {
    for (int dst_rack = 0; dst_rack < totRack; dst_rack++)
    {
      if (src_rack == dst_rack)
        continue;
      sd_id = sd_to_sdid[src_rack][dst_rack];
      if (src_rack < dst_rack)
      {
        downlink_indicator = 0;
      }
      else
      {
        downlink_indicator = 1;
      }
      for (link_idx = 0; link_idx < abs(dst_rack - src_rack); link_idx++)
      {
        link_id = min(src_rack, dst_rack) + link_idx;
        sdid_to_linkid[sd_id][sdid_to_nlink[sd_id][downlink_indicator]][downlink_indicator] = link_id;
        sdid_to_nlink[sd_id][downlink_indicator]++;

        linkid_to_sdid[link_id][linkid_to_nsd[link_id][downlink_indicator]][downlink_indicator] = sd_id;
        linkid_to_nsd[link_id][downlink_indicator]++;
      }
    }
  }
}

void pl_build_topology_two_layer(int l, long long int *BW)
{
  int i, j;
  int flowid, linkid, sd_id, node_level;

  for (linkid = 0; linkid < totLink; linkid++)
  {
    node_level = linkid / l;
    linkid_to_bw_ori[linkid][UPLINK] = BW[node_level];
    linkid_to_bw_ori[linkid][DOWNLINK] = BW[node_level];
    linkid_to_load[linkid][UPLINK] = 0;
    linkid_to_load[linkid][DOWNLINK] = 0;
    linkid_to_bw[linkid][UPLINK] = BW[node_level];
    linkid_to_bw[linkid][DOWNLINK] = BW[node_level];
    linkid_to_fanout[linkid] = 1;
    linkid_to_nsd[linkid][UPLINK] = 0;
    linkid_to_nsd[linkid][DOWNLINK] = 0;
  }

  for (flowid = 0; flowid < MAX_NFLOW; flowid++)
  {
    final_flow_vector[flowid] = -1;
  }

  // nsd_active = 0;
  sd_id = 0;
  for (i = 0; i < totRack; i++)
  {
    for (j = 0; j < totRack; j++)
    {
      if (i != j)
      {
        sd_to_sdid[i][j] = sd_id;
        sdid_to_nlink[sd_id][UPLINK] = 0;
        sdid_to_nlink[sd_id][DOWNLINK] = 0;
        sdid_to_nflow[sd_id] = 0;
        sdid_status[sd_id] = 0;
        sd_id++;
      }
    }
  }
}

void pl_build_topology_one_layer(int l, long long int *BW)
{
  int i, j;
  int flowid, linkid,sd_id, node_level;

  for (linkid = 0; linkid < totLink; linkid++)
  {
    if ((linkid==0) || (linkid==totLink-1)) {
      node_level = 0;
    } else {
      node_level = 1;
    }
    linkid_to_bw_ori[linkid][UPLINK] = BW[node_level];
    linkid_to_bw_ori[linkid][DOWNLINK] = BW[node_level];
    linkid_to_load[linkid][UPLINK] = 0;
    linkid_to_load[linkid][DOWNLINK] = 0;
    linkid_to_bw[linkid][UPLINK] = BW[node_level];
    linkid_to_bw[linkid][DOWNLINK] = BW[node_level];
    linkid_to_fanout[linkid] = 1;
    linkid_to_nsd[linkid][UPLINK] = 0;
    linkid_to_nsd[linkid][DOWNLINK] = 0;
  }

  for (flowid = 0; flowid < MAX_NFLOW; flowid++)
  {
    final_flow_vector[flowid] = -1;
  }

  // nsd_active = 0;
  sd_id = 0;
  for (i = 0; i < totRack; i++)
  {
    for (j = 0; j < totRack; j++)
    {
      if (i != j)
      {
        sd_to_sdid[i][j] = sd_id;
        sdid_to_nlink[sd_id][UPLINK] = 0;
        sdid_to_nlink[sd_id][DOWNLINK] = 0;
        sdid_to_nflow[sd_id] = 0;
        sdid_status[sd_id] = 0;
        sd_id++;
      }
    }
  }
}

void pl_reset_topology_two_layer()
{
  int i, j;
  int sd_id, node_level, flowid, linkid;

  for (linkid = 0; linkid < totLink; linkid++)
  {
    node_level = linkid / pl_l;
    linkid_to_bw[linkid][0] = pl_BW[node_level];
    linkid_to_bw[linkid][1] = pl_BW[node_level];
    linkid_to_load[linkid][0] = 0;
    linkid_to_load[linkid][1] = 0;
    // linkid_to_nsd[linkid][0] = 0;
    // linkid_to_nsd[linkid][1] = 0;
  }

  for (flowid = 0; flowid < MAX_NFLOW; flowid++)
  {
    if (final_flow_vector[flowid] == -1) break;
    final_flow_vector[flowid] = -1;
  }
  sd_id = 0;
  for (i = 0; i < totRack; i++)
  {
    for (j = 0; j < totRack; j++)
    {
      if (i != j)
      {
        // sd_to_sdid[i][j] = sd_id;
        sdid_to_nflow[sd_id] = 0;
        sdid_status[sd_id] = 0;
        sd_id++;
      }
    }
  }
}

void pl_reset_topology_one_layer()
{
  int i, j;
  int sd_id, flowid, linkid, node_level;

  for (linkid = 0; linkid < totLink; linkid++)
  {
    if ((linkid==0) || (linkid==totLink-1)) {
      node_level = 0;
    } else {
      node_level = 1;
    }
    linkid_to_bw[linkid][0] = pl_BW[node_level];
    linkid_to_bw[linkid][1] = pl_BW[node_level];
    linkid_to_load[linkid][0] = 0;
    linkid_to_load[linkid][1] = 0;
    // linkid_to_nsd[linkid][0] = 0;
    // linkid_to_nsd[linkid][1] = 0;
  }

  for (flowid = 0; flowid < MAX_NFLOW; flowid++)
  {
    if (final_flow_vector[flowid] == -1) break;
    final_flow_vector[flowid] = -1;
  }
  sd_id = 0;
  for (i = 0; i < totRack; i++)
  {
    for (j = 0; j < totRack; j++)
    {
      if (i != j)
      {
        // sd_to_sdid[i][j] = sd_id;
        sdid_to_nflow[sd_id] = 0;
        sdid_status[sd_id] = 0;
        sd_id++;
      }
    }
  }
}


void pl_topology_init_two_layer(int l, long long int *bw)
{
  int i;
  pl_l = l;

  for (i = 0; i < 2; i++)
  {
    pl_BW[i] = bw[i];
  }
  // determine sizeL, baseL, baseL_link, node_level
  pl_compute_baseL_two_layer(pl_l);
  pl_build_topology_two_layer(l, bw);

  printf("Simulating two-layer parking lot: length = %d\n  bw = [", l);

  for (i = 0; i < totLink-1; i++)
    printf("%lf, ", *linkid_to_bw_ori[i]);
  printf("%lf]\n", *linkid_to_bw_ori[i]);

  printf("%d PEs, %d switches, %d nodes (switch+PE), %d links\n", totPE, totNode - totPE, totNode, totLink);
}
void pl_topology_init_one_layer(int l, long long int *bw)
{
  int i;
  pl_l = l;

  for (i = 0; i < 2; i++)
  {
    pl_BW[i] = bw[i];
  }
  // determine sizeL, baseL, baseL_link, node_level
  pl_compute_baseL_one_layer(pl_l);
  pl_build_topology_one_layer(l, bw);

  // printf("Simulating one-layer parking lot: length = %d\n  bw = [", l);

  // for (i = 0; i < totLink-1; i++)
  //   printf("%lf, ", *linkid_to_bw_ori[i]);
  // printf("%lf]\n", *linkid_to_bw_ori[i]);

  // printf("%d PEs, %d switches, %d nodes (switch+PE), %d links\n", totPE, totNode - totPE, totNode, totLink);
}


/** utility function
 *  returns elapsed time of the given interval in seconds
 */
double timediff(struct timeval start, struct timeval end)
{
  return (double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)) / 1000000.0);
}

void pl_ppf_from_array(unsigned int nflow_active, int *src, int *dst, int *iteration_count, double *exec_time)
{
  // struct timeval t1,t2,t3,t4;
  struct timeval t1, t4;

  int iterator = 0;
  long long int nflow_done;
  int src_rack, dst_rack, sd_id, sd_idx, link_id, link_idx, flow_id, flow_idx, downlink_indicator;
  long long int min_rate_limit_linkid;
  double rate_limit_loadfactor, min_rate_limit;

  // %% Step 1: initialize linked lists
  gettimeofday(&t1, NULL);
  nsd_active = 0;
  assert(nflow_active < MAX_NFLOW);
  for (flow_id = 0; flow_id < nflow_active; flow_id++)
  {
    assert(src[flow_id] != dst[flow_id]);
    src_rack = floor(src[flow_id] / n_pe_per_rack);
    dst_rack = floor(dst[flow_id] / n_pe_per_rack);

    sd_id = sd_to_sdid[src_rack][dst_rack];
    sdid_to_flowid[sd_id][sdid_to_nflow[sd_id]] = flow_id;
    if (sdid_to_nflow[sd_id] == 0)
    {
      sdid_status[sd_id] = 1;
      sdid_active[nsd_active] = sd_id;
      nsd_active++;
    }
    sdid_to_nflow[sd_id]++;
  }
  // %% Step 2: calculate the num of flows for each link
  // gettimeofday(&t2, NULL);
  // printf("Time to initialize: %lf\n", timediff(t1,t2));

  for (sd_idx = 0; sd_idx < nsd_active; sd_idx++)
  {
    sd_id = sdid_active[sd_idx];
    for (downlink_indicator = 0; downlink_indicator < 2; downlink_indicator++)
    {
      for (link_idx = 0; link_idx < sdid_to_nlink[sd_id][downlink_indicator]; link_idx++)
      {
        link_id = sdid_to_linkid[sd_id][link_idx][downlink_indicator];
        linkid_to_load[link_id][downlink_indicator] += sdid_to_nflow[sd_id];
      }
    }
  }

  // Step %% 3: begin iterative algorithm
  //  gettimeofday(&t3, NULL);
  //  printf("Flows processed during input %d\n", nflow_active);

  nflow_done = 0;
  while (nflow_done != nflow_active)
  {
    iterator++;
    min_rate_limit = LLONG_MAX; // temporary
    min_rate_limit_linkid = -1;
    rate_limit_loadfactor = -1;

    // STEP 1: Find the most rate limiting link(L)
    for (link_id = 0; link_id < totLink; link_id++)
    {
      for (downlink_indicator = 0; downlink_indicator < 2; downlink_indicator++)
      {
        if (linkid_to_load[link_id][downlink_indicator] != 0)
        {
          double load_factor;
          load_factor = (linkid_to_load[link_id][downlink_indicator] * 1.0) / linkid_to_fanout[link_id];

          rate_limit_per_link[link_id][downlink_indicator] = linkid_to_bw[link_id][downlink_indicator] / load_factor;

          // STEP 2: compute the smallest rate that every flow can increase to
          if (rate_limit_per_link[link_id][downlink_indicator] < min_rate_limit)
          {
            min_rate_limit = rate_limit_per_link[link_id][downlink_indicator];
            min_rate_limit_linkid = link_id;
            rate_limit_loadfactor = load_factor; // for debugging
          }
        }
      }
    }

    // printf("At iteration %d, most limiting rate %lf\n", iterator, min_rate_limit);

    for (link_id = 0; link_id < totLink; link_id++)
    {
      for (downlink_indicator = 0; downlink_indicator < 2; downlink_indicator++)
      {
        if (fabs(rate_limit_per_link[link_id][downlink_indicator] - min_rate_limit) < 0.0001)
        {
          for (sd_idx = 0; sd_idx < linkid_to_nsd[link_id][downlink_indicator]; sd_idx++)
          {
            sd_id = linkid_to_sdid[link_id][sd_idx][downlink_indicator];
            // if (sdid_status[sd_id])
            // {
            for (flow_idx = 0; flow_idx < sdid_to_nflow[sd_id]; flow_idx++)
            {
              flow_id = sdid_to_flowid[sd_id][flow_idx];
              if (final_flow_vector[flow_id] == -1)
              {
                final_flow_vector[flow_id] = min(min_rate_limit,pl_BW[0]);
                // final_flow_vector[flow_id] = min_rate_limit;
                nflow_done++;
              }
            }
            //   sdid_status[sd_id] = 0;
            // }
          }
        }
      }
    }

    // printf("Number of flows saturated in itern %d is %lld\n",iterator,nflow_done);

    // STEP 4: update the link loads of saturated flows
    // STEP 6.1: Update the available bw at each link
    double used_bw;

    for (link_id = 0; link_id < totLink; link_id++)
    {
      for (downlink_indicator = 0; downlink_indicator < 2; downlink_indicator++)
      {
        used_bw = 0;
        linkid_to_load[link_id][downlink_indicator] = 0;
        for (sd_idx = 0; sd_idx < linkid_to_nsd[link_id][downlink_indicator]; sd_idx++)
        {
          sd_id = linkid_to_sdid[link_id][sd_idx][downlink_indicator];
          for (flow_idx = 0; flow_idx < sdid_to_nflow[sd_id]; flow_idx++)
          {
            flow_id = sdid_to_flowid[sd_id][flow_idx];
            if (final_flow_vector[flow_id] == -1)
              linkid_to_load[link_id][downlink_indicator]++;
            else
              used_bw += final_flow_vector[flow_id];
          }
        }
        linkid_to_bw[link_id][downlink_indicator] = linkid_to_bw_ori[link_id][downlink_indicator] - used_bw * 1.0 / linkid_to_fanout[link_id];
      }
    }

    // for (link_id = 0; link_id < totLink; link_id++)
    // {
    //   for (downlink_indicator = 0; downlink_indicator < 2; downlink_indicator++)
    //   {
    //     used_bw = 0;
    //     for (sd_idx = 0; sd_idx < linkid_to_nsd[link_id][downlink_indicator]; sd_idx++)
    //     {
    //       sd_id = linkid_to_sdid[link_id][sd_idx][downlink_indicator];
    //       for (flow_idx = 0; flow_idx < sdid_to_nflow[sd_id]; flow_idx++)
    //       {
    //         flow_id = sdid_to_flowid[sd_id][flow_idx];
    //         if (fabs(final_flow_vector[flow_id] - min_rate_limit) < 0.0001)
    //         {
    //           used_bw += final_flow_vector[flow_id];
    //           linkid_to_load[link_id][downlink_indicator]--;
    //         }
    //       }
    //     }
    //     linkid_to_bw[link_id][downlink_indicator] -= used_bw * 1.0 / linkid_to_fanout[link_id];
    //   }
    // }

  } // end while
  // Step %% 4: end iterative algorithm
  gettimeofday(&t4, NULL);

  *iteration_count = iterator;
  *exec_time = timediff(t1, t4);
}