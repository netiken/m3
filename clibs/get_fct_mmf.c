#include "topo.h"
struct FCTStruct
{
    double *estimated_fcts;
    double *t_flows;
    unsigned int *num_flows;
    unsigned int *num_flows_enq;
};

typedef struct
{
    double remaining_size;
} Flow;

struct FCTStruct get_fct_mmf(unsigned int n_flows, double *fats, double *sizes, int *src, int *dst, int h, int *topo, int method_mmf, int method_routing, int type_topo, int base_lr);
void free_fctstruct(struct FCTStruct input_struct);
void update_rate_mmf(unsigned int traffic_count, int *src, int *dst, int method_mmf, int type_topo);

void update_rate_mmf(unsigned int traffic_count, int *src, int *dst, int method_mmf, int type_topo)
{
    int iteration_count = 0;
    double exec_time = 0.0;
    
    pl_ppf_from_array(traffic_count, src, dst, &iteration_count, &exec_time);
      
    // long long int tot_mmf_bw=0;
    // for(int i=0;i<traffic_count;i++){
    //     //fprintf(ofd, "%lld\n", final_flow_vector[i]);
    //     tot_mmf_bw+=final_flow_vector[i];
    // }

    // printf("\nAggregate throughput for method %d is %8.6lf\n %d\t %8.6lf (s)\n", method_mmf, tot_mmf_bw*1.0 / traffic_count, iteration_count, exec_time);

    // int i;
    // printf("final_flow_vector = [");
    // for (i = 0; i < traffic_count - 1; i++)
    //     printf("%f, ", final_flow_vector[i]);
    // printf("%f]\n", final_flow_vector[i]);
    // printf("%d\t %8.6lf (s)\n", iteration_count, exec_time);
    // printf("%d\t %d\n", iteration_count, traffic_count);
}

// res = C_LIB.get_fct_mmf(n_flows, fats_pt, sizes_pt, src_pt, dst_pt, nhost, topo_pt, 1, 8, 2, bw)
struct FCTStruct get_fct_mmf(unsigned int n_flows, double *fats, double *sizes, int *src, int *dst, int h, int *topo, int method_mmf, int method_routing, int type_topo, int base_lr)
{
    assert (type_topo==PL);
    assert (method_routing==PL_ECMP_ROUTING);
    if (method_mmf==PL_TWO_LAYER){
        long long int BW[2];
        for (int i = 0; i < 2; i++)
            BW[i] = topo[i] * ((long long int)base_lr);
        pl_topology_init_two_layer(h, BW);
        pl_routing_init_two_layer();
    }
    else if (method_mmf==PL_ONE_LAYER){
        long long int BW[2];
        for (int i = 0; i < 2; i++)
            BW[i] = topo[i] * ((long long int)base_lr);
        pl_topology_init_one_layer(h, BW);
        pl_routing_init_one_layer();
    }
    else{
        assert(false);
    }

    // printf("n_flows: %u\n", n_flows);
    // printf("fats:");
    // for (int i = 0; i < n_flows; i++)
    // {
    //     printf("%f ", fats[i]);
    // }
    // printf("\n");
    // printf("sizes:");
    // for (int i = 0; i < n_flows; i++)
    // {
    //     printf("%u ", sizes[i]);
    // }
    // printf("\n");
    // printf("weights:");
    // for (int i = 0; i < n_flows; i++)
    // {
    //     printf("%f ", weights[i]);
    // }
    // printf("\n");
    struct FCTStruct res;
    double t = 0.0;
    unsigned int j = 0;
    unsigned int t_index = 0;
    Flow *active_flows = (Flow *)malloc(n_flows * sizeof(Flow));
    unsigned int *active_flows_idx = (unsigned int *)malloc(n_flows * sizeof(unsigned int));
    double *estimated_fcts = (double *)malloc(n_flows * sizeof(double));
    double *t_flows = (double *)malloc((2 * n_flows) * sizeof(double));
    unsigned int *num_flows = (unsigned int *)malloc((2 * n_flows) * sizeof(unsigned int));
    unsigned int *num_flows_enq = (unsigned int *)malloc((n_flows) * sizeof(unsigned int));
    // double lr = 10.0;

    memset(estimated_fcts, 0.0, n_flows * sizeof(double));
    memset(num_flows, 0, 2 * n_flows * sizeof(unsigned int));
    memset(num_flows_enq, 0, n_flows * sizeof(unsigned int));
    // double a_nan = strtod("NaN", NULL);
    double time_to_next_arrival = NAN;
    double time_to_next_completion = NAN;
    unsigned int num_active_flows = 0;
    double sum_weights = 0.0;
    int min_remaining_time_index = -1;

    int *src_active = (int *)malloc(n_flows * sizeof(int));
    int *dst_active = (int *)malloc(n_flows * sizeof(int));

    while (true)
    {
        if (j < n_flows)
        {
            time_to_next_arrival = fats[j] - t;
            // printf("time_to_next_arrival:%f\n", time_to_next_arrival);
            assert(time_to_next_arrival >= 0);
        }
        else
        {
            time_to_next_arrival = NAN;
        }
        min_remaining_time_index = -1;
        if (num_active_flows)
        {
            update_rate_mmf(num_active_flows, src_active, dst_active, method_mmf, type_topo);

            time_to_next_completion = INFINITY;
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                double remaining_time = active_flows[flow_idx].remaining_size / final_flow_vector[i];
                if (remaining_time < time_to_next_completion)
                {
                    time_to_next_completion = remaining_time;
                    min_remaining_time_index = i;
                }
            }
        }
        else
        {
            time_to_next_completion = NAN;
        }

        if (num_active_flows > 0 && (j >= n_flows || time_to_next_completion <= time_to_next_arrival))
        {
            // Completion Event
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                estimated_fcts[flow_idx] += time_to_next_completion;
                active_flows[flow_idx].remaining_size -= time_to_next_completion * final_flow_vector[i];
            }
            t += time_to_next_completion;
            num_active_flows -= 1;
            assert(min_remaining_time_index != -1);
            active_flows_idx[min_remaining_time_index] = active_flows_idx[num_active_flows];
            src_active[min_remaining_time_index] = src_active[num_active_flows];
            dst_active[min_remaining_time_index] = dst_active[num_active_flows];
        }
        else
        {
            // Arrival Event
            if (j >= n_flows)
            {
                // No more flows left - terminate
                break;
            }
            for (int i = 0; i < num_active_flows; i++)
            {
                unsigned int flow_idx = active_flows_idx[i];
                estimated_fcts[flow_idx] += time_to_next_arrival;
                active_flows[flow_idx].remaining_size -= time_to_next_arrival * final_flow_vector[i];
            }
            t += time_to_next_arrival;
            active_flows[j].remaining_size = (sizes[j] + ceil(sizes[j] / 1000.0) * 48.0) * 8.0;
            // active_flows[j].remaining_size = sizes[j] * 8.0;
            active_flows_idx[num_active_flows] = j;
            src_active[num_active_flows] = src[j];
            dst_active[num_active_flows] = dst[j];
            num_active_flows += 1;
            num_flows_enq[j] = num_active_flows;
            j += 1;
        }
        if (method_mmf==PL_TWO_LAYER) {
            pl_reset_topology_two_layer();
        }
        else if (method_mmf==PL_ONE_LAYER) {
            pl_reset_topology_one_layer();
        }
        else{
            assert(false);
        }
        t_flows[t_index] = t;
        num_flows[t_index] = num_active_flows;
        t_index += 1;
        // if (j % 100000 == 0)
        // {
        //     printf("%d/%d simulated in seconds\n", j, n_flows);
        // }
    }

    res.estimated_fcts = estimated_fcts;
    res.t_flows = t_flows;
    res.num_flows = num_flows;
    res.num_flows_enq = num_flows_enq;
    free(active_flows_idx);
    free(src_active);
    free(dst_active);
    free(active_flows);
    // free(estimated_fcts);
    // free(t_flows);
    // free(num_flows);
    // free(num_flows_enq);
    return res;
}

void free_fctstruct(struct FCTStruct input_struct)
{
    free(input_struct.estimated_fcts);
    free(input_struct.t_flows);
    free(input_struct.num_flows);
    free(input_struct.num_flows_enq);
}

int main(int argc, char *argv[])
{

    int method_routing = atoi(argv[1]);
    int method_mmf = atoi(argv[2]);
    int type_topo = atoi(argv[3]);

    int l = 5;
    int topo[2] = {1, 4};
    long long int BW[2];

    for (int i = 0; i < 2; i++)
        BW[i] = topo[i] * ((long long int)10);

    if (method_mmf==PL_TWO_LAYER){
        pl_topology_init_two_layer(l, BW);
        pl_routing_init_two_layer();
    }
    else if (method_mmf==PL_ONE_LAYER){
        pl_topology_init_one_layer(l, BW);
        pl_routing_init_one_layer();
    }
    else{
        assert(false);
    }
    
    
    unsigned int num_scenarios = atoi(argv[4]);

    for (int i = 0; i < num_scenarios; i++)
    {
        unsigned int num_active_flows = atoi(argv[5 + i]);

        // int* src_active = (int *) malloc(sizeof(int) * num_active_flows);
        // for (int i=0; i<num_active_flows; i++)
        //     src_active[i] = i%totPE;

        // int* dst_active = (int *) malloc(sizeof(int) * num_active_flows);
        // for (int i=0; i<num_active_flows; i++)
        //     dst_active[i] = (totPE-1-i)%totPE;

        // int array1[6] = {0, 1, 2, 3, 0, 1};
        // int(*src_active)[6] = &array1;

        // int array2[6] = {4, 4, 4, 4, 1, 2};
        // int(*dst_active)[6] = &array2;
        int array1[6] = {0, 1, 1, 1, 2, 3};
        int(*src_active)[6] = &array1;

        int array2[6] = {4, 2, 2, 3, 3, 4};
        int(*dst_active)[6] = &array2;

        update_rate_mmf(num_active_flows, src_active, dst_active, method_mmf, type_topo);
        pl_reset_topology_two_layer();
        if (method_mmf==PL_TWO_LAYER){
            pl_reset_topology_two_layer();
        }
        else if (method_mmf==PL_ONE_LAYER){
            pl_reset_topology_one_layer();
        }
        else{
            assert(false);
        }
    }
}