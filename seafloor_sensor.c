/*
This file consists of the code to simulate a seafloor seismic sensor node.

The following functions are referenced from https://www.geodatasource.com/developers/c
- distance
- deg2rad
- rad2deg
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

#define AVG_CYCLE 3     // Cycle to generate values
#define VAL_RANGE 3    // Range to generate mtd_upperbound and mtd_lowerbound values
#define VAL_TOLERANCE 100.0     // Minimum range for the magnitude
#define DIST_THRESHOLD 200.0    // Minimum threshold for the distance between 2 nodes
#define MAX_TIME_TAKEN 9
#define BASE_STATION_MSG 0
#define REQ_MSG 1
#define MTD_MSG 2
#define NBR_MSG 3
#define LAT_MSG 4
#define LONG_MSG 5
#define SHIFT_ROW 0
#define SHIFT_COL 1
#define DISP 1
#define pi 3.14159265358979323846

// Declare functions
float generate_random_float(unsigned int seed, float min, float max);
int count_valid_neighbours(int *p_neighbours);
float distance(float lat1, float lon1, float lat2, float lon2);
float deg2rad(float);
float rad2deg(float);

// Alert report struct
struct SReport { 
    double alert_time;
    double nbr_comm_time;
    float tolerance;
    float mtd_avg[5];
    int node_matched;
    int rank[5];
    float coord[2];
    int num_messages;
    float depth;
};

/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
/*::  This function simulate a seafloor seismic node           :*/
/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
void seafloor_sensor(int num_rows, int num_cols, float threshold, MPI_Comm world_comm, MPI_Comm nodes_comm) {
    float lat_lowerbound, lat_upperbound, long_lowerbound, long_upperbound;
    float depth_lowerbound, depth_upperbound;
    float mtd_lowerbound;
    float mtd_upperbound;
    float global_mtd_avg;
    int valid_num_neig = 0;
    int num_neig = 4; // Store number of neighbours and number of valid neighbours  

    // Variables to generate MPI Topology
    int i, my_rank, my_cart_rank, base_station_rank, size;
    int ndims = 2, reorder = 1, ierr = 0;
    int p_dims[ndims], real_coord[ndims], p_wrap_around[ndims], p_neighbours[num_neig];
    float p_coord[ndims];
    MPI_Comm cart_comm;

    // Flags
    int terminate_msg = 0;  // 0 - do not terminate, 1 - terminate

    // Default set 2 threads
    omp_set_num_threads(2);

    // Assign rows and cols to dims array
    p_dims[0] = num_rows;
    p_dims[1] = num_cols;

    // Get the base station rank number
    MPI_Comm_size(world_comm, &size);
    base_station_rank = size - 1;

    // Store the rank number from nodes_comm
    MPI_Comm_rank(nodes_comm, &my_rank);

    // Set periodic shift to false by initializing wrap_around array to 0
    for (i = 0; i < ndims; i++) {
        p_wrap_around[i] = 0;
    }

    // Create cartesian topology using nodes communicator   
    ierr = MPI_Cart_create(nodes_comm, ndims, p_dims, p_wrap_around, reorder, &cart_comm);
    if(ierr != 0) {
        printf("ERROR[%d] creating 2D CART\n", ierr);
    }
    
    MPI_Cart_coords(cart_comm, my_rank, ndims, real_coord); // use my rank to find my coordinates in the cartesian communicator group
    MPI_Cart_rank(cart_comm, real_coord, &my_cart_rank); // use my cartesian coordinates to find my cart rank in cartesian group
    
    // Get the adjacent neighbor's rank number (top, bottom, left, right)
    MPI_Cart_shift(cart_comm, SHIFT_ROW, DISP, &p_neighbours[0], &p_neighbours[1]);
	MPI_Cart_shift(cart_comm, SHIFT_COL, DISP, &p_neighbours[2], &p_neighbours[3]);

    // Generate random mtd_lowerbound and mtd_upperbound values for the sensor node
    mtd_lowerbound = threshold - VAL_RANGE, 
    mtd_upperbound = threshold + VAL_RANGE;
    global_mtd_avg = mtd_lowerbound; // Global magnitude values, default to mtd_lowerbound

    // Generate random latitude and longitude
    lat_lowerbound = 10.0;
    lat_upperbound = 15.0;
    long_lowerbound = 100.0;
    long_upperbound = 105.0;
    float latitude = generate_random_float(my_rank, lat_lowerbound, lat_upperbound);
    float longitude = generate_random_float(my_rank, long_lowerbound, long_upperbound);
    p_coord[0] = latitude;
    p_coord[1] = longitude;

    printf("Cart rank: %d. Cart Coord: (%d, %d). Lat: (%.2f). Long: (%.2f). .\n", my_cart_rank, real_coord[0], real_coord[1], p_coord[0], p_coord[1]);

    // Get the number of valid neighborhood nodes
    valid_num_neig = count_valid_neighbours(p_neighbours);
    sleep(my_rank);

    #pragma omp parallel sections 
    {
        // Listen to its neighbours and send its magnitude values if received a request
        #pragma omp section 
        {
            MPI_Status terminate_status;
            int req_msg;
            int base_flag = 0; // placeholder to indicate a message is received or not
            int local_terminate = 0; // local terminiate variable
            float local_mtd_avg = mtd_lowerbound; // local mtd_avg variable (default to mtd_lowerbound)
            float l_lat = p_coord[0]; // assign latitude to local variable
            float l_long = p_coord[1]; // assign longitude to local variable

            do {
                #pragma omp critical 

                // Listen to any request messages
                MPI_Iprobe(MPI_ANY_SOURCE, REQ_MSG, cart_comm, &base_flag, &terminate_status);

                // Send moving average when request is received
                local_mtd_avg = global_mtd_avg;
                if (base_flag) {
                    // Receive message from source
                    MPI_Recv(&req_msg, 1, MPI_INT, terminate_status.MPI_SOURCE, REQ_MSG, cart_comm, MPI_STATUS_IGNORE);

                    // Send magnitude, latitude and longitude to requester
                    MPI_Send(&local_mtd_avg, 1, MPI_FLOAT, terminate_status.MPI_SOURCE, MTD_MSG, cart_comm);
                    MPI_Send(&l_lat, 1, MPI_FLOAT, terminate_status.MPI_SOURCE, LAT_MSG, cart_comm);
                    MPI_Send(&l_long, 1, MPI_FLOAT, terminate_status.MPI_SOURCE, LONG_MSG, cart_comm);

                    base_flag = 0; // Reset flag
                }

                #pragma omp critical
                local_terminate = terminate_msg;

            } while (!local_terminate);
        }

        #pragma omp section 
        {            
            // Create report typestruct instance
            struct SReport report;
            MPI_Datatype report_type;
            MPI_Datatype type[9] = { MPI_DOUBLE, MPI_DOUBLE, MPI_FLOAT, MPI_FLOAT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT };
            int blocklen[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1};
            MPI_Aint disp[9];

            // Get the address of each member of the struct
            MPI_Get_address(&report.alert_time, &disp[0]);
            MPI_Get_address(&report.nbr_comm_time, &disp[1]);
            MPI_Get_address(&report.tolerance, &disp[2]);
            MPI_Get_address(&report.mtd_avg, &disp[3]);
            MPI_Get_address(&report.node_matched, &disp[4]);
            MPI_Get_address(&report.rank, &disp[5]);
            MPI_Get_address(&report.coord, &disp[6]);
            MPI_Get_address(&report.num_messages, &disp[7]);
            MPI_Get_address(&report.depth, &disp[8]);
            
            // Calculate the displacements
            for (i = 8; i > 0; i--) {
                disp[i] = disp[i] - disp[i - 1];
            }
	        disp[0] = 0;

            // Create MPI struct
            MPI_Type_create_struct(8, blocklen, disp, type, &report_type);
            MPI_Type_commit(&report_type);

            // Status to listen to terminating message from base station
            MPI_Status terminate_status;

            // Array to send and receive between nodes and their neighbours
            MPI_Request n_req[num_neig * 4];
            MPI_Status n_status[num_neig * 4];

            // Request and status for send operation between node and base station
            MPI_Request req;
            MPI_Status status;

            // Variables to store the time
            double curr_time;
            double time_taken;
            struct timespec timestamp;

            // Variables to deal with magnitude, latitude and longitude values
            int mtd_pointer = 0; // Pointer
            int matched_count; // Number of matched magnitudes
            float total;
            float local_mtd_avg = mtd_lowerbound; // Local magnitude average
            int mtd_window = 7; // Window size of magnitude values
            float *n_mtd_arr = calloc(mtd_window, sizeof(float)); // initialize the array

            // Initialize receiving arrays for magnitude, latitude and longitude
            float n_recv_vals[4] = { -1.0, -1.0, -1.0, -1.0 };
            float lat_recv_vals[4] = { -1.0, -1.0, -1.0, -1.0 };
            float long_recv_vals[4] = { -1.0, -1.0, -1.0, -1.0 };

            // Send and receive variables
            int alert_flag = 0;
            int req_flag = 0;
            int base_flag = 0;
            int base_station_msg;
            int local_terminate = 0; // local terminiate variable

            // General variables
            char *buffer;
            int buffer_size;
            int pos;
            int j, request = 1;

            do {
                timespec_get(&timestamp, TIME_UTC); // Get timestamp

                unsigned int seed = time(NULL);

                // Generate random depth value
                depth_lowerbound = 5.0;
                depth_upperbound = 7.0;
                float depth = generate_random_float(seed, depth_lowerbound, depth_upperbound);

                // Generate random magnitude value
                float mtd_val = generate_random_float(seed, mtd_lowerbound, mtd_upperbound);
                n_mtd_arr[mtd_pointer] = mtd_val;  // Push the magnitude value into the array
                mtd_pointer = (mtd_pointer + 1) % mtd_window; // Update pointer

                // Print message
                char buff[100];
                strftime(buff, sizeof(buff), "%D %T", gmtime(&timestamp.tv_sec));
                printf("Time %s UTC. Cart rank %d has random magnitude %lf and depth %lf.\n", buff, my_rank, mtd_val, depth);
                
                // Check if the array is filled up with values
                if (n_mtd_arr[mtd_pointer] != 0.0) {
                    total = 0.0;
                    for (j = 0; j < mtd_window; j++) {
                        total += n_mtd_arr[j];
                    }

                    // Calculate the new average magnitude
                    #pragma omp critical 
                    global_mtd_avg = total / mtd_window; // Update global magnitude average

                    #pragma omp critical 
                    local_mtd_avg = global_mtd_avg; // Update local magnitude average

                    // Check if the average magnitude exceeds the threshold
                    if (local_mtd_avg > threshold) {

                        // Non-blocking send request to all neighbors with tag REQ_MSG to get values
                        for (j = 0; j < num_neig; j++) {
                            MPI_Isend(&request, 1, MPI_INT, p_neighbours[j], REQ_MSG, cart_comm, &n_req[j]);
                        }

                        // Non-blocking receive mtd_avg from all neighbors with tag MTD_MSG
                        for (j = 0; j < num_neig; j++) {
                            n_recv_vals[j] = -1.0; // initialize to -1.0
                            MPI_Irecv(&n_recv_vals[j], 1, MPI_FLOAT, p_neighbours[j], MTD_MSG, cart_comm, &n_req[num_neig + j]);
                        }

                        // Non-blocking receive Lat from all neighbours with tag LAT_MSG
                        for (j = 0; j < num_neig; j++) {
                            lat_recv_vals[j] = -1.0; // initialize to -1.0
                            MPI_Irecv(&lat_recv_vals[j], 1, MPI_FLOAT, p_neighbours[j], LAT_MSG, cart_comm, &n_req[num_neig*2 + j]);
                        }

                        // Non-blocking receive Long from all neighbours with tag LONG_MSG
                        for (j = 0; j < num_neig; j++) {
                            long_recv_vals[j] = -1.0; // initialize to -1.0
                            MPI_Irecv(&long_recv_vals[j], 1, MPI_FLOAT, p_neighbours[j], LONG_MSG, cart_comm, &n_req[num_neig*3 + j]);
                        }

                        // Calculate time taken
                        curr_time = MPI_Wtime();
                        time_taken = MPI_Wtime() - curr_time;

                        // Keep checking for completed requests
                        do {
                            req_flag = 0; // Reset flag
                            MPI_Testall(num_neig * 4, n_req, &req_flag, n_status);  // Test requests
                            time_taken = MPI_Wtime() - curr_time; // calculate the time taken
                        } while (!req_flag && time_taken < MAX_TIME_TAKEN);

                        if (!req_flag) {
                            MPI_Cancel(n_req); // cancel all requests
                            printf("Cart rank %d cancel all requests due to time taken %.2f.\n", my_rank, time_taken);
                        } else {
                            req_flag = 0; // Reset flag
                            printf("Cart rank %d has average magnitude of %.3f. Received top (%.2f): %.2f, bottom (%.2f): %.2f, left (%.2f): %.2f, right (%.2f): %.2f.\n", my_rank, local_mtd_avg, 
                            distance(p_coord[0], p_coord[1], lat_recv_vals[0], long_recv_vals[0]), n_recv_vals[0], 
                            distance(p_coord[0], p_coord[1], lat_recv_vals[1], long_recv_vals[1]), n_recv_vals[1], 
                            distance(p_coord[0], p_coord[1], lat_recv_vals[2], long_recv_vals[2]), n_recv_vals[2], 
                            distance(p_coord[0], p_coord[1], lat_recv_vals[3], long_recv_vals[3]), n_recv_vals[3]);
                          
                            // Step 2: Compare values with neighbours
                            matched_count = 0;
                            for (j = 0; j < num_neig; j++) {    
                                if (n_recv_vals[j] != -1.0 && 
                                p_neighbours[j] != -2 && 
                                fabs(n_recv_vals[j] - local_mtd_avg) <= VAL_TOLERANCE && 
                                n_recv_vals[j] > threshold && // Check if the magnitude is greater than the threshold
                                distance(p_coord[0], p_coord[1], lat_recv_vals[j], long_recv_vals[j]) <= DIST_THRESHOLD) { // Check if the distance is less than the threshold
                                    matched_count += 1;
                                }
                            }

                            // If got more than or equal to 2 matches, send report to base station
                            if (matched_count > 1) {
                                // Construct the report
                                report.num_messages = valid_num_neig * 2;  // Set the number of messages
                                report.alert_time = (timestamp.tv_sec * 1e9 + timestamp.tv_nsec) * 1e-9; // Calculate alert time
                                report.nbr_comm_time = time_taken;  // Calculate communication time
                                report.tolerance = VAL_TOLERANCE;   // Set tolerance
                                report.mtd_avg[0] = local_mtd_avg;  // Set local magnitude average at first position
                                for (j = 0; j < num_neig; j++) {    // Set all the magnitude averages
                                    report.mtd_avg[j + 1] = n_recv_vals[j];
                                }
                                report.node_matched = matched_count; // Set number of nodes matched
                                report.rank[0] = my_rank;   // Set local rank at first position
                                for (j = 0; j < num_neig; j++) { // Set all the ranks
                                    report.rank[j + 1] = p_neighbours[j];
                                }

                                // Set latitude and longitude                                
                                report.coord[0] = p_coord[0];
                                report.coord[1] = p_coord[1];

                                // Set depth
                                report.depth = depth;

                                // Pack data into a buffer
                                pos = 0;
                                MPI_Pack_size(8, report_type, world_comm, &buffer_size);
                                buffer = (char *)malloc((unsigned) buffer_size);
                                MPI_Pack(&report.alert_time, 1, MPI_DOUBLE, buffer, buffer_size, &pos, world_comm);
                                MPI_Pack(&report.nbr_comm_time, 1, MPI_DOUBLE, buffer, buffer_size, &pos, world_comm);
                                MPI_Pack(&report.tolerance, 1, MPI_INT, buffer, buffer_size, &pos, world_comm);
                                MPI_Pack(&report.mtd_avg, 5, MPI_FLOAT, buffer, buffer_size, &pos, world_comm);
                                MPI_Pack(&report.node_matched, 1, MPI_INT, buffer, buffer_size, &pos, world_comm);
                                MPI_Pack(&report.rank, 5, MPI_INT, buffer, buffer_size, &pos, world_comm);
                                MPI_Pack(&report.coord, 2, MPI_FLOAT, buffer, buffer_size, &pos, world_comm);
                                MPI_Pack(&report.num_messages, 1, MPI_INT, buffer, buffer_size, &pos, world_comm);
                                MPI_Pack(&report.depth, 1, MPI_FLOAT, buffer, buffer_size, &pos, world_comm);

                                // Send packed data to base station
                                MPI_Isend(buffer, buffer_size, MPI_PACKED, base_station_rank, BASE_STATION_MSG, world_comm, &req);

                                // Calculate time
                                curr_time = MPI_Wtime();
                                time_taken = MPI_Wtime() - curr_time;
                                // Keep checking for completed requests
                                do {
                                    alert_flag = 0; // reset base_flag to false
                                    MPI_Test(&req, &alert_flag, &status);
                                    time_taken = MPI_Wtime() - curr_time; // calculate the time taken
                                } while ((! alert_flag) && time_taken < MAX_TIME_TAKEN);

                                if (!alert_flag) {
                                    MPI_Cancel(&req); // cancel all requests
                                    printf("Cart rank %d cancel all requests for sending report due to time taken %.2f.\n", my_rank, time_taken);                                    
                                } else {
                                    alert_flag = 0; // reset base_flag to false
                                    printf("Report sent to base station by Cart rank %d.\n", my_rank);
                                    printf("\tAlert time: %.2f, Number of matches: %d, Coord: (%.2f, %.2f)\n", report.alert_time, report.node_matched, report.coord[0], report.coord[1]);
                                }
                            }
                        }
                    }
                }
                
                // Keep listening to the base station
                MPI_Iprobe(base_station_rank, BASE_STATION_MSG, world_comm, &base_flag, &terminate_status);
                
                // Receive message from the base station
                if (base_flag) {
                    base_flag = 0; // reset base_flag to false
                    MPI_Recv(&base_station_msg, 1, MPI_INT, terminate_status.MPI_SOURCE, BASE_STATION_MSG, world_comm, MPI_STATUS_IGNORE);
                    printf("Cart rank %d is terminating as requested from base station.\n", my_rank);
                    #pragma omp critical 
                    terminate_msg = base_station_msg;
                }

                // Get terminate value
                #pragma omp critical
                local_terminate = terminate_msg;
                sleep(AVG_CYCLE);

            } while (!local_terminate);
            free(n_mtd_arr);
        }
    }

    MPI_Comm_free(&cart_comm);
}

/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
/*::  This function generate random float value between mtd_lowerbound and mtd_upperbound           :*/
/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
float generate_random_float(unsigned int seed, float min, float max) {
    srand(seed);
    float diff = max - min;
    float multiplier = ((float)rand()/(float)(RAND_MAX));
    return min + multiplier * diff;
}

/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
/*::  This function calculate the number of valid neighbours           :*/
/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
int count_valid_neighbours(int *p_neighbours) {
    int i = 0;
    int matched_count = 0;
    for (i = 0; i < 4; i++) {
        if (p_neighbours[i] != -2) {
            matched_count++;
        }
    }
    return matched_count;
}

/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
/*::  This function converts calculates the distance between 2 nodes            :*/
/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
float distance(float lat1, float lon1, float lat2, float lon2) {
float r = 6371;
  float dlat, dlon, dist;
  if ((lat1 == lat2) && (lon1 == lon2)) {
    return 0;
  }
  else {
    lat1 = deg2rad(lat1);
    lat2= deg2rad(lat2);
    lon1 = deg2rad(lon1);
    lon2 = deg2rad(lon2);

    dlat = lat2 - lat1;
    dlon = lon2 - lon1;

    dist = pow(sin(dlat/2), 2) + cos(lat1) * cos(lat1) * pow(sin(dlon/2), 2);

    dist = 2 * asin(sqrt(dist));
    dist = dist * r;
    return (dist);
  }
}

/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
/*::  This function converts decimal degrees to radians             :*/
/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
float deg2rad(float deg) {
  return (deg * pi / 180);
}

/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
/*::  This function converts radians to decimal degrees             :*/
/*:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*/
float rad2deg(float rad) {
  return (rad * 180 / pi);
}
