/*
This file contains function for simulating the base station and the satellite altimeter
*/
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

/* Constants */
#define BASE_INTERVAL 1
#define BALLOON_RANGE 3
#define BALLOON_INTERVAL 1
#define MAGNITUDE_DIFFERENCE 0.5
#define COORDINATE_THRESHOLD_DIFFERENCE 100
#define DATE_TIME "%a %F %T"
#define BALLOON_SIZE 5

/* Struct types */

struct SReport
{
    double report_time;
    double nbr_comm_time;
    float tolerance;
    float magnitude[5];
    int node_matched;
    int rank[5];
    float coord[2];
    int num_messages;
    float threshold;
    float depth;
};

struct SEvent
{
    int filled;
    int iteration;
    int iterate;
    char *match;
    struct SReport report;
    char *alt_time;
    float magnitude;
    double comm_time;
    float distance;
    float lat;
    float lon;
};

struct SPerformance
{
    double sim_time;
    int total_report;
    int total_match;
    int total_mismatch;
    double total_comm_time;
    double avg_comm_time;
    int **coord;
    int *total_report_per_node;
};

struct SBalloon
{
    struct timespec time;
    double report_time;
    double nbr_comm_time;
    float tolerance;
    float seafloor_sensor[5];
    int node_matched;
    int rank[5];
    float coord[2];
    int num_messages;
    float magnitude;
};

/* Function declarations */
void log_event(char *event_log_name, struct SEvent event);
void log_performance(char *event_log_name, int cart_size, struct SPerformance performance);
float generate_random_float(unsigned int seed, float min, float max);
float distance(float lat1, float lon1, float lat2, float lon2);
int exit_program(char* pFile);

void base_station(int num_rows, int num_cols, float threshold, int max_iteration, MPI_Comm world_comm)
{
    int terminate_msg = 0; // 0 - do not terminate, 1 - terminate

    char *event_log_name = "result_log.txt";
    char *sentinel_name = "sentinel.txt";

    int i, size;
    MPI_Comm_size(world_comm, &size);

    int cart_size = size - 1;
    printf("cart_size: %d\n", cart_size);

    // stores balloon sensor generated values
    struct SBalloon balloon[BALLOON_SIZE];
    int pointer = 0; // pointer to next empty position to the array

    struct SPerformance performance;
    struct timespec start;

    // get start time of base simulation
    timespec_get(&start, TIME_UTC);

    // initialize the array to store the total reports per node
    performance.total_report_per_node = calloc(cart_size, sizeof(int));

    // initialize the array to store the coordinate of reporting node
    performance.coord = (int **)malloc(cart_size * sizeof(int *));
    for (i = 0; i < cart_size; i++)
    {
        performance.coord[i] = (int *)malloc(2 * sizeof(int));
        
    }
    for (i = 0; i < cart_size; i++)
    {
        performance.coord[i][0] = (int)round(i / num_cols);
        performance.coord[i][1] = (int)round(i % num_cols);
    }


    #pragma omp parallel sections
        {
    /* one thread for receiving and sending messages from sensor nodes */
    #pragma omp section
        {
            int i, l_terminate_msg = 0;

            struct SEvent events[max_iteration * BALLOON_SIZE];
            struct SEvent event;
            struct SReport report;
            struct timespec alt_time, end, comm_end;
            
            double comm_time, time_taken;
            char alt_time_str[50];
            int match;

            MPI_Request send_request[cart_size];
            MPI_Status send_status[cart_size];

            // initialize values in event and performance
            event.iterate = 0;
            event.iteration = 0;
            performance.total_report = 0;
            performance.total_match = 0;
            performance.total_mismatch = 0;
            performance.total_comm_time = 0;

            do
            {
                int flag = 0, buffer_size, position = 0;
                char *buffer;

                MPI_Status probe_status;
                MPI_Iprobe(MPI_ANY_SOURCE, 0, world_comm, &flag, &probe_status);

                if (flag)
                {
                    flag = 0;
                    MPI_Get_count(&probe_status, MPI_PACKED, &buffer_size);
                    buffer = (char *)malloc((unsigned)buffer_size);
                    MPI_Recv(buffer, buffer_size, MPI_PACKED, probe_status.MPI_SOURCE, 0, world_comm, MPI_STATUS_IGNORE);

                    // end of communication time between node and base
                    timespec_get(&comm_end, TIME_UTC);
                    comm_time = (comm_end.tv_sec * 1e9 + comm_end.tv_nsec) * 1e-9;

                    // unpack the data into a buffer
                    MPI_Unpack(buffer, buffer_size, &position, &report.report_time, 1, MPI_DOUBLE, world_comm);
                    MPI_Unpack(buffer, buffer_size, &position, &report.nbr_comm_time, 1, MPI_DOUBLE, world_comm);
                    MPI_Unpack(buffer, buffer_size, &position, &report.tolerance, 1, MPI_INT, world_comm);
                    MPI_Unpack(buffer, buffer_size, &position, &report.magnitude, 5, MPI_FLOAT, world_comm);
                    MPI_Unpack(buffer, buffer_size, &position, &report.node_matched, 1, MPI_INT, world_comm);
                    MPI_Unpack(buffer, buffer_size, &position, &report.rank, 5, MPI_INT, world_comm);
                    MPI_Unpack(buffer, buffer_size, &position, &report.coord, 2, MPI_INT, world_comm);
                    MPI_Unpack(buffer, buffer_size, &position, &report.num_messages, 1, MPI_INT, world_comm);

                    

                    // update event comm_time
                    comm_time = comm_time - report.report_time;
                    event.comm_time = comm_time;

                    printf("Base Station received %d report:\n", report.rank[0]);
                    printf("\tReport time: %.2f, Rank: %d, Coord: (%.2f, %.2f)\n", report.report_time, report.rank[0], report.coord[0], report.coord[1]);

                    // update total number of reports received
                    performance.total_report += 1;

                    // update total number of reports received from reporting node
                    performance.total_report_per_node[report.rank[0]] += 1;

                    // update total communication time in performance
                    performance.total_comm_time += comm_time;

                    // compare the receive report with the balloon sensor readings
                    #pragma omp parallel
                    #pragma omp for

                    for (int i = 0; i < BALLOON_SIZE; i++)
                    {
                        float coor_distance, latitude, longitude;
                        latitude = balloon[i].coord[0];
                        longitude = balloon[i].coord[1];
                        coor_distance = distance(report.coord[0], report.coord[1], latitude, longitude);
                        alt_time = balloon[i].time;
                        strftime(alt_time_str, sizeof(alt_time_str), DATE_TIME, gmtime(&alt_time.tv_sec));
                        event.alt_time = alt_time_str;
                        report.threshold = threshold;

                        // if the distance and magnitude difference is within the threshold
                        if (coor_distance <= COORDINATE_THRESHOLD_DIFFERENCE && fabs(balloon[i].magnitude - report.magnitude[0]) <= MAGNITUDE_DIFFERENCE)
                        {
                            alt_time = balloon[i].time;
                            event.match = "Conclusive";
                            performance.total_match += 1;
                            event.lat = latitude;
                            event.lon = longitude;
                            event.magnitude = balloon[i].magnitude;
                            event.distance = coor_distance;
                            event.report = report;
                            match += 1;

                            #pragma omp critical
                            {
                                events[event.iteration] = event;
                            }
                        }
                        event.iteration++;
                        event.filled = 1;
                    }
                    // if none of the balloon seismic sensor readings match the reporting seafloor seismic sensor node
                    if (!match)
                    {
                        event.match = "Inconclusive";
                        performance.total_mismatch += 1;
                        event.report = report;
                        event.lat = 0;
                        event.lon = 0;
                        event.magnitude = 0;
                        event.distance = 0;
                        events[event.iteration] = event;
                    }
                    match = 0;
                }
                else
                {
                    event.filled = 0;
                }

                
                // if quit has been changed to 1 then break loop and stop simulation
                if (exit_program(sentinel_name) == 1)
                {
                    break;
                }

                // sleep for a specified amount of time before going to the next iteration
                sleep(BASE_INTERVAL);
                event.iterate++;

            } while (event.iterate < max_iteration);

            // set terminate flag to true
            #pragma omp critical
            {
                terminate_msg = 1;
                l_terminate_msg = terminate_msg;
            }

            // send termination message to all sensor nodes
            for (i = 0; i < cart_size; i++)
            {
                MPI_Isend(&l_terminate_msg, 1, MPI_INT, i, 0, world_comm, &send_request[i]);
            }
            // wait for the terminate message to send to all the sensor nodes
            MPI_Waitall(cart_size, send_request, send_status);

            // add events to log file
            for (i = 0; i < max_iteration * BALLOON_SIZE; i++)
            {
                if (events[i].filled == 1)
                {
                    log_event(event_log_name, events[i]);
                }
            }

            // calculate average communication time among reports
            performance.avg_comm_time = performance.total_comm_time / performance.total_report;

            // end of simulation time
            timespec_get(&end, TIME_UTC);
            time_taken = end.tv_sec - start.tv_sec;
            time_taken = (time_taken + (end.tv_nsec - start.tv_nsec) * 1e-9);
            performance.sim_time = time_taken;

            // generate performance event
            log_performance(event_log_name, cart_size, performance);
        }

        // balloon seismic thread using openmp
        #pragma omp section
        {
            int magntiude_upper = threshold + BALLOON_RANGE;
            float generated_magnitude;
            float coord[2];
            struct timespec alt_time;
            int l_terminate_msg;

            do
            {
                // read global terminate flag and store it in local terminate flag
                #pragma omp critical
                
                l_terminate_msg = terminate_msg;
                char buff[100];

                unsigned int rand_seed = (unsigned int)time(NULL);
                srand(rand_seed);

                // Generate random latitude and longitude
                float latitude = generate_random_float(rand_seed, 10.0, 15.0);
                float longitude = generate_random_float(rand_seed, 100.0, 105.0);
                coord[0] = latitude;
                coord[1] = longitude;

                // randomly generate magnitude
                float scale = ((float)rand() / (float)(RAND_MAX)); /* [0, 1.0] */
                generated_magnitude = threshold + scale * (magntiude_upper - threshold);

                // get current time
                timespec_get(&alt_time, TIME_UTC);
                strftime(buff, sizeof(buff), "%D %T", gmtime(&alt_time.tv_sec));

                // update balloon
                #pragma omp critical
                {
                    balloon[pointer].time = alt_time;
                    balloon[pointer].coord[0] = coord[0];
                    balloon[pointer].coord[1] = coord[1];
                    balloon[pointer].magnitude = generated_magnitude;
                }
                printf("Balloon sensor node has magnitude of %lf at time %s UTC.\n", generated_magnitude, buff);
                pointer += 1;

                // if the array is full, point to the first element
                if (pointer >= BALLOON_SIZE)
                {
                    pointer = 0;
                }

                sleep(BALLOON_INTERVAL);

            } while (l_terminate_msg != 1);
        }
    }

    free(performance.coord);
    free(performance.total_report_per_node);
    fflush(stdout);
}

int exit_program(char* pFile) {
    int sentinel_value;
    FILE *pSentinelFile = fopen(pFile, "r");
    fscanf(pSentinelFile, "%d", &sentinel_value);
    fclose(pSentinelFile);
    return sentinel_value;
}

void log_event(char *event_log_name, struct SEvent event)
{
    struct SReport report = event.report;
    // get log time
    struct timespec log_time;
    timespec_get(&log_time, TIME_UTC);
    
    const time_t report_time_convert = report.report_time;
    char buff[100];

    // open log file to append new information
    FILE *pFile = fopen(event_log_name, "a");

    //print information into file
    fprintf(pFile, "------------------------------------------------------------------------------------------------\n");
    fprintf(pFile, "Iteration: %d\n", event.iterate + 1);
    // change log time to date time string
    strftime(buff, sizeof buff, DATE_TIME, gmtime(&log_time.tv_sec));
    fprintf(pFile, "Logged time:\t\t\t\t%s\n", buff);
    // change report time to date time string
    strftime(buff, sizeof buff, DATE_TIME, gmtime(&report_time_convert));
    fprintf(pFile, "Alert reported time:\t\t%s\n", buff);
    fprintf(pFile, "Alert type: %s\n\n", event.match);

    // information from the reporting node
    fprintf(pFile, "Reporting Node\t\tSeismic Coord\t\t\t\t\t\t\tMagnitude\t\t\t\t\tDepth\n");
    fprintf(pFile, "%d\t\t\t\t\t(%.2f, %.2f)\t\t\t\t\t\t\t%.2lf\t\t\t\t\t\t%.2lf\n\n", report.rank[0], report.coord[0], report.coord[1], report.magnitude[0], report.depth);

    // Details of reporting nodes's neigbouring ndoes
    if (event.magnitude) {
        fprintf(pFile, "Adjacent Nodes\t\tSeismic Coord\t\tDiff(Coord, km)\t\tMagnitude\t\tDiff(Mag)\n");
    } else {
        fprintf(pFile, "Adjacent Nodes\t\tSeismic Coord\t\t\t\t\t\t\tMagnitude\t\t\n");
    }
    int i, row_disp, col_disp;
    for (i = 1; i < 5; i++)
    {
        // reset row and column displacement
        row_disp = 0;
        col_disp = 0;
        if (report.rank[i] != -2)
        { // if neighbour is non existant, then rank will be -2
            // switch to find the displacement for neighbour coordinates
            switch (i)
            {
            case 1:
            { // top
                row_disp = -1;
                break;
            }
            case 2:
            { // bottom
                row_disp = 1;
                break;
            }
            case 3:
            { // left
                col_disp = -1;
                break;
            }
            case 4:
            { // right
                col_disp = 1;
                break;
            }
            }
            float nbr_coord_row = report.coord[0] + row_disp;
            float nbr_coord_col = report.coord[1] + col_disp;
            if (event.magnitude) {
                fprintf(pFile, "%d\t\t\t\t\t(%.2f, %.2f)\t\t%.2lf\t\t\t\t%.3lf\t\t\t\t%.2f\n", report.rank[i], nbr_coord_row, nbr_coord_col, distance(nbr_coord_row, nbr_coord_col,event.lat, event.lon),report.magnitude[i], fabs(event.magnitude-report.magnitude[i]));
            } else {
                fprintf(pFile, "%d\t\t\t\t\t(%.2f, %.2f)\t\t\t\t\t\t\t%.2lf\t\t\t\n", report.rank[i], nbr_coord_row, nbr_coord_col, report.magnitude[i]);
            }
            }
    }
    fprintf(pFile, "\n");

    if (event.magnitude)
    {
        // printing balloon seismic information if it is a conclusive event
        fprintf(pFile, "Balloon seismic reporting time: %s\n", event.alt_time);
        fprintf(pFile, "Balloon seismic reporting Coord.: (%.2f, %.2f)\n", event.lat, event.lon);
        fprintf(pFile, "Balloon seismic reporing Coord Diff. with Reporting Node (km): %.2lf\n", event.distance);
        fprintf(pFile, "Balloon seismic reporting magnitude: %.3lf\n", event.magnitude);
        fprintf(pFile, "Balloon seismic reporting Magnitude Diff. with Reporting Node: %.2f\n\n", fabs(event.magnitude-report.magnitude[0]));
    }

    // Additional Information
    int msg = 1;
    fprintf(pFile, "Total Communication time for Reporting node (seconds): %lf\n", event.comm_time);
    fprintf(pFile, "\tCommunication time between its neighbours: %lf\n", report.nbr_comm_time);
    fprintf(pFile, "Total messages sent and received by Reporting node for this report: %d\n", msg + report.num_messages);
    fprintf(pFile, "\tMessages to base station: %d\n", msg);
    fprintf(pFile, "\tMessages between neighbours: %d\n", report.num_messages);
    fprintf(pFile, "Number of adjacent matches to Reporting node: %d\n", report.node_matched);
    fprintf(pFile, "Coordinate difference threshold (km): %.3f\n", report.tolerance);
    fprintf(pFile, "Magnitude difference threshold:: %.3f\n", MAGNITUDE_DIFFERENCE);
    fprintf(pFile, "Earthquake magnitude threshold:: %.2f\n", report.threshold);
    fprintf(pFile, "------------------------------------------------------------------------------------------------\n");
    
    // Close file
    fclose(pFile);
}

void log_performance(char *event_log_name, int cart_size, struct SPerformance performance)
{
    int i;
    // open log file to append performance
    FILE *pFile = fopen(event_log_name, "a");

    // write performance
    fprintf(pFile, "------------------------------------------------------------------------------------------------\n\n");
    fprintf(pFile, "Performance\n\n");
    fprintf(pFile, "Average communication time: %lf\n", performance.avg_comm_time);
    fprintf(pFile, "Total simulation time: %lf\n", performance.sim_time);
    fprintf(pFile, "Total number of events: %d\n", performance.total_report);
    fprintf(pFile, "\tConclusive events: \t%d\n", performance.total_report-performance.total_mismatch);
    fprintf(pFile, "\tInconclusive events: \t%d\n", performance.total_mismatch);
    fprintf(pFile, "Total number of reports per node:\n");
    fprintf(pFile, "\tCoordinate\tNumber of reports\n");
    for (i = 0; i < cart_size; i++)
    {
        fprintf(pFile, "\t(%d, %d)\t\t%d\n", performance.coord[i][0], performance.coord[i][1], performance.total_report_per_node[i]);
    }
    fprintf(pFile, "\n------------------------------------------------------------------------------------------------\n\n");

    // close the log file
    fclose(pFile);
}
