/******************************************************************************
* FILE: mergesort.cpp
* DESCRIPTION:  
*   Parallelized merge sort algorithm using MPI
* AUTHOR: Harini Kumar
* LAST REVISED: 11/02/23
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int main(int argc, char **argv)
{
    int NUM_VALS = atoi(argv[1]);
    int num_threads;

    cout << NUM_VALS << " " << num_threads << endl;
}