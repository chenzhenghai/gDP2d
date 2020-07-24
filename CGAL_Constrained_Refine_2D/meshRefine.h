#pragma once

typedef struct SyntheticParam
{
	int distribution;
	int seed;
	double min_input_angle;
	int numberofpoints;
	int numberofsegments;
} SyntheticParam;

typedef struct SyntheticIO
{
	int numberofpoints;
	int numberofsegments;
	int numberoftriangles;
	double *pointlist;
	int *segmentlist;
	int *trianglelist;
	int *neighborlist;
} SyntheticIO;

void refineInputByCGAL_Synthetic(
	char* input_path,
	double input_theta,
	double input_size,
	SyntheticParam input_sparam,
	char* output_path
);

void refineInputByCGAL_Real(
	char* input_path,
	char* input_file,
	double input_theta,
	double input_size,
	char* output_path
);