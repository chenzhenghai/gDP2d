#pragma once
#include <time.h>
#include "triangle.h"

void clear_Triangulateio(
	triangulateio* io
);

void elements_statistic(
	triangulateio *output,
	int &numberofbadtris_theta,
	int &numberofbadtris_size,
	int &numberofbadtris_total,
	double &total_bad_angle_area,
	double &total_bad_size_area,
	double &total_bad_area,
	double &total_area,
	double input_theta,
	double input_size
);

// Routines for synthetic data
typedef struct SyntheticParam
{
	int distribution;
	int seed;
	double min_input_angle;
	int numberofpoints;
	int numberofsegments;
} SyntheticParam;

bool readInput_Synthetic(
	triangulateio *input,
	char* input_path,
	SyntheticParam input_sparam
);

bool saveInput_Synthetic(
	triangulateio *input,
	char* input_path,
	SyntheticParam input_sparam
);

void refineInputByTriangle_Synthetic(
	char* input_path,
	double input_theta,
	double input_size,
	int input_mode,
	SyntheticParam input_sparam,
	char* output_path
);

void refineInputByGPU_Synthetic(
	char* input_path,
	double input_theta,
	double input_size,
	int enc_mode,
	int run_mode,
	int filter_mode,
	int unify_mode,
	SyntheticParam input_sparam,
	char* output_path
);

// Routines for real-world data

bool readInput_Real(
	triangulateio *input,
	char* input_path,
	char* input_file
);

void refineInputByTriangle_Real(
	char* input_path,
	char* input_file,
	double input_theta,
	double input_size,
	int input_mode,
	char* output_path
);

void refineInputByGPU_Real(
	char* input_path,
	char* input_file,
	double input_theta,
	double input_size,
	int enc_mode,
	int run_mode,
	int filter_mode,
	int unify_mode,
	char* output_path
);