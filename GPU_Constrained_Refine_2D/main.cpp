#include <stdio.h>
#include <windows.h>
#include <time.h>
#include "meshRefine.h"
#include "Experiment.h"

int
main(int argc, char** argv)
{
	// Synthetic
	if(false)
	{
		char * input_path = "input/";
		char * output_path = "result/";

		double input_theta = 25;
		double input_size = 0;

		int enc_mode = 1; // 1: Ruppert, 0: Chew
		int run_mode = 1; // 1: New, 0: Old
		int filter_mode = 1; // 1: Fast filtering, 0: No filtering
		int unify_mode = 1; // 1: Unify, 0: No unify
		bool output = false;

		SyntheticParam input_sparam;
		input_sparam.distribution = 0;
		input_sparam.seed = 0;
		input_sparam.min_input_angle = 5.0;
		input_sparam.numberofpoints = 100000;
		input_sparam.numberofsegments = input_sparam.numberofpoints * 0.5;

		//printf("Refine synthetic data by Triangle...\n");
		//refineInputByTriangle_Synthetic(
		//	input_path,
		//	input_theta,
		//	input_size,
		//	enc_mode,
		//	input_sparam,
		//	output ? output_path : NULL
		//);

		printf("Refine synthetic data by GPU...\n");
		refineInputByGPU_Synthetic(
			input_path,
			input_theta,
			input_size,
			enc_mode,
			run_mode,
			filter_mode,
			unify_mode,
			input_sparam,
			output ? output_path : NULL
		);
	}

	// Real-world
	if(true)
	{
		char* input_path = "input_realworld/";
		char* output_path = "result_realworld/";
		char* input_file = "3.2M";

		double input_theta = 20;
		double input_size = 0.12;
		bool output = false;

		int enc_mode = 1; // 1: Ruppert, 0: Chew
		int run_mode = 1; // 1: New, 0: Old
		int filter_mode = 1; // 1: Fast filtering, 0: No filtering
		int unify_mode = 1; // 1: Unify, 0: No unify

		refineInputByGPU_Real(
			input_path,
			input_file,
			input_theta,
			input_size,
			enc_mode,
			run_mode,
			filter_mode,
			unify_mode,
			output ? output_path : NULL
		);

		//refineInputByTriangle_Real(
		//	input_path,
		//	input_file,
		//	input_theta,
		//	input_size,
		//	enc_mode,
		//	output? output_path : NULL
		//);
	}

	return 0;
}