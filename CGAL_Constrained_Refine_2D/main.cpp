#include <stdio.h>
#include "meshRefine.h"
#include "experiment.h"

int main()
{
	//experiment_cgal_Synthetic();

	experiment_cgal_real();
	return 0;

	// Synthetic
	if (false)
	{
		char * input_path = "../../2d_data/input/";
		char * output_path = "../../2d_data/result/";

		double input_theta = 15;
		double input_size = 250;
		bool output = true;

		SyntheticParam input_sparam;
		input_sparam.distribution = 0;
		input_sparam.seed = 0;
		input_sparam.min_input_angle = 5.0;
		input_sparam.numberofpoints = 100000;
		input_sparam.numberofsegments = input_sparam.numberofpoints * 0.1;

		printf("Refine synthetic data by CGAL...\n");
		refineInputByCGAL_Synthetic(
			input_path,
			input_theta,
			input_size,
			input_sparam,
			output ? output_path : NULL
		);
	}

	// Real-world
	if (true)
	{
		char* input_path = "../../2d_data/input_realworld/";
		char* output_path = "../../2d_data/result_realworld/";
		char* input_file = "9.4M";

		double input_theta = 25;
		double input_size = 0;
		bool output = true;

		refineInputByCGAL_Real(
			input_path,
			input_file,
			input_theta,
			input_size,
			output_path
		);
	}

	return 0;
}