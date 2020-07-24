#include <stdio.h>
#include "meshRefine.h"

void experiment_gpu(
	int enc_mode,
	int run_mode,
	int filter_mode,
	int unify_mode
)
{
	char* input_path = "../../2d_data/input/";
	char* output_path = "../../2d_data/result/";

	double min_input_angle = 5.0;
	int numOfPoints, distribution, seed, numOfSegments;
	seed = 0;
	double theta;
	double size = 0;
	for (theta = 15; theta <= 25; theta += 5)
	{
		for (distribution = 0; distribution <= 3; distribution++)
		{
			for (numOfPoints = 100000; numOfPoints <= 100000; numOfPoints += 10000)
			{
				for (numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5; numOfSegments += numOfPoints*0.1)
				{
					SyntheticParam input_sparam;
					input_sparam.distribution = distribution;
					input_sparam.seed = seed;
					input_sparam.min_input_angle = min_input_angle;
					input_sparam.numberofpoints = numOfPoints;
					input_sparam.numberofsegments = numOfSegments;

					refineInputByGPU_Synthetic(
						input_path,
						theta,
						size,
						enc_mode,
						run_mode,
						filter_mode,
						unify_mode,
						input_sparam,
						output_path
					);
				}
			}
		}
	}
}