#include <iostream>
#include <sstream> 

#include "experiment.h"
#include "meshRefine.h"

bool readOutput_Synthetic(
	char* output_path,
	int * numberofpoints,
	int * numberoftriangles,
	int * numberofsegments,
	double * runtime,
	double input_theta,
	double input_size,
	SyntheticParam input_sparam
)
{
	std::ostringstream strs;
	strs << output_path << "d" << input_sparam.distribution << "_s" << input_sparam.seed
		<< "_p" << input_sparam.numberofpoints << "_c" << input_sparam.numberofsegments << "_t" << input_theta ;
	if (input_size > 0)
		strs << "_e" << input_size;
	strs << "_cgal" << "_ruppert";
	if (input_sparam.min_input_angle < 60.0)
		strs << "_with_minimum_input_angle_" << input_sparam.min_input_angle;
	std::string com_str = strs.str();
	char *com = new char[com_str.length() + 1];
	strcpy(com, com_str.c_str());

	FILE *fp;
	fp = fopen(com, "r");

	if (fp == NULL)
	{
		printf("Cannot find the output file\n");
		return false;
	}

	int np, nt, nc;
	np = nt = nc = 0;
	double mytime;
	int ln = 0;
	char buf[100];
	while (fgets(buf, 100, fp) != NULL) {
		int n;
		if (ln == 0)
			n = sscanf(buf, "Number of points = %d", &np);
		else if (ln == 1)
			n = sscanf(buf, "Number of triangles = %d", &nt);
		else if (ln == 2)
			n = sscanf(buf, "Number of segments = %d", &nc);
		else if (ln == 3)
			n = sscanf(buf, "Runtime = %lf", &mytime);
		if (!n)
			break;
		ln++;
	}

	if (numberofpoints != NULL)
		*numberofpoints = np;
	if (numberoftriangles != NULL)
		*numberoftriangles = nt;
	if (numberofsegments != NULL)
		*numberofsegments = nc;
	if (runtime != NULL)
		*runtime = mytime;

	fclose(fp);
	return true;
}

void experiment_cgal_Synthetic()
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
					if (readOutput_Synthetic(output_path, NULL, NULL, NULL, NULL, theta, size, input_sparam))
					{
						printf("Output found! Skip.\n\n");
						continue;
					}

					refineInputByCGAL_Synthetic(
						input_path,
						theta,
						size,
						input_sparam,
						output_path
					);

					printf("\n");
				}
			}
		}
	}
}

void experiment_cgal_real()
{
	char* input_path = "../../2d_data/input_realworld/";
	char* output_path = "../../2d_data/result_realworld/";

	char file_name[6][20] = {
		"1.2M",
		"3.2M",
		"4.3M",
		"5.6M",
		"8.4M",
		"9.4M"
	};

	double edge_size[6] = {
		0.12,
		0.12,
		0.11,
		0.1,
		0.12,
		0.16
	};

	double theta = 20;

	for (int i = 3; i < 6; i++)
	{
		refineInputByCGAL_Real(
			input_path,
			*(file_name + i),
			theta,
			edge_size[i],
			output_path
		);
	}
}