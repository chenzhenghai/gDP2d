#include <stdio.h>
#include <iostream>
#include <sstream>

bool read_synthetic_output(char * filename, int * numberofpoints, int * numberoftriangles, int * numberofsegments, double * runtime)
{
	FILE *fp;
	fp = fopen(filename, "r");

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

void experiment_statistic()
{
	FILE *fp;
	fp = fopen("../../2d_data/result/auto", "w");
	int numOfPoints, distribution, seed, numOfSegments;
	seed = 0;
	double theta = 15;
	for (distribution = 0; distribution <= 3; distribution++)
	{
		for (numOfPoints = 100000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for (numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5; numOfSegments += numOfPoints*0.1)
			{

				printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d\n",
					numOfPoints, numOfSegments, distribution);

				int r_p[3], r_t[3], r_c[3];
				double runtime[3];

				std::ostringstream strs0;
				strs0 << "../../2d_data/result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_cpu"
					<< "_with_minimum_input_angle_5";
				std::string fn0 = strs0.str();
				char *com0 = new char[fn0.length() + 1];
				strcpy(com0, fn0.c_str());

				if (!read_synthetic_output(com0, &r_p[0], &r_t[0], &r_c[0], &runtime[0]))
					continue;

				std::ostringstream strs1;
				strs1 << "../../2d_data/result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_gpu_new"
					<< "_with_minimum_input_angle_5";
				std::string fn1 = strs1.str();
				char *com1 = new char[fn1.length() + 1];
				strcpy(com1, fn1.c_str());

				if (!read_synthetic_output(com1, &r_p[1], &r_t[1], &r_c[1], &runtime[1]))
					continue;

				std::ostringstream strs2;
				strs2 << "../../2d_data/result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_gpu"
					<< "_with_minimum_input_angle_5";
				std::string fn2 = strs2.str();
				char *com2 = new char[fn2.length() + 1];
				strcpy(com2, fn2.c_str());

				if (!read_synthetic_output(com2, &r_p[2], &r_t[2], &r_c[2], &runtime[2]))
					continue;

				fprintf(fp, "%d,%d,%d,%d,%lf,%d,%d,%d,%d,%lf,%d,%d,%d,%lf,%d,%d,%d,%lf\n",
					distribution, seed, numOfPoints, numOfSegments, theta, 0,
					r_p[0], r_t[0], r_c[0], runtime[0],
					r_p[1], r_t[1], r_c[1], runtime[1],
					r_p[2], r_t[2], r_c[2], runtime[2]);

			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

void experiment_statistic_ruppert()
{
	FILE *fp;
	fp = fopen("../../2d_data/result/auto_ruppert", "w");
	int numOfPoints,distribution,seed,numOfSegments;
	seed = 0;
	double theta = 15;
	for(distribution = 0; distribution <= 3; distribution++)
	{
		for(numOfPoints = 100000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
			{
				printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d\n",
					numOfPoints, numOfSegments, distribution);

				int r_p[4],r_t[4],r_c[4];
				r_p[0] = r_p[1] = r_p[2] = r_p[3] =
					r_t[0] = r_t[1] = r_t[2] = r_t[3] =
					r_c[0] = r_c[1] = r_c[2] = r_c[3] = 0;
				double runtime[4];
				runtime[0] = runtime[1] = runtime[2] = runtime[3] = 0;

				std::ostringstream strs0;
				strs0 << "../../2d_data/result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_cpu_ruppert"
					<< "_with_minimum_input_angle_5";
				std::string fn0 = strs0.str();
				char *com0 = new char[fn0.length() + 1];
				strcpy(com0, fn0.c_str());

				if(!read_synthetic_output(com0,&r_p[0], &r_t[0], &r_c[0], &runtime[0]))
				{
				}

				std::ostringstream strs1;
				strs1 << "../../2d_data/result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_cgal_ruppert"
					<< "_with_minimum_input_angle_5";
				std::string fn1 = strs1.str();
				char *com1 = new char[fn1.length() + 1];
				strcpy(com1, fn1.c_str());

				if (!read_synthetic_output(com1, &r_p[1], &r_t[1], &r_c[1], &runtime[1]))
				{
				}

				std::ostringstream strs2;
				strs2 << "../../2d_data/result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_gpu_ruppert_new"
					<< "_with_minimum_input_angle_5";
				std::string fn2 = strs2.str();
				char *com2 = new char[fn2.length() + 1];
				strcpy(com2, fn2.c_str());

				if (!read_synthetic_output(com2, &r_p[2], &r_t[2], &r_c[2], &runtime[2]))
				{
				}

				std::ostringstream strs3;
				strs3 << "../../2d_data/result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_gpu_ruppert"
					<< "_with_minimum_input_angle_5";
				std::string fn3 = strs3.str();
				char *com3 = new char[fn3.length() + 1];
				strcpy(com3, fn3.c_str());
				
				if(!read_synthetic_output(com3,&r_p[3], &r_t[3], &r_c[3], &runtime[3]))
				{
				}

				fprintf(fp, "%d,%d,%d,%d,%lf,%d,%d,%d,%d,%lf,%d,%d,%d,%lf,%d,%d,%d,%lf,%d,%d,%d,%lf\n",
					distribution, seed, numOfPoints, numOfSegments, theta, 0,
					r_p[0], r_t[0], r_c[0], runtime[0],
					r_p[1], r_t[1], r_c[1], runtime[1],
					r_p[2], r_t[2], r_c[2], runtime[2],
					r_p[3], r_t[3], r_c[3], runtime[3]
				);

			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}
