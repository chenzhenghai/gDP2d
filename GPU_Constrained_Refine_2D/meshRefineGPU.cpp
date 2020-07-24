#include <stdio.h>
#include <windows.h>
#include <iostream>
#include <sstream>  
#include "meshRefine.h"
#include "mesh.h"
#include "cudaRefine.h"
#include "Viewer.h"

bool saveOutput_Synthetic(
	triangulateio *output,
	char* output_path,
	double total_time,
	double input_theta,
	double input_size,
	int enc_mode,
	int run_mode,
	SyntheticParam input_sparam
)
{
	std::ostringstream strs;
	strs << output_path << "d" << input_sparam.distribution << "_s" << input_sparam.seed
		<< "_p" << input_sparam.numberofpoints << "_c" << input_sparam.numberofsegments << "_t" << input_theta ;
	if (input_size > 0)
		strs << "_e" << input_size;
	strs << "_gpu";
	if (enc_mode)
		strs << "_ruppert";
	if (run_mode)
		strs << "_new";
	if (input_sparam.min_input_angle < 60.0)
		strs << "_with_minimum_input_angle_" << input_sparam.min_input_angle;
	std::string com_str = strs.str();
	char *com = new char[com_str.length() + 1];
	strcpy(com, com_str.c_str());

	FILE *fp;
	fp = fopen(com, "w");

	fprintf(fp, "Number of points = %d\n", output->numberofpoints);
	fprintf(fp, "Number of triangles = %d\n", output->numberoftriangles);
	fprintf(fp, "Number of segments = %d\n", output->numberofsegments);
	fprintf(fp, "Runtime = %lf\n", total_time);
	fclose(fp);

	delete[] com;
}

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
)
{
	numberofbadtris_theta = 0;
	numberofbadtris_size = 0;
	numberofbadtris_total = 0;
	total_bad_angle_area = 0;
	total_bad_size_area = 0;
	total_bad_area = 0;
	total_area = 0;

	double dx[3], dy[3], edgelength[3];
	double p[3][2];

	double goodAngle = cos(input_theta * PI / 180.0);
	goodAngle *= goodAngle;
	double sizebound = input_size*input_size;

	bool isbadangle, isbadsize;
	double area, halfperimeter, a, b, c;
	for (int id = 0; id < output->numberoftriangles; id++)
	{
		isbadangle = false;
		isbadsize = false;

		for (int i = 0; i < 3; i++)
		{
			int vertex = output->trianglelist[3 * id + i];
			p[i][0] = output->pointlist[2 * vertex + 0];
			p[i][1] = output->pointlist[2 * vertex + 1];
		}

		for (int i = 0; i < 3; i++)
		{
			int j = (i + 1) % 3;
			int k = (i + 2) % 3;
			dx[i] = p[j][0] - p[k][0];
			dy[i] = p[j][1] - p[k][1];
			edgelength[i] = dx[i] * dx[i] + dy[i] * dy[i];
			if (input_size > 0.0 && edgelength[i] > sizebound)
				isbadsize = true;
		}

		for (int i = 0; i < 3; i++)
		{
			int  j = (i + 1) % 3;
			int  k = (i + 2) % 3;
			double dotproduct = dx[j] * dx[k] + dy[j] * dy[k];
			double cossquare = dotproduct * dotproduct / (edgelength[j] * edgelength[k]);
			if (cossquare > goodAngle)
				isbadangle = true;
		}

		if (isbadangle)
			numberofbadtris_theta++;
		if (isbadsize)
			numberofbadtris_size++;
		if (isbadangle || isbadsize)
			numberofbadtris_total++;

		a = sqrt(edgelength[0]); b = sqrt(edgelength[1]); c = sqrt(edgelength[2]);
		halfperimeter = (a + b + c) / 2;
		area = sqrt(halfperimeter*(halfperimeter - a)*(halfperimeter - b)*(halfperimeter - c));
		total_area += area;
		if (isbadangle || isbadsize)
			total_bad_area += area;
		if (isbadangle)
			total_bad_angle_area += area;
		if (isbadsize)
			total_bad_size_area += area;
	}
}

bool saveOutput_Real(
	triangulateio *output,
	char* output_path,
	double total_time,
	char* input_file,
	double input_theta,
	double input_size,
	int enc_mode,
	int run_mode
)
{
	// Statistic
	std::ostringstream strs;
	strs << output_path << input_file << "_t" << input_theta;
	if (input_size > 0)
		strs << "_e" << input_size;
	strs << "_gpu";
	if (enc_mode)
		strs << "_ruppert";
	if (run_mode)
		strs << "_new";
	std::string com_str = strs.str();
	char *com = new char[com_str.length() + 1];
	strcpy(com, com_str.c_str());

	FILE *fp;
	fp = fopen(com, "w");

	int numberofbadtris_theta, numberofbadtris_size, numberofbadtris_total;
	double total_bad_angle_area, total_bad_size_area, total_bad_area, total_area;
	elements_statistic(output, numberofbadtris_theta, numberofbadtris_size, numberofbadtris_total,
		total_bad_angle_area, total_bad_size_area, total_bad_area, total_area, input_theta, input_size);

	fprintf(fp, "Number of points = %d\n", output->numberofpoints);
	fprintf(fp, "Number of triangles = %d\n", output->numberoftriangles);
	fprintf(fp, "Number of segments = %d\n", output->numberofsegments);
	fprintf(fp, "Runtime = %lf\n", total_time);
	fprintf(fp, "Number of bad triangles = %d\n", numberofbadtris_total);
	fprintf(fp, "Number of bad triangles (angle) = %d\n", numberofbadtris_theta);
	fprintf(fp, "Number of bad triangles (size) = %d\n", numberofbadtris_size);
	fprintf(fp, "Bad triangle ratio = %.3f\n", numberofbadtris_total*1.0 / output->numberoftriangles);
	fprintf(fp, "Total area = %.3f\n", total_area);
	fprintf(fp, "Total bad area = %.3f\n", total_bad_area);
	fprintf(fp, "Total bad area (angle) = %.3f\n", total_bad_angle_area);
	fprintf(fp, "Total bad area (size) = %.3f\n", total_bad_size_area);
	fprintf(fp, "Bad area ratio = %.3f\n", total_bad_area / total_area);
	fclose(fp);

	delete[] com;

	// Mesh
	strs << ".ply";
	std::string ply_str = strs.str();
	com = new char[ply_str.length() + 1];
	strcpy(com, ply_str.c_str());
	fp = fopen(com, "w");

	fprintf(fp, "ply\n");
	fprintf(fp, "format ascii 1.0\n");
	fprintf(fp, "element vertex %d\n", output->numberofpoints);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "element edge %d\n", output->numberofsegments);
	fprintf(fp, "property int vertex1\n");
	fprintf(fp, "property int vertex2\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "end_header\n");

	for (int i = 0; i < output->numberofpoints; i++)
	{
		fprintf(fp, "%lf %lf 0 255 0 0\n", output->pointlist[2 * i + 0], output->pointlist[2 * i + 1]);
	}

	for (int i = 0; i < output->numberofsegments; i++)
	{
		fprintf(fp, "%d %d 255 0 0\n", output->segmentlist[2 * i + 0], output->segmentlist[2 * i + 1]);
	}

	fclose(fp);
	
	delete[] com;
}

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
)
{
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

	printf("Distribution %d, Seed = %d, Number of points = %d, Number of segments = %d\n",
		input_sparam.distribution, input_sparam.seed,
		input_sparam.numberofpoints, input_sparam.numberofsegments);
	printf("Minimum allowable angle = %lf, Maximum edge size = %lf, Encroachment mode = %s, Run mode = %s\n",
		input_theta, input_size, enc_mode ? "Ruppert" : "Chew", run_mode ? "New" : "Old");

	printf("Preparing for synthetic input: ");
	triangulateio triInput, triOutput;
	memset(&triInput, 0, sizeof(triangulateio));
	if (!readInput_Synthetic(&triInput, input_path, input_sparam))
	{
		GenerateRandomInput(input_path, input_sparam.numberofpoints, input_sparam.numberofsegments,
			input_sparam.seed, input_sparam.distribution, &triInput, input_sparam.min_input_angle, true);
		saveInput_Synthetic(&triInput, input_path, input_sparam);
	}

	printf("Refine mesh by GPU...\n");
	clock_t tv[2];
	tv[0] = clock();

	InsertPolicy insertpolicy = Offcenter;
	DeletePolicy deletepolicy = Connected;

	GPU_Refine_Quality(&triInput,&triOutput,input_theta,input_size,insertpolicy,deletepolicy,enc_mode, run_mode,
		filter_mode, unify_mode, -1, NULL, NULL);

	tv[1] = clock();
	double total_time = (REAL)(tv[1] - tv[0]);
	printf("Total time = %.3f ms\n", total_time);
	printf("Number of points = %d\nnumber of triangles = %d\nnumber of segments = %d\n",
		triOutput.numberofpoints, triOutput.numberoftriangles, triOutput.numberofsegments);
	printf("\n");

	if (output_path != NULL)
	{
		printf("Output statistic...\n");
		saveOutput_Synthetic(&triOutput, output_path, total_time, input_theta, input_size, enc_mode, run_mode, input_sparam);
	}

	clear_Triangulateio(&triInput);
	clear_Triangulateio(&triOutput);
}

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
)
{
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
	printf("Minimum allowable angle = %lf, Maximum edge size = %lf, Encroachment mode = %s, Run mode = %s\n",
		input_theta, input_size, enc_mode ? "Ruppert" : "Chew", run_mode ? "New" : "Old");

	printf("Preparing for real-world input %s%s\n", input_path, input_file);
	triangulateio triInput, triOutput;
	memset(&triInput, 0, sizeof(triangulateio));
	if (!readInput_Real(&triInput, input_path, input_file))
	{
		exit(0);
	}
	printf("Input: number of points = %d, number of triangles = %d, number of segments = %d\n",
		triInput.numberofpoints, triInput.numberoftriangles, triInput.numberofsegments);

	//return;

	printf("Refine mesh by GPU...\n");
	clock_t tv[2];
	tv[0] = clock();

	InsertPolicy insertpolicy = Offcenter;
	DeletePolicy deletepolicy = Connected;

	GPU_Refine_Quality(&triInput, &triOutput, input_theta, input_size, insertpolicy, deletepolicy, enc_mode, run_mode, 
		filter_mode, unify_mode, -1, NULL, NULL);

	tv[1] = clock();
	double total_time = (REAL)(tv[1] - tv[0]);
	printf("Total time = %.3f ms\n", total_time);
	printf("Number of points = %d\nnumber of triangles = %d\nnumber of segments = %d\n",
		triOutput.numberofpoints, triOutput.numberoftriangles, triOutput.numberofsegments);
	printf("\n");

	if (output_path != NULL)
	{
		printf("Output statistic...\n");
		saveOutput_Real(&triOutput, output_path, total_time, input_file, input_theta, input_size, enc_mode, run_mode);
	}

	//char* title = "";
	//drawTriangulation(0, &title, &triOutput);

	clear_Triangulateio(&triInput);
	clear_Triangulateio(&triOutput);
}