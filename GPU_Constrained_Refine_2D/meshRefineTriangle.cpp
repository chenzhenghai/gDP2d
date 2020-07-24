#include <stdio.h>
#include <windows.h>
#include <iostream>
#include <sstream>  
#include "meshRefine.h"
#include "mesh.h"

void clear_Triangulateio(
	triangulateio* io
)
{
	if (io->pointlist != NULL)
		delete[] io->pointlist;
	if (io->trianglelist != NULL)
		delete[] io->trianglelist;
	if (io->neighborlist != NULL)
		delete[] io->neighborlist;
	if (io->segmentlist != NULL)
		delete[] io->segmentlist;
}

bool readInput_Synthetic(
	triangulateio *input,
	char* input_path,
	SyntheticParam input_sparam
)
{
	double angle = input_sparam.min_input_angle;
	if (angle > 60.0)
		angle = 60.0;

	printf("Try to read from synthetic input file: ");

	std::ostringstream strs;
	if (angle == 60.0)
		strs << input_path << "d" << input_sparam.distribution << "_s" << input_sparam.seed 
		<< "_p" << input_sparam.numberofpoints << "_c" << input_sparam.numberofsegments;
	else
		strs << input_path << "d" << input_sparam.distribution << "_s" << input_sparam.seed
		<< "_p" << input_sparam.numberofpoints << "_c" << input_sparam.numberofsegments
		<< "_with_minimum_input_angle_" << angle;

	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	triangulateio triInput, triResult;
	memset(&triInput, 0, sizeof(triangulateio));
	memset(&triResult, 0, sizeof(triangulateio));
	bool r = readCDT(&triInput, com);
	if (r)
	{
		triangulate("pzQn", &triInput, &triResult, NULL); //CDT mesh
		memcpy(input, &triResult, sizeof(triangulateio));
		delete[] com;
		clear_Triangulateio(&triInput);
		return true;
	}
	else
	{
		delete[] com;
		return false;
	}
}

bool saveInput_Synthetic(
	triangulateio *input,
	char* input_path,
	SyntheticParam input_sparam
)
{
	double angle = input_sparam.min_input_angle;
	if (angle > 60.0)
		angle = 60.0;

	std::ostringstream strs;
	if (angle == 60.0)
		strs << input_path << "d" << input_sparam.distribution << "_s" << input_sparam.seed
		<< "_p" << input_sparam.numberofpoints << "_c" << input_sparam.numberofsegments;
	else
		strs << input_path << "d" << input_sparam.distribution << "_s" << input_sparam.seed
		<< "_p" << input_sparam.numberofpoints << "_c" << input_sparam.numberofsegments
		<< "_with_minimum_input_angle_" << angle;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	saveCDT(input, com);

	delete[] com;
}

bool saveOutput_Synthetic(
	triangulateio *output,
	char* output_path,
	double total_time,
	double input_theta,
	double input_size,
	int input_mode,
	SyntheticParam input_sparam
)
{
	std::ostringstream strs;
	strs << output_path << "d" << input_sparam.distribution << "_s" << input_sparam.seed
		<< "_p" << input_sparam.numberofpoints << "_c" << input_sparam.numberofsegments << "_t" << input_theta;
	if (input_size > 0)
		strs << "_e" << input_size;
	strs << "_cpu";
	if (input_mode)
		strs << "_ruppert";
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

void refineInputByTriangle_Synthetic(
	char* input_path,
	double input_theta,
	double input_size,
	int input_mode,
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
	printf("Minimum allowable angle = %lf, Maximum edge size = %lf, Encroachment mode = %s\n", 
		input_theta, input_size, input_mode ? "Ruppert" : "Chew");

	printf("Preparing for synthetic input: ");
	triangulateio triInput, triOutput;
	memset(&triInput, 0, sizeof(triangulateio));
	if (!readInput_Synthetic(&triInput, input_path, input_sparam))
	{
		GenerateRandomInput(input_path, input_sparam.numberofpoints, input_sparam.numberofsegments, 
			input_sparam.seed, input_sparam.distribution, &triInput, input_sparam.min_input_angle, true);
		saveInput_Synthetic(&triInput, input_path, input_sparam);
	}

	printf("Refine mesh by Triangle...\n");
	clock_t tv[2];
	tv[0] = clock();
	std::ostringstream strs;
	strs << "pz";
	if (input_mode)
		strs << "D";
	strs << "Qnrq" << input_theta;
	if (input_size > 0)
		strs << "u" << input_size;
	std::string com_str = strs.str();
	char *com = new char[com_str.length() + 1];
	strcpy(com, com_str.c_str());

	memset(&triOutput, 0, sizeof(triangulateio));
	triangulate(com, &triInput, &triOutput, NULL); //CDT mesh
	tv[1] = clock();
	double total_time = (REAL)(tv[1] - tv[0]);
	printf("Total time = %.3f ms\n", total_time);
	printf("Number of points = %d\nnumber of triangles = %d\nnumber of segments = %d\n",
		triOutput.numberofpoints, triOutput.numberoftriangles, triOutput.numberofsegments);
	printf("\n");

	if (output_path != NULL)
	{
		printf("Output statistic...\n");
		saveOutput_Synthetic(&triOutput, output_path, total_time, input_theta, input_size, input_mode, input_sparam);
	}

	delete[] com;
	clear_Triangulateio(&triInput);
	clear_Triangulateio(&triOutput);
}

bool readInput_Real(
	triangulateio *input,
	char* input_path,
	char* input_file
)
{
	std::ostringstream strs;
	strs << input_path << input_file << "_vertex.txt";
	std::ostringstream strs1;
	strs1 << input_path << input_file << "_constraint.txt";
	std::string vertex_str = strs.str();
	char *file_vertex = new char[vertex_str.length() + 1];
	strcpy(file_vertex, vertex_str.c_str());
	std::string constraint_str = strs1.str();
	char *file_constraint = new char[constraint_str.length() + 1];
	strcpy(file_constraint, constraint_str.c_str());

	triangulateio triInput;
	memset(&triInput, 0, sizeof(triangulateio));
	
	printf("Try to read from vertex file: ");
	// Read points first
	FILE *fp;
	fopen_s(&fp, file_vertex, "rb");
	
	if(fp == NULL)
	{
		printf("Cannot find the input file\n");
		return false;
	}
	else
		printf("Succeed\n");
	
	int numofpoints;
	fread(&numofpoints, sizeof(int), 1, fp);
	
	float * pointlist = new float[2*numofpoints];
	fread(pointlist, sizeof(float), 2*numofpoints, fp);
	
	fclose(fp);
	
	printf("Try to read from constraint file: ");
	// Read constraints
	fopen_s(&fp,file_constraint, "rb");
	
	if(fp == NULL)
	{
		printf("Cannot find the input file\n");
		return false;
	}
	else
		printf("Succeed\n");
	
	int numofsegs = 0; 
	fread(&numofsegs, sizeof(int), 1, fp);
	
	int * segmentlist = new int[numofsegs*2];
	fread(segmentlist,sizeof(int),numofsegs*2,fp);
	
	fclose(fp);
	
	triInput.numberofpoints = numofpoints;
	triInput.pointlist = new double[2*numofpoints];
	for(int i=0; i<numofpoints; i++)
	{
		triInput.pointlist[2*i] = pointlist[2*i];
		triInput.pointlist[2*i+1] = pointlist[2*i+1];
	}
	
	triInput.numberofsegments = numofsegs;
	triInput.segmentlist = new int[2* numofsegs];
	for(int i=0; i<numofsegs; i++)
	{
		int segid = i;
		int p1 = segmentlist[2*segid];
		int p2 = segmentlist[2*segid+1];
		triInput.segmentlist[2*i] = p1;
		triInput.segmentlist[2*i+1] = p2;
	}
	
	// Compute CDT
	triangulateio triCDT;
	memset(&triCDT, 0, sizeof(triangulateio));
	triangulate("pzQnc", &triInput, &triCDT, NULL);
	
	memcpy(input, &triCDT, sizeof(triangulateio));
	clear_Triangulateio(&triInput);
	delete[] file_vertex;
	delete[] file_constraint;
	delete[] pointlist;
	delete[] segmentlist;

	return true;
}

bool saveOutput_Real(
	triangulateio *output,
	char* output_path,
	double total_time,
	char* input_file,
	double input_theta,
	double input_size,
	int input_mode
)
{
	std::ostringstream strs;
	strs << output_path << input_file << "_t" << input_theta;
	if (input_size > 0)
		strs << "_e" << input_size;
	strs << "_cpu";
	if (input_mode)
		strs << "_ruppert";
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
	/*strs << ".ply";
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
	fprintf(fp, "element edge %d\n", output->numberofsegments);
	fprintf(fp, "property int vertex1\n");
	fprintf(fp, "property int vertex2\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "end_header\n");

	for (int i = 0; i < output->numberofpoints; i++)
	{
		fprintf(fp, "%lf %lf 0\n", output->pointlist[2 * i + 0], output->pointlist[2 * i + 1]);
	}

	for (int i = 0; i < output->numberofsegments; i++)
	{
		fprintf(fp, "%d %d 255 0 0\n", output->segmentlist[2 * i + 0], output->segmentlist[2 * i + 1]);
	}

	fclose(fp);

	delete[] com;*/
}

void refineInputByTriangle_Real(
	char* input_path,
	char* input_file,
	double input_theta,
	double input_size,
	int input_mode,
	char* output_path
)
{
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

	printf("Minimum allowable angle = %lf, Maximum edge size = %lf, Encroachment mode = %s\n",
		input_theta, input_size, input_mode ? "Ruppert" : "Chew");

	printf("Preparing for real-world input %s%s\n", input_path, input_file);
	triangulateio triInput, triOutput;
	memset(&triInput, 0, sizeof(triangulateio));
	if (!readInput_Real(&triInput, input_path, input_file))
	{
		printf("Coundn't find the input!\n");
		exit(0);
	}

	printf("Refine mesh by Triangle...\n");
	clock_t tv[2];
	tv[0] = clock();
	std::ostringstream strs;
	strs << "pz";
	if (input_mode)
		strs << "D";
	strs << "Qnrq" << input_theta;
	if (input_size > 0)
		strs << "u" << input_size;
	std::string com_str = strs.str();
	char *com = new char[com_str.length() + 1];
	strcpy(com, com_str.c_str());

	memset(&triOutput, 0, sizeof(triangulateio));
	triangulate(com, &triInput, &triOutput, NULL); //CDT mesh
	tv[1] = clock();
	double total_time = (REAL)(tv[1] - tv[0]);
	printf("Total time = %.3f ms\n", total_time);
	printf("Number of points = %d\nnumber of triangles = %d\nnumber of segments = %d\n",
		triOutput.numberofpoints, triOutput.numberoftriangles, triOutput.numberofsegments);
	printf("\n");

	if (output_path != NULL)
	{
		printf("Output statistic...\n");
		saveOutput_Real(&triOutput, output_path, total_time, input_file, input_theta, input_size, input_mode);
	}

	delete[] com;
	clear_Triangulateio(&triInput);
	clear_Triangulateio(&triOutput);
}