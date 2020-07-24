#include <stdio.h>
#include <time.h>
#include <iostream>
#include <sstream> 

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <iostream>

#include "meshRefine.h"

#define PI 3.141592653589793238462643383279502884197169399375105820974944592308

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;

bool readInput_Synthetic(
	SyntheticIO* input,
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

	FILE *fp;
	fp = fopen(com, "r");

	if (fp == NULL)
	{
		printf("Cannot find the input file\n");
		delete[] com;
		return false;
	}

	char buf[100];

	int ln = 0;

	while (fgets(buf, 100, fp) != NULL) {
		if (ln == 0)
		{
			if (sscanf(buf, "%d", &(input->numberofpoints)) != 1)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
			else
				input->pointlist = new double[2 * input->numberofpoints];
		}
		else if (ln == 1)
		{
			if (sscanf(buf, "%d", &(input->numberoftriangles)) != 1)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
			else
			{
				input->trianglelist = new int[3 * input->numberoftriangles];
				input->neighborlist = new int[3 * input->numberoftriangles];
			}
		}
		else if (ln == 2)
		{
			if (sscanf(buf, "%d", &(input->numberofsegments)) != 1)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
			else
				input->segmentlist = new int[2 * input->numberofsegments];
		}
		else if (ln < input->numberofpoints + 3)
		{
			if (sscanf(buf, "%lf %lf",
				input->pointlist + 2 * (ln - 3),
				input->pointlist + 2 * (ln - 3) + 1) != 2)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
		}
		else if (ln < input->numberofpoints + input->numberoftriangles + 3)
		{
			if (sscanf(buf, "%d %d %d",
				input->trianglelist + 3 * (ln - input->numberofpoints - 3),
				input->trianglelist + 3 * (ln - input->numberofpoints - 3) + 1,
				input->trianglelist + 3 * (ln - input->numberofpoints - 3) + 2) != 3)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
		}
		else if (ln < input->numberofpoints + 2 * input->numberoftriangles + 3)
		{
			if (sscanf(buf, "%d %d %d",
				input->neighborlist + 3 * (ln - input->numberofpoints - input->numberoftriangles - 3),
				input->neighborlist + 3 * (ln - input->numberofpoints - input->numberoftriangles - 3) + 1,
				input->neighborlist + 3 * (ln - input->numberofpoints - input->numberoftriangles - 3) + 2) != 3)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
		}
		else if (ln < input->numberofpoints + 2 * input->numberoftriangles + input->numberofsegments + 3)
		{
			if (sscanf(buf, "%d %d",
				input->segmentlist + 2 * (ln - input->numberofpoints - 2 * input->numberoftriangles - 3),
				input->segmentlist + 2 * (ln - input->numberofpoints - 2 * input->numberoftriangles - 3) + 1) != 2)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
		}
		else
			break;
		ln++;
	}

	fclose(fp);
	printf("Succeed\n");
	delete[] com;
	return true;
}

bool saveOutput_Synthetic(
	CDT *output,
	char* output_path,
	double total_time,
	double input_theta,
	double input_size,
	SyntheticParam input_sparam
)
{
	std::ostringstream strs;
	strs << output_path << "d" << input_sparam.distribution << "_s" << input_sparam.seed
		<< "_p" << input_sparam.numberofpoints << "_c" << input_sparam.numberofsegments << "_t" << input_theta;
	if (input_size > 0.0)
		strs << "_e" << input_size;
	strs << "_cgal" << "_ruppert";
	if (input_sparam.min_input_angle < 60.0)
		strs << "_with_minimum_input_angle_" << input_sparam.min_input_angle;
	std::string com_str = strs.str();
	char *com = new char[com_str.length() + 1];
	strcpy(com, com_str.c_str());

	FILE *fp;
	fp = fopen(com, "w");

	int numberoftriangles = std::distance(output->finite_faces_begin(), output->finite_faces_end());
	int numberofsegments = std::distance(output->constrained_edges_begin(), output->constrained_edges_end());
	fprintf(fp, "Number of points = %d\n", output->number_of_vertices());
	fprintf(fp, "Number of triangles = %d\n", numberoftriangles);
	fprintf(fp, "Number of segments = %d\n", numberofsegments);
	fprintf(fp, "Runtime = %lf\n", total_time);
	fclose(fp);

	delete[] com;
}

void refineInputByCGAL_Synthetic(
	char* input_path,
	double input_theta,
	double input_size,
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
	printf("Minimum allowable angle = %lf, Maximum edge size = %lf\n", input_theta, input_size);

	printf("Preparing for synthetic input: ");
	SyntheticIO input;
	if (!readInput_Synthetic(&input, input_path, input_sparam))
	{
		printf("Coundn't find the input!\n");
		exit(0);
	}

	printf("Initializing the CDT...\n");
	CDT cdt;
	Vertex_handle * vh = new Vertex_handle[input.numberofpoints];
	for (int i = 0; i < input.numberofpoints; i++)
	{
		vh[i] = cdt.insert(Point(input.pointlist[2 * i + 0], input.pointlist[2 * i + 1]));
	}
	for (int i = 0; i < input.numberofsegments; i++)
	{
		cdt.insert_constraint(vh[input.segmentlist[2 * i + 0]], vh[input.segmentlist[2 * i + 1]]);
	}

	std::cout << "Refine mesh by CGAL..." << std::endl;
	clock_t tv[2];
	tv[0] = clock();
	double B = 1 / (2 * sin(input_theta / 180 * PI));
	double b = 1 / (4 * B * B);
	//printf("b = %lf\n", b);
	CGAL::refine_Delaunay_mesh_2(cdt, Criteria(b, input_size));
	tv[1] = clock();
	double total_time = (double)(tv[1] - tv[0]);
	printf("Total time = %.3f ms\n", total_time);
	printf("Number of points = %d\nnumber of triangles = %d\nnumber of segments = %d\n",
		cdt.number_of_vertices(), std::distance(cdt.finite_faces_begin(), cdt.finite_faces_end()), 
		std::distance(cdt.constrained_edges_begin(), cdt.constrained_edges_end()));
	printf("\n");

	if (output_path != NULL)
	{
		printf("Output statistic...\n");
		saveOutput_Synthetic(&cdt, output_path, total_time, input_theta, input_size, input_sparam);
	}
}

bool readInput_Real(
	SyntheticIO* input,
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

	printf("Try to read from vertex file: ");
	// Read points first
	FILE *fp;
	fopen_s(&fp, file_vertex, "rb");

	if (fp == NULL)
	{
		printf("Cannot find the input file\n");
		return false;
	}
	else
		printf("Succeed\n");

	int numofpoints;
	fread(&numofpoints, sizeof(int), 1, fp);

	float * pointlist = new float[2 * numofpoints];
	fread(pointlist, sizeof(float), 2 * numofpoints, fp);

	fclose(fp);

	printf("Try to read from constraint file: ");
	// Read constraints
	fopen_s(&fp, file_constraint, "rb");

	if (fp == NULL)
	{
		printf("Cannot find the input file\n");
		return false;
	}
	else
		printf("Succeed\n");

	int numofsegs = 0;
	fread(&numofsegs, sizeof(int), 1, fp);

	int * segmentlist = new int[numofsegs * 2];
	fread(segmentlist, sizeof(int), numofsegs * 2, fp);

	fclose(fp);

	input->numberofpoints = numofpoints;
	input->pointlist = new double[2 * numofpoints];
	for (int i = 0; i<numofpoints; i++)
	{
		input->pointlist[2 * i] = pointlist[2 * i];
		input->pointlist[2 * i + 1] = pointlist[2 * i + 1];
	}

	input->numberofsegments = numofsegs;
	input->segmentlist = new int[2 * numofsegs];
	for (int i = 0; i<numofsegs; i++)
	{
		int segid = i;
		int p1 = segmentlist[2 * segid];
		int p2 = segmentlist[2 * segid + 1];
		input->segmentlist[2 * i] = p1;
		input->segmentlist[2 * i + 1] = p2;
	}

	delete[] file_vertex;
	delete[] file_constraint;
	delete[] pointlist;
	delete[] segmentlist;

	return true;
}

void elements_statistic(
	CDT &output,
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
	for (CDT::Finite_faces_iterator it = output.finite_faces_begin(),
		end = output.finite_faces_end(); it != end; ++it)
	{
		isbadangle = false;
		isbadsize = false;
		CDT::Face face = *it;

		for (int i = 0; i < 3; i++)
		{
			p[i][0] = face.vertex(i)->point().x();
			p[i][1] = face.vertex(i)->point().y();
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
	CDT *output,
	char* output_path,
	double total_time,
	char* input_file,
	double input_theta,
	double input_size
)
{
	std::ostringstream strs;
	strs << output_path << input_file <<  "_t" << input_theta;
	if (input_size > 0.0)
		strs << "_e" << input_size;
	strs << "_cgal" << "_ruppert";
	std::string com_str = strs.str();
	char *com = new char[com_str.length() + 1];
	strcpy(com, com_str.c_str());

	FILE *fp;
	fp = fopen(com, "w");

	int numberoftriangles = std::distance(output->finite_faces_begin(), output->finite_faces_end());
	int numberofsegments = std::distance(output->constrained_edges_begin(), output->constrained_edges_end());
	int numberofbadtris_theta, numberofbadtris_size, numberofbadtris_total;
	double total_bad_angle_area, total_bad_size_area, total_bad_area, total_area;
	elements_statistic(*output, numberofbadtris_theta, numberofbadtris_size, numberofbadtris_total,
		total_bad_angle_area, total_bad_size_area, total_bad_area, total_area, input_theta, input_size);
	fprintf(fp, "Number of points = %d\n", output->number_of_vertices());
	fprintf(fp, "Number of triangles = %d\n", numberoftriangles);
	fprintf(fp, "Number of segments = %d\n", numberofsegments);
	fprintf(fp, "Runtime = %lf\n", total_time);
	fprintf(fp, "Number of bad triangles = %d\n", numberofbadtris_total);
	fprintf(fp, "Number of bad triangles (angle) = %d\n", numberofbadtris_theta);
	fprintf(fp, "Number of bad triangles (size) = %d\n", numberofbadtris_size);
	fprintf(fp, "Bad triangle ratio = %.3f\n", numberofbadtris_total*1.0 / numberoftriangles);
	fprintf(fp, "Total area = %.3f\n", total_area);
	fprintf(fp, "Total bad area = %.3f\n", total_bad_area);
	fprintf(fp, "Total bad area (angle) = %.3f\n", total_bad_angle_area);
	fprintf(fp, "Total bad area (size) = %.3f\n", total_bad_size_area);
	fprintf(fp, "Bad area ratio = %.3f\n", total_bad_area / total_area);
	fclose(fp);

	delete[] com;
}

void refineInputByCGAL_Real(
	char* input_path,
	char* input_file,
	double input_theta,
	double input_size,
	char* output_path
)
{
	time_t rawtime;
	struct tm * timeinfo;
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Launch time is %d:%d:%d\n", timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
	printf("Minimum allowable angle = %lf, Maximum edge size = %lf\n", input_theta, input_size);

	printf("Preparing for real-world input %s%s\n", input_path, input_file);
	SyntheticIO input;
	if (!readInput_Real(&input, input_path, input_file))
	{
		printf("Coundn't find the input!\n");
		exit(0);
	}

	printf("Initializing the CDT...\n");
	CDT cdt;
	Vertex_handle * vh = new Vertex_handle[input.numberofpoints];
	for (int i = 0; i < input.numberofpoints; i++)
	{
		vh[i] = cdt.insert(Point(input.pointlist[2 * i + 0], input.pointlist[2 * i + 1]));
	}
	for (int i = 0; i < input.numberofsegments; i++)
	{
		cdt.insert_constraint(vh[input.segmentlist[2 * i + 0]], vh[input.segmentlist[2 * i + 1]]);
	}
	printf("numberofpoints = %d, numberofsegments = %d\n", input.numberofpoints, input.numberofsegments);

	std::cout << "Refine mesh by CGAL..." << std::endl;
	clock_t tv[2];
	tv[0] = clock();
	double B = 1 / (2 * sin(input_theta / 180 * PI));
	double b = 1 / (4 * B * B);
	CGAL::refine_Delaunay_mesh_2(cdt, Criteria(b, input_size));
	tv[1] = clock();
	double total_time = (double)(tv[1] - tv[0]);
	printf("Total time = %.3f ms\n", total_time);
	printf("Number of points = %d\nnumber of triangles = %d\nnumber of segments = %d\n",
		cdt.number_of_vertices(), std::distance(cdt.finite_faces_begin(), cdt.finite_faces_end()),
		std::distance(cdt.constrained_edges_begin(), cdt.constrained_edges_end()));
	printf("\n");

	if (output_path != NULL)
	{
		printf("Output statistic...\n");
		saveOutput_Real(&cdt, output_path, total_time, input_file, input_theta, input_size);
	}
}