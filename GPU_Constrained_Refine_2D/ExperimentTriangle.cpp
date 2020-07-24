#include <stdio.h>
#include <iostream>
#include <sstream>  
#include "Experiment.h"

bool readOutput_Synthetic(
	char * filename, 
	int * numberofpoints, 
	int * numberoftriangles, 
	int * numberofsegments, 
	double * runtime
)
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

//void experiment_triangle(int mode, double min_allowable_angle, double min_input_angle)
//{
//	int numOfPoints,distribution,seed,numOfSegments;
//	seed = 0;
//	for(distribution = 0; distribution <= 3; distribution++)
//	{
//		for(numOfPoints = 100000; numOfPoints <= 100000; numOfPoints += 10000)
//		{
//			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
//			{
//				if(true)
//				{
//					//if( (distribution == 0 && numOfPoints == 60000) ||
//					//	(distribution == 0 && numOfPoints == 100000 && (numOfSegments == 40000 || numOfSegments == 50000)) ||
//					//	(distribution == 2 && numOfPoints == 60000) ||
//					//	(distribution == 3 && numOfPoints == 70000))
//					//	seed = 1;
//					//else if ( (distribution == 1 && numOfPoints == 60000) ||
//					//		  (distribution == 2 && numOfPoints == 80000))
//					//	seed = 2;
//					//else
//					//	seed = 0;
//
//					printf("Random Input: Numberofpoints = %d, Numberofsegment = %d, Distribution = %d, Seed = %d\n",
//						numOfPoints, numOfSegments, distribution, seed);
//
//					printf("Running Mode: %s, minimum allowable angle = %lf, minimum input angle = %lf\n",mode? "Ruppert":"Chew",
//						min_allowable_angle, min_input_angle);
//
//					triangulateio triInput;
//					std::ostringstream strs;
//					strs << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << min_allowable_angle << "_cpu";
//					if(mode)
//						strs << "_ruppert";
//					if(min_input_angle < 60.0)
//						strs << "_with_minimum_input_angle_" << min_input_angle;
//					std::string fn = strs.str();
//					char *com = new char[fn.length() + 1];
//					strcpy(com, fn.c_str());
//					if(readOutput(com,NULL, NULL, NULL, NULL))
//					{
//						printf("Find the output file, Skip!\n");
//					}
//					else if(!readInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle))
//					{
//						printf("Don't find the input file, Skip!\n");
//					}
//					else
//					{
//						printf("Triangle is running...\n");
//						StopWatchInterface *timer = 0; // timer
//						sdkCreateTimer( &timer );
//						double cpu_time;
//						sdkResetTimer( &timer );
//						sdkStartTimer( &timer );
//						triangulateio cpu_result;
//						CPU_Triangle_Quality(&triInput,&cpu_result,min_allowable_angle,mode);
//						sdkStopTimer( &timer );
//						cpu_time = sdkGetTimerValue( &timer );
//						saveOutput(com,&cpu_result,distribution,seed,numOfPoints,numOfSegments,min_allowable_angle,cpu_time);
//						delete[] triInput.pointlist;
//						delete[] triInput.trianglelist;
//						delete[] triInput.neighborlist;
//						delete[] triInput.segmentlist;
//						delete[] cpu_result.pointlist;
//						delete[] cpu_result.trianglelist;
//						delete[] cpu_result.neighborlist;
//						delete[] cpu_result.segmentlist;
//					}
//					
//					printf("\n");
//				}
//			}
//		}
//	}
//}

//void experiment_triangle_ruppert(double angle)
//{
//	double min_input_angle = 15.0;
//	int numOfPoints,distribution,seed,numOfSegments;
//	seed = 0;
//	for(distribution = 0; distribution <= 3; distribution++)
//	{
//		for(numOfPoints = 50000; numOfPoints <= 100000; numOfPoints += 10000)
//		{
//			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
//			{
//				if( true )
//				{
//					if( //(distribution == 0 && numOfPoints == 100000 && (numOfSegments == 40000 || numOfSegments == 50000)) ||
//						(distribution == 1 && numOfPoints == 70000 && (numOfSegments == 28000 || numOfSegments == 35000)) ||
//						(distribution == 2 && numOfPoints == 80000 && (numOfSegments == 32000 || numOfSegments == 40000)) ||
//						(distribution == 3 && numOfPoints == 70000))
//						seed = 1;
//					else if ((distribution == 3 && numOfPoints == 90000 && numOfSegments == 45000))
//						seed = (angle == 20) ? 0:2;
//					else
//						seed = 0;
//					double theta = angle;
//
//					printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d\n",
//						numOfPoints, numOfSegments, distribution);
//					triangulateio triInput;
//					std::ostringstream strs;
//					strs << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_cpu_ruppert";
//					std::string fn = strs.str();
//					char *com = new char[fn.length() + 1];
//					strcpy(com, fn.c_str());
//					if(readOutput(com, NULL, NULL, NULL, NULL))
//					{
//						printf("Find the output file, Skip!\n");
//					}
//					else if(!readInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle))
//					{
//						printf("Don't find the input file, Skip!\n");
//					}
//					else
//					{
//						printf("Triangle is running...\n");
//						StopWatchInterface *timer = 0; // timer
//						sdkCreateTimer( &timer );
//						double cpu_time;
//						sdkResetTimer( &timer );
//						sdkStartTimer( &timer );
//						triangulateio cpu_result;
//						CPU_Triangle_Quality(&triInput,&cpu_result,theta,1);
//						sdkStopTimer( &timer );
//						cpu_time = sdkGetTimerValue( &timer );
//						saveOutput(com,&cpu_result,distribution,seed,numOfPoints,numOfSegments,theta,cpu_time);
//						delete[] triInput.pointlist;
//						delete[] triInput.trianglelist;
//						delete[] triInput.neighborlist;
//						delete[] triInput.segmentlist;
//						delete[] cpu_result.pointlist;
//						delete[] cpu_result.trianglelist;
//						delete[] cpu_result.neighborlist;
//						delete[] cpu_result.segmentlist;
//					}
//					
//					printf("\n");
//				}
//			}
//		}
//	}
//}
