
//void experiment_input(int mode, double min_input_angle, bool checkinter)
//{
//	int numOfPoints,distribution,seed,numOfSegments;
//	for(distribution = 3; distribution <= 3; distribution++)
//	{
//		for(numOfPoints = 100000; numOfPoints <= 100000; numOfPoints += 10000)
//		{
//			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
//			{
//				//if(mode)
//				//{
//				//	if( (distribution == 0 && numOfPoints == 90000)  ||
//				//		(distribution == 1 && numOfPoints == 60000)  ||
//				//		(distribution == 2 && numOfPoints == 80000)  ||
//				//		(distribution == 3 && numOfPoints == 70000))
//				//		seed = 1;
//				//	else
//				//		seed = 0;
//				//}
//				//else
//				//{
//				//	if( (distribution == 0 && (numOfPoints == 60000 || numOfPoints == 90000)) ||
//				//		(distribution == 2 && numOfPoints == 60000) ||
//				//		(distribution == 3 && numOfPoints == 70000))
//				//		seed = 1;
//				//	else if ( (distribution == 1 && numOfPoints == 60000) ||
//				//				(distribution == 2 && numOfPoints == 80000))
//				//		seed = 2;
//				//	else
//				//		seed = 0;
//				//}
//				seed = 0;
//
//				printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d, Seed = %d\n",
//					numOfPoints, numOfSegments, distribution, seed);
//
//				std::ostringstream strs;
//				if(min_input_angle == 60.0)
//					strs << "input/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments;
//				else
//					strs << "input/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_with_minimum_input_angle_"  << min_input_angle;
//				std::string fn = strs.str();
//				char *com = new char[fn.length() + 1];
//				strcpy(com, fn.c_str());
//
//				FILE *fp;
//				fp = fopen(com, "r");
//
//				triangulateio triInput;
//				if(fp == NULL)
//				{
//					printf("Failed to find input file, start generating...\n");
//					GenerateRandomInput(numOfPoints,numOfSegments,seed,distribution,&triInput,min_input_angle, checkinter);
//					saveInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle);
//					delete[] triInput.pointlist;
//					delete[] triInput.trianglelist;
//					delete[] triInput.segmentlist;
//				}
//				else
//				{
//					printf("Found input file, Skip!\n");
//					fclose(fp);
//				}
//						
//				printf("\n");
//			}
//		}
//	}
//}