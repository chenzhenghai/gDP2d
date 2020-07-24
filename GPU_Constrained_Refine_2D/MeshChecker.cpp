#include "MeshChecker.h"
#include "cudaRefine.h"
#include "predicates.h"

// Find orientataion of tri2's edge that is incident to tri1
int findIncidentOri_Host(int * trianglelist, int tri1, int tri2)
{
	if (tri1 < 0 || tri2 < 0)
		return -1;
	int inc0 = -1, inc1 = -1;
	int tri1_p[3] = {
		trianglelist[3 * tri1],
		trianglelist[3 * tri1 + 1],
		trianglelist[3 * tri1 + 2]
	};
	int tri2_p[3] = {
		trianglelist[3 * tri2],
		trianglelist[3 * tri2 + 1],
		trianglelist[3 * tri2 + 2]
	};

	// incident edge
	int count = 0;
	for (int i = 0; i<3; i++)
	{
		for (int j = 0; j<3; j++)
		{
			if (tri1_p[i] == tri2_p[j])
			{
				if (count == 0)
				{
					inc0 = j;
					count += 1;
					continue;
				}
				else
				{
					inc1 = j;
					break;
				}
			}
		}
	}

	if (inc0 == -1 || inc1 == -1) // not found
		return -1;

	// orientation
	int differ = inc0 - inc1;
	int index;
	if (differ == -1 || differ == 2)
		index = inc0;
	else
		index = inc1;

	if (index == 0)
		return 2;
	else if (index == 1)
		return 0;
	else if (index == 2)
		return 1;
	else
		return -1;
}

bool checkNeighbors(triangulateio *input)
{
	for (int i = 0; i<input->numberoftriangles; i++)
	{
		for (int j = 0; j<3; j++)
		{
			int neighbor = input->neighborlist[3 * i + j];
			if (neighbor == -1)
				continue;
			int flag = 0;
			for (int k = 0; k<3; k++)
			{
				if (input->neighborlist[3 * neighbor + k] == i)
				{
					flag++;
					if (findIncidentOri_Host(input->trianglelist, i, neighbor) != -1)
						flag++;
					break;
				}
			}
			if (flag != 2)
			{
				printf("Invalid vertices or neighbors %d: (%d,%d,%d) - (%d,%d,%d) | %d: (%d,%d,%d) - (%d,%d,%d) !\n",
					i,
					input->trianglelist[3 * i], input->trianglelist[3 * i + 1], input->trianglelist[3 * i + 2],
					input->neighborlist[3 * i], input->neighborlist[3 * i + 1], input->neighborlist[3 * i + 2],
					neighbor,
					input->trianglelist[3 * neighbor], input->trianglelist[3 * neighbor + 1], input->trianglelist[3 * neighbor + 2],
					input->neighborlist[3 * neighbor], input->neighborlist[3 * neighbor + 1], input->neighborlist[3 * neighbor + 2]);
				return false;
			}
		}
	}
	return true;
}

void printTriangles(triangulateio *input)
{
	for (int i = 0; i<input->numberoftriangles; i++)
	{
		printf("%d: ", i);
		for (int j = 0; j<3; j++)
			printf("%d ", input->trianglelist[3 * i + j]);
		printf("- ");
		for (int j = 0; j<3; j++)
			printf("%d ", input->neighborlist[3 * i + j]);
		printf("\n");
	}
}

void printPoints(triangulateio *input, PStatus* debug_ps)
{
	for (int i = 0; i<input->numberofpoints; i++)
	{
		printf("%f, %f", input->pointlist[2 * i], input->pointlist[2 * i + 1]);
		if (debug_ps != NULL)
		{
			if (debug_ps[i].isSegmentSplit())
				printf(" midpoint");
			else
				printf(" steiner");
		}
		printf("\n");
	}
}

void printSegments(triangulateio *input)
{
	for (int i = 0; i<input->numberofsegments; i++)
	{
		printf("%d,%d\n", input->segmentlist[2 * i], input->segmentlist[2 * i + 1]);
	}
}

bool checkIncircle(triangulateio *input)
{
	int edgecount = 0;
	int segcount = 0;

	for (int i = 0; i<input->numberoftriangles; i++)
	{
		for (int ori = 0; ori<3; ori++)
		{
			int neighbor = input->neighborlist[3 * i + ori];
			if (neighbor == -1)
			{
				edgecount += 2;
				segcount += 2;
				continue;
			}
			else
				edgecount++;

			int org, dest, apex;
			double x, y;

			// origin point
			org = input->trianglelist[3 * i + (ori + 1) % 3];
			x = input->pointlist[2 * org];
			y = input->pointlist[2 * org + 1];
			triVertex triOrg(x, y);

			// destination point
			dest = input->trianglelist[3 * i + (ori + 2) % 3];
			x = input->pointlist[2 * dest];
			y = input->pointlist[2 * dest + 1];
			triVertex triDest(x, y);

			bool seg = false;
			for (int j = 0; j<input->numberofsegments; j++)
			{
				int p0 = input->segmentlist[2 * j];
				int p1 = input->segmentlist[2 * j + 1];
				if ((org == p0 && dest == p1) || (org == p1 && dest == p0))
				{
					// dont need to check this edge
					segcount++;
					seg = true;
					break;
				}
			}
			if (seg)
				continue; // skip this edge

						  // apex point
			apex = input->trianglelist[3 * i + ori];
			x = input->pointlist[2 * apex];
			y = input->pointlist[2 * apex + 1];
			triVertex triApex(x, y);

			// opposite Apex point
			int oppOri = findIncidentOri_Host(input->trianglelist, i, neighbor);
			int oppApex = input->trianglelist[3 * neighbor + oppOri];
			x = input->pointlist[2 * oppApex];
			y = input->pointlist[2 * oppApex + 1];
			triVertex triOppApex(x, y);

			REAL test = incircle(&triOrg, &triDest, &triApex, &triOppApex);

			if (test > 0)
			{
				printf("Incircle test fail for triangle %d and %d - %d(%f,%f), %d(%f,%f), %d(%f,%f), %d(%f,%f)",
					i, neighbor,
					org, input->pointlist[2 * org], input->pointlist[2 * org + 1],
					dest, input->pointlist[2 * dest], input->pointlist[2 * dest + 1],
					apex, input->pointlist[2 * apex], input->pointlist[2 * apex + 1],
					oppApex, input->pointlist[2 * oppApex], input->pointlist[2 * oppApex + 1]);
				printf(",incircle = %lf\n", test);
				//return false;
			}

		}
	}
	//printf("%d\n",segcount/2);
	edgecount /= 2;
	int euler = input->numberofpoints - edgecount + input->numberoftriangles + 1;
	if (euler != 2)
	{
		printf("Euler equation test fail!\n");
		return false;
	}

	return true;
}

bool isPureBadTriangle(triVertex vOrg, triVertex vDest, triVertex vApex, double theta)
{
	REAL dx[3], dy[3], edgelength[3];

	REAL goodAngle = cos(theta * PI / 180.0);
	goodAngle *= goodAngle;

	triVertex p[3];
	p[0] = vOrg;
	p[1] = vDest;
	p[2] = vApex;

	for (int i = 0; i < 3; i++)
	{
		int j = (i + 1) % 3;
		int k = (i + 2) % 3;
		dx[i] = p[j].x - p[k].x;
		dy[i] = p[j].y - p[k].y;
		edgelength[i] = dx[i] * dx[i] + dy[i] * dy[i];
	}

	for (int i = 0; i < 3; i++)
	{
		int  j = (i + 1) % 3;
		int  k = (i + 2) % 3;
		REAL dotproduct = dx[j] * dx[k] + dy[j] * dy[k];
		REAL cossquare = dotproduct * dotproduct / (edgelength[j] * edgelength[k]);
		if (cossquare > goodAngle)
		{
			return true;
		}
	}

	return false;
}

bool checkQuality(triangulateio *input, double theta)
{
	triVertex p[3];
	REAL dx[3], dy[3], edgelength[3];

	REAL goodAngle = cos(theta * PI / 180.0);
	goodAngle *= goodAngle;

	for (int num = 0; num < input->numberoftriangles; num++)
	{
		int org, dest, apex;

		org = input->trianglelist[3 * num + 1];
		dest = input->trianglelist[3 * num + 2];
		apex = input->trianglelist[3 * num];

		p[0].x = input->pointlist[2 * org];
		p[0].y = input->pointlist[2 * org + 1];
		p[1].x = input->pointlist[2 * dest];
		p[1].y = input->pointlist[2 * dest + 1];
		p[2].x = input->pointlist[2 * apex];
		p[2].y = input->pointlist[2 * apex + 1];

		for (int i = 0; i < 3; i++)
		{
			int j = (i + 1) % 3;
			int k = (i + 2) % 3;
			dx[i] = p[j].x - p[k].x;
			dy[i] = p[j].y - p[k].y;
			edgelength[i] = dx[i] * dx[i] + dy[i] * dy[i];
		}

		for (int i = 0; i < 3; i++)
		{
			int  j = (i + 1) % 3;
			int  k = (i + 2) % 3;
			REAL dotproduct = dx[j] * dx[k] + dy[j] * dy[k];
			REAL cossquare = dotproduct * dotproduct / (edgelength[j] * edgelength[k]);
			if (cossquare > goodAngle)
			{
				printf("Bad triangle %i, smallest angles's cossquare = %f, goodAngle = %f\n",
					num, cossquare, goodAngle);
				//return false;
			}
		}
	}

	return true;
}

bool checkResult(triangulateio *input, triangulateio *output, double theta)
{
	printf("Checking result......\n");

	// check input points
	printf("Checking input points......\n");
	for (int i = 0; i<input->numberofpoints; i++)
	{
		if (input->pointlist[2 * i] != output->pointlist[2 * i] ||
			input->pointlist[2 * i + 1] != output->pointlist[2 * i + 1])
		{
			printf("Missing input point %d !\n", i);
			return false;
		}
	}

	// check vertices' indices
	printf("Checking indices......\n");
	for (int i = 0; i<input->numberoftriangles; i++)
	{
		int index0, index1, index2;
		index0 = input->trianglelist[3 * i];
		index1 = input->trianglelist[3 * i + 1];
		index2 = input->trianglelist[3 * i + 2];
		if (index0 < 0 || index0 >= input->numberofpoints ||
			index1 < 0 || index1 >= input->numberofpoints ||
			index2 < 0 || index2 >= input->numberofpoints)
		{
			printf("Invalid triangle indices %d: %d, %d, %d\n",
				i, index0, index1, index2);
			return false;
		}
	}

	// check neighbors
	printf("Checking neighbors......\n");
	if (!checkNeighbors(output))
		return false;

	// check quality
	printf("Checking quality......\n");
	if (!checkQuality(output, theta))
		return false;

	// check incircle property (slow)
	printf("Checking incircle property......\n");
	if(!checkIncircle(output))
		return false;

	return true;
}
