/*************************************************************************************/
/*  A Two-Dimensional Constrained Delaunay Refiner on GPU.							 */
/*  (refine.cu)																		 */
/*																					 */
/*  Version XX																		 */
/*  June 26, 2016																	 */
/*																					 */
/*  Copyright 2016																	 */
/*	Chen Zhenghai																	 */
/*  National University of Singapore												 */
/*  chenzhenghai@u.nus.edu															 */
/*  																				 */
/*																					 */
/*************************************************************************************/

#include "cudaCCW.h"
#include "cudaMesh.h"

REAL host_constData[13];

/* Initialize Geometric primitives */
void cudaexactinit()
{
	REAL half;
	REAL check, lastcheck;
	int every_other;

	every_other = 1;
	half = 0.5;
	host_constData[1] /*epsilon*/ = 1.0;
	host_constData[0] /*splitter*/ = 1.0;
	check = 1.0;
	/* Repeatedly divide `epsilon' by two until it is too small to add to    */
	/*   one without causing roundoff.  (Also check if the sum is equal to   */
	/*   the previous sum, for machines that round up instead of using exact */
	/*   rounding.  Not that this library will work on such machines anyway. */
	do {
		lastcheck = check;
		host_constData[1] /*epsilon*/ *= half;
		if (every_other) {
			host_constData[0] /*splitter*/ *= 2.0;
		}
		every_other = !every_other;
		check = 1.0 + host_constData[1] /*epsilon*/;
	} while ((check != 1.0) && (check != lastcheck));
	host_constData[0] /*splitter*/ += 1.0;

	/* Error bounds for orientation and incircle tests. */
	host_constData[2] /*resulterrbound*/ = (3.0 + 8.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[3] /*ccwerrboundA*/ = (3.0 + 16.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[4] /*ccwerrboundB*/ = (2.0 + 12.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[5] /*ccwerrboundC*/ = (9.0 + 64.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	host_constData[6] /*o3derrboundA*/ = (7.0 + 56.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[7] /*o3derrboundB*/ = (3.0 + 28.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[8] /*o3derrboundC*/ = (26.0 + 288.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	host_constData[9] /*iccerrboundA*/ = (10.0 + 96.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[10] /*iccerrboundB*/ = (4.0 + 48.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	host_constData[11] /*iccerrboundC*/ = (44.0 + 576.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;
	//host_constData[12] /*isperrboundA*/ = (16.0 + 224.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	//host_constData[13] /*isperrboundB*/ = (5.0 + 72.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/;
	//host_constData[14] /*isperrboundC*/ = (71.0 + 1408.0 * host_constData[1] /*epsilon*/) * host_constData[1] /*epsilon*/ * host_constData[1] /*epsilon*/;

	// Calculate the two static filters for orient3d() and insphere() tests.
	// Added by H. Si, 2012-08-23.

	// Copy to const memory
	cudaMemcpyToSymbol(constData, host_constData, 13 * sizeof(REAL));
}

void gpuMemoryCheck()
{
	cudaDeviceSynchronize();
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	if (cudaSuccess != cuda_status)
	{
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}
	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

/********* Mesh manipulation primitives begin here							 *********/
/**																					**/
/**																					**/

__device__ int findIncidentOri(int * d_trianglelist, int tri1, int tri2)
{
	if( tri1 < 0 || tri2 < 0 )
		return -1;
	int inc0=-1,inc1=-1;
	int tri1_p[3] = {
		d_trianglelist[3*tri1],
		d_trianglelist[3*tri1+1],
		d_trianglelist[3*tri1+2]
	};
	int tri2_p[3] = {
		d_trianglelist[3*tri2],
		d_trianglelist[3*tri2+1],
		d_trianglelist[3*tri2+2]
	};

	// incident edge
	int count = 0;
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			if( tri1_p[i] == tri2_p[j] )
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
	
	if(inc0 == -1 || inc1 == -1) // not found
		return -1;

	// orientation
	int differ = inc0 - inc1;
	int index;
	if ( differ == -1 || differ == 2 )
		index =  inc0;
	else
		index =  inc1;

	if(index==0)
		return 2;
	else if(index==1)
		return 0;
	else if(index==2)
		return 1;
	else
		return -1;
}

__device__ int decode_tri(int tri)            
{
	if (tri >=0)
		return (tri >> 2);
	else
		return -1;
}
__device__ int decode_ori(int tri)
{
	return (tri & 3);
}

__device__ int encode_tri(int tri, int ori)
{
	if (tri >=0 )
		return ((tri<< 2) | (ori));
	else
		return -1;
}

__device__ uint64 encodeUInt64Priority(int priority, int index)
{
	return (((uint64)priority) << 32) + index;
}

__device__ int getUInt64PriorityIndex(uint64 priority)
{
	return (priority & 0xFFFFFFFF);
}

__device__ int getUInt64Priority(uint64 priority)
{
	return (priority >> 32);
}

__device__ bool checkseg4encroach(
	REAL2 vOrg,
	REAL2 vDest,
	REAL2 vApex,
	REAL theta,
	int run_mode
)
{
	REAL goodcoss = cos(theta * PI / 180.0);
	goodcoss *= goodcoss;
	REAL dotproduct = (vOrg.x - vApex.x)*(vDest.x - vApex.x) +
		(vOrg.y - vApex.y)*(vDest.y - vApex.y);

	if(dotproduct < 0.0) // angle > 90
	{
		// here, we use diametral lens to speedup the algorithm
		if( run_mode || dotproduct * dotproduct >=
			(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
			((vOrg.x - vApex.x)*(vOrg.x - vApex.x) + (vOrg.y - vApex.y)*(vOrg.y - vApex.y)) *
			((vDest.x - vApex.x)*(vDest.x - vApex.x) + (vDest.y - vApex.y)*(vDest.y - vApex.y)) )
			return true;
	}

	return false;
}

/**																					**/
/**																					**/
/********* Mesh manipulation primitives end here							 *********/

/********* Memory management routines begin here							 *********/
/**																					**/
/**																					**/

__device__ REAL kernelComputeSmallestAngle(REAL2 pApex, REAL2 pOrg, REAL2 pDest, int *longest, REAL * longestedge, REAL * shortestedge, int * shortestOri)
{
	REAL dxod, dyod, dxda, dyda, dxao, dyao;
	REAL dxod2, dyod2, dxda2, dyda2, dxao2, dyao2;
	REAL lenOrg, lenDest, lenApex;
	REAL angle_smallest;	
	
	dxod = pOrg.x - pDest.x;
	dyod = pOrg.y - pDest.y;
	dxda = pDest.x - pApex.x;
	dyda = pDest.y - pApex.y;
	dxao = pApex.x - pOrg.x;
	dyao = pApex.y - pOrg.y;
	dxod2 = dxod * dxod;
	dyod2 = dyod * dyod;
	dxda2 = dxda * dxda;
	dyda2 = dyda * dyda;
	dxao2 = dxao * dxao;
	dyao2 = dyao * dyao;

	/* Find the lengths of the triangle's three edges. */
	lenOrg = dxod2 + dyod2;
	lenDest = dxda2 + dyda2;
	lenApex = dxao2 + dyao2;

	// angle is actually cos^2
	if (lenOrg < lenDest && lenOrg < lenApex) 
	{
		*shortestedge	= lenOrg;
		*shortestOri = 0;
		angle_smallest	= dxda * dxao + dyda * dyao;
	}
	else if (lenDest < lenOrg && lenDest < lenApex)
	{
		*shortestedge	= lenDest; 
		*shortestOri = 1;
		angle_smallest	= dxod * dxao + dyod * dyao;
	}
	else 
	{
		*shortestedge	= lenApex;
		*shortestOri = 2;
		angle_smallest	= dxod * dxda + dyod * dyda;
	}

	angle_smallest = angle_smallest * angle_smallest * (*shortestedge) /
		( lenOrg * lenDest * lenApex );

	if (lenOrg > lenDest && lenOrg > lenApex)
	{
		*longest = 0;
		*longestedge = lenOrg;
	}
	else if (lenDest > lenOrg && lenDest > lenApex)
	{
		*longest = 1;
		*longestedge = lenDest;
	}
	else
	{
		*longest = 2;
		*longestedge = lenApex;
	}

	return angle_smallest;
	
}

__global__ void kernelMarkBadTriangles(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int * d_trianglelist,
	int * d_neighborlist,
	int * d_segmentlist,
	int * d_subseg2seg,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	int * d_active,
	int numberofactive,
	REAL theta,
	REAL size
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int index = d_active[pos];
	if(!d_TStatus[index].isCheck())
		return;

	d_TStatus[index].setCheck(false);
	d_TStatus[index].setNull(false);

	int p[3] = {
		d_trianglelist[3*index],
		d_trianglelist[3*index+1],
		d_trianglelist[3*index+2]
	};

	REAL2 v[3] = {
		d_pointlist[p[0]],
		d_pointlist[p[1]],
		d_pointlist[p[2]]
	};

	int longest;
	REAL longestedge;
	int shortest;
	REAL shortestedge;
	REAL smallestCos2 = kernelComputeSmallestAngle(v[0], v[1], v[2], &longest, &longestedge, &shortestedge, &shortest);

	if (size > 0.0 && longestedge > size*size) // Imperatively bad
	{
		d_TStatus[index].setBad(true);
		return;
	}

	REAL goodCos2 = cos(theta * PI/180.0)*cos(theta * PI/180.0);
	if (smallestCos2 <= goodCos2)	// Good triangle
	{
		d_TStatus[index].setBad(false); 
		return;
	}
	else // has bad angle 
	{
		// first check if two points are on segments
		// if not
		if(d_tri2subseg[3*index+shortest] == -1)
		{
			int eOrg,eDest; // shortest edge endpoints
			eOrg = p[(shortest+1)%3];
			eDest = p[(shortest+2)%3];

			// both points should be midpoint
			if(d_PStatus[eOrg].isSegmentSplit() && d_PStatus[eDest].isSegmentSplit())
			{
				int s1 = -1,s2 = -1; // record two segments
				int pOrg1,pDest1,pOrg2,pDest2, common; // record the segment endpoints that include the shortest edge endpoints
				int curTri,curOri;
				// search the segment that include eOrg
				curTri = index;
				curOri = shortest;
				do{
					int otri = d_neighborlist[3*curTri+curOri];
					curTri = decode_tri(otri);
					curOri = decode_ori(otri);
					curOri = (curOri+1)%3;
					s1 = d_tri2subseg[3*curTri+curOri];
				}while(s1 == -1);
				// search the segment that include eDest
				curTri = index;
				curOri = shortest;
				do{
					int otri = d_neighborlist[3*curTri+curOri];
					curTri = decode_tri(otri);
					curOri = decode_ori(otri);
					curOri = (curOri+2)%3;
					s2 = d_tri2subseg[3*curTri+curOri];
				}while(s2 == -1);
				// get the segments that include s1 and s2
				s1 = d_subseg2seg[s1];
				s2 = d_subseg2seg[s2];
				// get the endpoints
				pOrg1 = d_segmentlist[2*s1];
				pDest1 = d_segmentlist[2*s1+1];
				pOrg2 = d_segmentlist[2*s2];
				pDest2 = d_segmentlist[2*s2+1];
				// test if s1 and s2 have common point
				common = -1;
				if(pOrg1 == pOrg2 || pOrg1 == pDest2)
					common = pOrg1;
				else if (pDest1 == pOrg2 || pDest1 == pDest2)
					common = pDest1;
				// if have common point
				if(common != -1)
				{
					// test if equal distant
					REAL dist1,dist2;
					REAL2 vOrg = v[(shortest+1)%3];
					REAL2 vDest = v[(shortest+2)%3];
					REAL2 vCom = d_pointlist[common];
					dist1 = (vCom.x - vOrg.x)*(vCom.x - vOrg.x) + (vCom.y - vOrg.y)*(vCom.y - vOrg.y);
					dist2 = (vCom.x - vDest.x)*(vCom.x - vDest.x) + (vCom.y - vDest.y)*(vCom.y - vDest.y);
					// if equal
					if ((dist1 < 1.001 * dist2) && (dist1 > 0.999 * dist2)) 
					{
						d_TStatus[index].setBad(false); 
						return;
					}
				}
			}
		}
	}

	d_TStatus[index].setBad(true);
}

int updateActiveListToBadTriangles
(
Real2D		&t_pointlist,
PStatusD	&t_PStatus,
IntD		&t_trianglelist,
IntD		&t_neighborlist,
IntD		&t_segmentlist,
IntD		&t_subseg2seg,
IntD		&t_tri2subseg,
TStatusD	&t_TStatus, 
IntD		&t_active0,
IntD		&t_active1,
int         numberoftriangles,
REAL theta,
REAL size
)
{
	int numberofcheck;
	int numberofbad;

	int numberofblocks;

	// Update active list 0 (The triangles that need to check or are bad)
	thrust::counting_iterator<int> first0(0);
	thrust::counting_iterator<int> last0 = first0 + numberoftriangles;

	t_active0.resize(numberoftriangles);
			
	t_active0.erase(
		thrust::copy_if( 
			first0, 
			last0, 
			t_TStatus.begin(), 
			t_active0.begin(), 
			isCheckOrBadTriangle() ),
		t_active0.end() );

	numberofcheck = t_active0.size();

	// Mark bad triangles
	numberofblocks = (ceil)((float)numberofcheck / BLOCK_SIZE);
	kernelMarkBadTriangles<<<numberofblocks,BLOCK_SIZE>>>(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_PStatus[0]),
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_segmentlist[0]),
		thrust::raw_pointer_cast(&t_subseg2seg[0]),
		thrust::raw_pointer_cast(&t_tri2subseg[0]),
		thrust::raw_pointer_cast(&t_TStatus[0]),
		thrust::raw_pointer_cast(&t_active0[0]),
		numberofcheck,
		theta,
		size);

	// Update active list 1 (the bad triangle list)
	thrust::counting_iterator<int> first1(0);
	thrust::counting_iterator<int> last1 = first1 + numberoftriangles;

	t_active1.resize(numberoftriangles);

	t_active1.erase(
		thrust::copy_if( 
			first1, 
			last1, 
			t_TStatus.begin(), 
			t_active1.begin(), 
			isBadTriangle() ),
		t_active1.end() );

	numberofbad = t_active1.size();

	return numberofbad;
}

/**																					**/
/**																					**/
/********* Memory management routines end here								 *********/

/********* Mesh transformation routines begin here							 *********/
/**																					**/
/**																					**/

__device__ void markFan(
	int * d_trianglelist,
	int * d_neighborlist,
	TStatus * d_TStatus,
	int triIndex,
	int triOri
)
{
	int curIndex, curOri;		
	int nextOtri;

	// scan triangles in ccw order
	curIndex = triIndex;
	curOri = triOri;

	do{
		d_TStatus[curIndex].setFlip(true);

		nextOtri = d_neighborlist[curIndex * 3 + (curOri + 2) % 3];
		curIndex = decode_tri(nextOtri);
		curOri = decode_ori(nextOtri);
	}while( curIndex > -1 && curIndex != triIndex);
}

__global__ void kernelMarkReduantPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int * d_trianglelist,
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_delmarker,
	int * d_dellist,
	int numberofdels
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofdels)
		return;

	d_delmarker[pos] = -1; // reset

	int sindex = d_dellist[pos];

	int pOrg,pDest,pApex;
	REAL2 vOrg, vDest, vApex;
	int otri,triIndex,triOri;

	// check the first triangle
	otri = d_subseg2tri[sindex];
	triIndex = decode_tri(otri);
	triOri = decode_ori(otri);

	pOrg = d_trianglelist[3*triIndex + (triOri+1)%3];
	pDest = d_trianglelist[3*triIndex + (triOri+2)%3];
	pApex = d_trianglelist[3*triIndex + triOri];
	vOrg = d_pointlist[pOrg];
	vDest = d_pointlist[pDest];
	vApex = d_pointlist[pApex];

	// free steiner points and inside the diametral circle
	if( d_PStatus[pApex].isSteiner() && !d_PStatus[pApex].isSegmentSplit() && 
		((vOrg.x - vApex.x)*(vDest.x - vApex.x) + (vOrg.y - vApex.y)*(vDest.y - vApex.y) < 0.0) )
	{
		// mark this point as reduntant
		d_PStatus[pApex].setRedundant();

		// mark incident triangles for flippable
		markFan(d_trianglelist,d_neighborlist,d_TStatus,triIndex, (triOri+2)%3);

		// mark subsegs for further deletion
		d_delmarker[pos] = sindex;
	}

	// check the triangle on the other side
	otri = d_neighborlist[3*triIndex + triOri];
	if(otri != -1)
	{
		triIndex = decode_tri(otri);
		triOri = decode_ori(otri);
		pApex = d_trianglelist[3*triIndex + triOri];
		vApex = d_pointlist[pApex];

		// free steiner points and inside the diametral circle
		if( d_PStatus[pApex].isSteiner() && !d_PStatus[pApex].isSegmentSplit() && 
			((vOrg.x - vApex.x)*(vDest.x - vApex.x) + (vOrg.y - vApex.y)*(vDest.y - vApex.y) < 0.0) )
		{
			// mark this point as reduntant
			d_PStatus[pApex].setRedundant();

			// mark incident triangles for flippable
			markFan(d_trianglelist,d_neighborlist,d_TStatus,triIndex, (triOri+2)%3);

			// mark subsegs for further deletion
			d_delmarker[pos] = sindex;
		}
	}
}

void markReduntantPoints(
	Real2D	 &t_pointlist,
	PStatusD &t_PStatus,
	IntD	 &t_trianglelist,
	IntD	 &t_neighborlist,
	TStatusD &t_TStatus,
	IntD	 &t_subseg2tri,
	IntD	 &t_delmarker,
	IntD	 &t_dellist,
	int	numberofdels
)
{
	int numberofblocks = (ceil)((float)numberofdels / BLOCK_SIZE);
	kernelMarkReduantPoints<<<numberofblocks, BLOCK_SIZE>>>(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_PStatus[0]),
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_TStatus[0]),
		thrust::raw_pointer_cast(&t_subseg2tri[0]),
		thrust::raw_pointer_cast(&t_delmarker[0]),
		thrust::raw_pointer_cast(&t_dellist[0]),
		numberofdels);
}

__device__ int negate(int index)
{
	return (-(index) - 3);
}

__device__ void atomic_compete_two(int index0, int index1, int ori, int * d_flipBy, TStatus * d_TStatus)
{
	d_TStatus[index0].setFlipOri((ori+2)%3);
	atomicMin(d_flipBy + index0, index0);
	atomicMin(d_flipBy + index1, index0);
}

__device__ void atomic_compete_three(int index0, int index1, int index2, int ori, int * d_flipBy, TStatus * d_TStatus)
{
	d_TStatus[index0].setFlipOri(3+ori);
	atomicMin(d_flipBy + index0, index0);
	atomicMin(d_flipBy + index1, index0);
	atomicMin(d_flipBy + index2, index0);
}

__device__ void atomic_compete_four(int index0, int index1, int index2, int index3, int ori, int * d_flipBy, TStatus * d_TStatus)
{
	d_TStatus[index0].setFlipOri(6+ori);
	atomicMin(d_flipBy + index0, index0);
	atomicMin(d_flipBy + index1, index0);
	atomicMin(d_flipBy + index2, index0);
	atomicMin(d_flipBy + index3, index0);
}

__global__ void kernelInitFlipOri
(
	TStatus	  *d_TStatus,
	int	numberoftriangle
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberoftriangle ) 
		return;

	bool tag = false;
	if(d_TStatus[pos].isNull())
		tag = true;

	d_TStatus[pos].setFlipOri(15); // indicate they are losers or undefined (at the beginning)

}

__device__ int deleteOneFromTwo(int pApex, int pOpp, PStatus * d_PStatus)
{
	// Both are edge split 
	// --> keep both
	if (d_PStatus[pApex].isSegmentSplit() && d_PStatus[pOpp].isSegmentSplit())
		return -1;

	// Both are new steiner points
	// --> delete the higher index one
	if (d_PStatus[pApex].isNew() && d_PStatus[pOpp].isNew())
	{
		if (d_PStatus[pApex].isSegmentSplit())
			return 1;

		if (d_PStatus[pOpp].isSegmentSplit())
			return 0;

		if (pApex < pOpp)
			return 1;
		else
			return 0;
	}

	return -1;
}

__global__ void kernelCheckFlipping
(
	REAL2	  *d_pointlist,
	PStatus	  *d_PStatus,
	int		  *d_trianglelist, 
	int		  *d_neighborlist,
	int		  *d_tri2subseg,
	TStatus   *d_TStatus,
	int		  *d_flipBy,
	int		  *d_active, 
	int		  numberofactive
 ) 
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofactive ) return ;

	int index = d_active[pos];

	int tri;
	int pTri, pOri; 

	int min = MAXINT;
	int min_original;		
		
	int p[3] = {
		d_trianglelist[index*3 + 0], 
		d_trianglelist[index*3 + 1],
		d_trianglelist[index*3 + 2]
	}; 
	
	for ( int i = 0; i < 3; ++i ) 
	{
		if ( d_PStatus[p[i]].isReduntant() && p[i] < min ) // is redundant
		{
			min = p[i]; 
		}
	}

	const REAL2 v[3] = { 
		d_pointlist[p[0]], 
		d_pointlist[p[1]], 
		d_pointlist[p[2]] };

	for ( int i = 0; i < 3; i++ )//three edges
	{
		if ( min == p[i] ) continue; 
		if ( d_tri2subseg[3*index + i] != -1 ) continue; // segment cannot flip

		tri  = d_neighborlist[3*index+i];
		pTri = decode_tri(tri);
		pOri = decode_ori(tri);

		if ( index < pTri || ( pTri >=0 && !d_TStatus[pTri].isFlip()) )// index < pTri or pTri is unmarked
		{			

			int pOpp = d_trianglelist[pTri * 3 + pOri];

			// Flipping decreases min's degree --> Flop
			if ( d_PStatus[pOpp].isReduntant() &&  pOpp < min)  
				continue; 

			REAL2 vOpp	= d_pointlist[pOpp];

			int org  = (i + 1) % 3;
			int dest = (i + 2) % 3;
			int apex = i;

			//Flop part
			if ( min != MAXINT )	// Involve a point to be deleted
			{
				// Check if this is a 3-1 flip or 4-1 flip
				int deleteOri, n1, n2;
				if ( min == p[org] ) 
				{
					deleteOri = (i + 2) % 3;
					n1 = d_neighborlist[index*3 + dest];
					n2 = d_neighborlist[pTri*3 + (pOri + 1) % 3]; 
				}
				else
				{
					deleteOri = i;
					n1 = d_neighborlist[index*3 + org]; 
					n2 = d_neighborlist[pTri*3 + (pOri + 2) % 3]; 
				}

				if( decode_tri(n1) == decode_tri(n2) ) // 3-1 flip
				{
					atomic_compete_three(index, pTri, decode_tri(n1), deleteOri, d_flipBy, d_TStatus);							
					return;
				}
				else // degree >= 4
				{
					int n3,n1_tri,n1_ori,n2_tri,n2_ori;
					int n1_pOpp,n2_pOpp;
					REAL2 n1_vOpp,n2_vOpp;

					n1_tri = decode_tri(n1);
					n1_ori = decode_ori(n1);
					n2_tri = decode_tri(n2);
					n2_ori = decode_ori(n2);

					n1_pOpp = d_trianglelist[3*n1_tri+n1_ori];
					n2_pOpp = d_trianglelist[3*n2_tri+n2_ori];

					n1_vOpp = d_pointlist[n1_pOpp];
					n2_vOpp = d_pointlist[n2_pOpp];

					if( min == p[org] )
						n3 = d_neighborlist[3*n1_tri + (n1_ori+2)%3];
					else
						n3 = d_neighborlist[3*n1_tri + (n1_ori+1)%3];

					if(decode_tri(n3) == decode_tri(n2)) // 4-degree reduntant point
					{
						// check if it has collinear diagonal

						// outward direction
						if( min == p[org])
						{
							REAL t0,t1;
							bool f0,f1;

							// index and pTri diagonal
							REAL d = cuda_fast(v[apex], v[org], vOpp);
#ifndef GQM2D_INEXACT
							if ( d == 0)
								d = cuda_ccw(v[apex], v[org], vOpp);
#endif
						
							if(d == 0)
							{
								// check if index and n1 is flippable
								t0 = cuda_fast(v[dest], v[apex], n1_vOpp);
#ifndef GQM2D_INEXACT
								if(t0 == 0)
									t0 = cuda_ccw(v[dest], v[apex], n1_vOpp);
#endif

								t1 = cuda_fast(v[dest], v[org], n1_vOpp);
#ifndef GQM2D_INEXACT
								if(t1 == 0)
									t1 = cuda_ccw(v[dest], v[org], n1_vOpp);
#endif

								if( t0 > 0 && t1 < 0)
									f0 = true;
								else
									f0 = false;

								// check if pTri and n2 if flippable
								t0 = cuda_fast(v[dest], v[org], n2_vOpp);
#ifndef GQM2D_INEXACT
								if(t0 == 0)
									t0 = cuda_ccw(v[dest], v[org], n2_vOpp);
#endif

								t1 = cuda_fast(v[dest], vOpp, n2_vOpp);
#ifndef GQM2D_INEXACT
								if(t1 == 0)
									t1 = cuda_ccw(v[dest], vOpp, n2_vOpp);
#endif

								if( t0 > 0 && t1 < 0)
									f1 = true;
								else
									f1 = false;

								if(!f0 && !f1) // both are unflippable, do 4-1 flip
								{
									if(d_PStatus[n1_pOpp].isReduntant() &&  n1_pOpp < min) // gurantees the smallest index reduntant point
										continue;
									deleteOri = i;
									atomic_compete_four(index,pTri,n1_tri,n2_tri,deleteOri,d_flipBy,d_TStatus);
									return;
								}
							}
						}
						else // inward direction
						{
							REAL t0,t1;

							bool f0,f1;

							// index and pTri: diagonal 0
							REAL d = cuda_fast(v[apex], v[dest], vOpp);
#ifndef GQM2D_INEXACT
							if ( d == 0)
								d = cuda_ccw(v[apex], v[dest], vOpp);
#endif
						
							if(d == 0)
							{
								// check if index and n1 is flippable
								t0 = cuda_fast(v[org], v[dest], n1_vOpp);
#ifndef GQM2D_INEXACT
								if(t0 == 0)
									t0 = cuda_ccw(v[org], v[dest], n1_vOpp);
#endif

								t1 = cuda_fast(v[org], v[apex], n1_vOpp);
#ifndef GQM2D_INEXACT
								if(t1 == 0)
									t1 = cuda_ccw(v[org], v[apex], n1_vOpp);
#endif

								if( t0 > 0 && t1 < 0)
									f0 = true;
								else
									f0 = false;

								// check if pTri and n2 if flippable
								t0 = cuda_fast(v[org], vOpp, n2_vOpp);
#ifndef GQM2D_INEXACT
								if(t0 == 0)
									t0 = cuda_ccw(v[org], vOpp, n2_vOpp);
#endif

								t1 = cuda_fast(v[org], v[dest], n2_vOpp);
#ifndef GQM2D_INEXACT
								if(t1 == 0)
									t1 = cuda_ccw(v[org], v[dest], n2_vOpp);
#endif

								if( t0 > 0 && t1 < 0)
									f1 = true;
								else
									f1 = false;

								if(!f0 && !f1) // both are unflippable, do 4-2 flip
								{
									if(d_PStatus[n1_pOpp].isReduntant() &&  n1_pOpp < min) // gurantees the smallest index reduntant point
										continue;
									deleteOri = i;
									atomic_compete_four(index,pTri,n1_tri,n2_tri,deleteOri,d_flipBy,d_TStatus);
									return;
								}
							}
						}
					}
				}

				// Check if this is a 2-2 flippable
				REAL t1 = cuda_fast(v[apex], v[org], vOpp);
#ifndef GQM2D_INEXACT
				if ( t1 == 0)
					t1 = cuda_ccw(v[apex], v[org], vOpp);
#endif

				if (t1 <= 0) continue;	// Unflippable

				REAL t2 = cuda_fast(v[apex], v[dest], vOpp);
#ifndef GQM2D_INEXACT
				if (t2 == 0)
					t2 = cuda_ccw(v[apex], v[dest], vOpp);
#endif

				if (t2 >= 0) continue; // Unflippable

				atomic_compete_two(index, pTri, i, d_flipBy, d_TStatus);					
				return;						
			}

			//flip part			
			//check DT
			REAL ret = cuda_inCircle(v[apex], v[org], v[dest], vOpp);
#ifndef GQM2D_INEXACT
			if ( ret == 0)
				ret = cuda_inCircle_exact(v[apex], v[org], v[dest], vOpp);
#endif

			REAL ret_opp = cuda_inCircle(v[org], vOpp, v[dest], v[apex]);
#ifndef GQM2D_INEXACT
			if ( ret_opp == 0)
				ret_opp = cuda_inCircle_exact(v[org], vOpp, v[dest], v[apex]);
#endif

			REAL ret_after = cuda_inCircle(v[apex], v[org], vOpp, v[dest]); // after flip
#ifndef GQM2D_INEXACT
			if ( ret_after == 0)
				ret_after = cuda_inCircle_exact(v[apex], v[org], vOpp, v[dest]);
#endif

			REAL ret_after_opp = cuda_inCircle(vOpp,v[dest],v[apex],v[org]);
#ifndef GQM2D_INEXACT
			if ( ret_after_opp == 0)
				ret_after_opp = cuda_inCircle_exact(vOpp,v[dest],v[apex],v[org]);
#endif

			if ( ret > 0 && ret_opp > 0 && ret_after < 0 && ret_after_opp < 0 ) // prevent calculation error
			{								
				
				int toDelete = deleteOneFromTwo(p[apex], pOpp, d_PStatus);

				switch(toDelete)
				{
				case -1: // Do normal flipping
					atomic_compete_two(index, pTri, i, d_flipBy, d_TStatus);		
					break; 
				case 0: 
					toDelete = negate(encode_tri(index, (i + 2) % 3));
					atomicMin( &d_flipBy[index], toDelete ); 
					break;
				case 1: 
					toDelete = negate(encode_tri(pTri, (pOri + 2) % 3));
					atomicMin( &d_flipBy[index], toDelete ); 
					atomicMin( &d_flipBy[pTri], toDelete ); 
					break; 
				}

				return ;
				

			}//end of DT check;
		}//end of if(x < pTri || (pTri >=0 && mark[pTri] != 0))
	}//end of three edge;	

	d_TStatus[index].setFlip(false); // unmark this triangle if nothing to do
}

__device__ int deleteOneFromTwo(int pApex, int pOpp, PStatus * d_PStatus, REAL* d_pointpriority)
{
	// Both are edge split 
	// --> keep both
	if (d_PStatus[pApex].isSegmentSplit() && d_PStatus[pOpp].isSegmentSplit())
		return -1;

	// Both are new steiner points
	// --> delete the higher index one
	if (d_PStatus[pApex].isNew() && d_PStatus[pOpp].isNew())
	{
		if (d_PStatus[pApex].isSegmentSplit())
			return 1;

		if (d_PStatus[pOpp].isSegmentSplit())
			return 0;

		if (d_pointpriority[pApex] > d_pointpriority[pOpp])
			return 1;
		else
			return 0;
	}

	return -1;
}

__global__ void kernelCheckFlipping_New
(
	REAL2	  *d_pointlist,
	PStatus	  *d_PStatus,
	REAL	  *d_pointpriority,
	int		  *d_trianglelist,
	int		  *d_neighborlist,
	int		  *d_tri2subseg,
	TStatus   *d_TStatus,
	int		  *d_flipBy,
	int		  *d_active,
	int		  numberofactive
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= numberofactive) return;

	int index = d_active[pos];

	int tri;
	int pTri, pOri;

	int min = MAXINT;
	int min_original;

	int p[3] = {
		d_trianglelist[index * 3 + 0],
		d_trianglelist[index * 3 + 1],
		d_trianglelist[index * 3 + 2]
	};

	for (int i = 0; i < 3; ++i)
	{
		if (d_PStatus[p[i]].isReduntant() && p[i] < min) // is redundant
		{
			min = p[i];
		}
	}

	const REAL2 v[3] = {
		d_pointlist[p[0]],
		d_pointlist[p[1]],
		d_pointlist[p[2]] };

	for (int i = 0; i < 3; i++)//three edges
	{
		if (min == p[i]) continue;
		if (d_tri2subseg[3 * index + i] != -1) continue; // segment cannot flip

		tri = d_neighborlist[3 * index + i];
		pTri = decode_tri(tri);
		pOri = decode_ori(tri);

		if (index < pTri || (pTri >= 0 && !d_TStatus[pTri].isFlip()))// index < pTri or pTri is unmarked
		{

			int pOpp = d_trianglelist[pTri * 3 + pOri];

			// Flipping decreases min's degree --> Flop
			if (d_PStatus[pOpp].isReduntant() && pOpp < min)
				continue;

			REAL2 vOpp = d_pointlist[pOpp];

			int org = (i + 1) % 3;
			int dest = (i + 2) % 3;
			int apex = i;

			//Flop part
			if (min != MAXINT)	// Involve a point to be deleted
			{
				// Check if this is a 3-1 flip or 4-1 flip
				int deleteOri, n1, n2;
				if (min == p[org])
				{
					deleteOri = (i + 2) % 3;
					n1 = d_neighborlist[index * 3 + dest];
					n2 = d_neighborlist[pTri * 3 + (pOri + 1) % 3];
				}
				else
				{
					deleteOri = i;
					n1 = d_neighborlist[index * 3 + org];
					n2 = d_neighborlist[pTri * 3 + (pOri + 2) % 3];
				}

				if (decode_tri(n1) == decode_tri(n2)) // 3-1 flip
				{
					atomic_compete_three(index, pTri, decode_tri(n1), deleteOri, d_flipBy, d_TStatus);
					return;
				}
				else // degree >= 4
				{
					int n3, n1_tri, n1_ori, n2_tri, n2_ori;
					int n1_pOpp, n2_pOpp;
					REAL2 n1_vOpp, n2_vOpp;

					n1_tri = decode_tri(n1);
					n1_ori = decode_ori(n1);
					n2_tri = decode_tri(n2);
					n2_ori = decode_ori(n2);

					n1_pOpp = d_trianglelist[3 * n1_tri + n1_ori];
					n2_pOpp = d_trianglelist[3 * n2_tri + n2_ori];

					n1_vOpp = d_pointlist[n1_pOpp];
					n2_vOpp = d_pointlist[n2_pOpp];

					if (min == p[org])
						n3 = d_neighborlist[3 * n1_tri + (n1_ori + 2) % 3];
					else
						n3 = d_neighborlist[3 * n1_tri + (n1_ori + 1) % 3];

					if (decode_tri(n3) == decode_tri(n2)) // 4-degree reduntant point
					{
						// check if it has collinear diagonal

						// outward direction
						if (min == p[org])
						{
							REAL t0, t1;
							bool f0, f1;

							// index and pTri diagonal
							REAL d = cuda_fast(v[apex], v[org], vOpp);
#ifndef GQM2D_INEXACT
							if (d == 0)
								d = cuda_ccw(v[apex], v[org], vOpp);
#endif

							if (d == 0)
							{
								// check if index and n1 is flippable
								t0 = cuda_fast(v[dest], v[apex], n1_vOpp);
#ifndef GQM2D_INEXACT
								if (t0 == 0)
									t0 = cuda_ccw(v[dest], v[apex], n1_vOpp);
#endif

								t1 = cuda_fast(v[dest], v[org], n1_vOpp);
#ifndef GQM2D_INEXACT
								if (t1 == 0)
									t1 = cuda_ccw(v[dest], v[org], n1_vOpp);
#endif

								if (t0 > 0 && t1 < 0)
									f0 = true;
								else
									f0 = false;

								// check if pTri and n2 if flippable
								t0 = cuda_fast(v[dest], v[org], n2_vOpp);
#ifndef GQM2D_INEXACT
								if (t0 == 0)
									t0 = cuda_ccw(v[dest], v[org], n2_vOpp);
#endif

								t1 = cuda_fast(v[dest], vOpp, n2_vOpp);
#ifndef GQM2D_INEXACT
								if (t1 == 0)
									t1 = cuda_ccw(v[dest], vOpp, n2_vOpp);
#endif

								if (t0 > 0 && t1 < 0)
									f1 = true;
								else
									f1 = false;

								if (!f0 && !f1) // both are unflippable, do 4-1 flip
								{
									if (d_PStatus[n1_pOpp].isReduntant() && n1_pOpp < min) // gurantees the smallest index reduntant point
										continue;
									deleteOri = i;
									atomic_compete_four(index, pTri, n1_tri, n2_tri, deleteOri, d_flipBy, d_TStatus);
									return;
								}
							}
						}
						else // inward direction
						{
							REAL t0, t1;

							bool f0, f1;

							// index and pTri: diagonal 0
							REAL d = cuda_fast(v[apex], v[dest], vOpp);
#ifndef GQM2D_INEXACT
							if (d == 0)
								d = cuda_ccw(v[apex], v[dest], vOpp);
#endif

							if (d == 0)
							{
								// check if index and n1 is flippable
								t0 = cuda_fast(v[org], v[dest], n1_vOpp);
#ifndef GQM2D_INEXACT
								if (t0 == 0)
									t0 = cuda_ccw(v[org], v[dest], n1_vOpp);
#endif

								t1 = cuda_fast(v[org], v[apex], n1_vOpp);
#ifndef GQM2D_INEXACT
								if (t1 == 0)
									t1 = cuda_ccw(v[org], v[apex], n1_vOpp);
#endif

								if (t0 > 0 && t1 < 0)
									f0 = true;
								else
									f0 = false;

								// check if pTri and n2 if flippable
								t0 = cuda_fast(v[org], vOpp, n2_vOpp);
#ifndef GQM2D_INEXACT
								if (t0 == 0)
									t0 = cuda_ccw(v[org], vOpp, n2_vOpp);
#endif

								t1 = cuda_fast(v[org], v[dest], n2_vOpp);
#ifndef GQM2D_INEXACT
								if (t1 == 0)
									t1 = cuda_ccw(v[org], v[dest], n2_vOpp);
#endif

								if (t0 > 0 && t1 < 0)
									f1 = true;
								else
									f1 = false;

								if (!f0 && !f1) // both are unflippable, do 4-2 flip
								{
									if (d_PStatus[n1_pOpp].isReduntant() && n1_pOpp < min) // gurantees the smallest index reduntant point
										continue;
									deleteOri = i;
									atomic_compete_four(index, pTri, n1_tri, n2_tri, deleteOri, d_flipBy, d_TStatus);
									return;
								}
							}
						}
					}
				}

				// Check if this is a 2-2 flippable
				REAL t1 = cuda_fast(v[apex], v[org], vOpp);
#ifndef GQM2D_INEXACT
				if (t1 == 0)
					t1 = cuda_ccw(v[apex], v[org], vOpp);
#endif

				if (t1 <= 0) continue;	// Unflippable

				REAL t2 = cuda_fast(v[apex], v[dest], vOpp);
#ifndef GQM2D_INEXACT
				if (t2 == 0)
					t2 = cuda_ccw(v[apex], v[dest], vOpp);
#endif

				if (t2 >= 0) continue; // Unflippable

				atomic_compete_two(index, pTri, i, d_flipBy, d_TStatus);
				return;
			}

			//flip part			
			//check DT
			REAL ret = cuda_inCircle(v[apex], v[org], v[dest], vOpp);
#ifndef GQM2D_INEXACT
			if (ret == 0)
				ret = cuda_inCircle_exact(v[apex], v[org], v[dest], vOpp);
#endif

			REAL ret_opp = cuda_inCircle(v[org], vOpp, v[dest], v[apex]);
#ifndef GQM2D_INEXACT
			if (ret_opp == 0)
				ret_opp = cuda_inCircle_exact(v[org], vOpp, v[dest], v[apex]);
#endif

			REAL ret_after = cuda_inCircle(v[apex], v[org], vOpp, v[dest]); // after flip
#ifndef GQM2D_INEXACT
			if (ret_after == 0)
				ret_after = cuda_inCircle_exact(v[apex], v[org], vOpp, v[dest]);
#endif

			REAL ret_after_opp = cuda_inCircle(vOpp, v[dest], v[apex], v[org]);
#ifndef GQM2D_INEXACT
			if (ret_after_opp == 0)
				ret_after_opp = cuda_inCircle_exact(vOpp, v[dest], v[apex], v[org]);
#endif

			if (ret > 0 && ret_opp > 0 && ret_after < 0 && ret_after_opp < 0) // prevent calculation error
			{

				int toDelete = deleteOneFromTwo(p[apex], pOpp, d_PStatus, d_pointpriority);

				switch (toDelete)
				{
				case -1: // Do normal flipping
					atomic_compete_two(index, pTri, i, d_flipBy, d_TStatus);
					break;
				case 0:
					toDelete = negate(encode_tri(index, (i + 2) % 3));
					atomicMin(&d_flipBy[index], toDelete);
					break;
				case 1:
					toDelete = negate(encode_tri(pTri, (pOri + 2) % 3));
					atomicMin(&d_flipBy[index], toDelete);
					atomicMin(&d_flipBy[pTri], toDelete);
					break;
				}

				return;


			}//end of DT check;
		}//end of if(x < pTri || (pTri >=0 && mark[pTri] != 0))
	}//end of three edge;	

	d_TStatus[index].setFlip(false); // unmark this triangle if nothing to do
}

__global__ void kernelMarkFlipWinner
(
	 PStatus 	*d_PStatus,
	 int		*d_trianglelist,
	 int        *d_neighborlist,
	 TStatus	*d_TStatus,
	 int		*d_flipBy, 
	 int		*d_active, 
	 int		numberofactive
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofactive ) return ;

	int index = d_active[pos];
	
	int ori = d_TStatus[index].getFlipOri(); 

	if ( ori == 15 ) // default value
		return;

    int pOpp, pOppTri, pOppOri, pOrg, pDest, pApex;
	bool isWinner = false;  

    if( d_flipBy[index] == index) 
	{
		if ( ori < 3 ) //2->2 flip
		{		
			pOpp = d_neighborlist[index*3 + (ori + 1) % 3]; 
			pOppTri = decode_tri(pOpp);
			pOppOri = decode_ori(pOpp);

			if ( d_flipBy[pOppTri] == index ) 
			{    // I'm also the one who win the right
				isWinner = true; 
			}	  
		}
		else if(ori < 6)// 3->1 flip
		{	       
			ori -= 3; // x -> t2 -> t1
			ori = (ori + 1) % 3;
			
			int t1     = d_neighborlist[index*3 + ori]; //cw
			int t2     = d_neighborlist[index*3 + (ori + 2) % 3];//ccw
			int t1_ori = decode_ori(t1);
			int t2_ori = decode_ori(t2);
			t1		   = decode_tri(t1);
			t2		   = decode_tri(t2);

			if ( (t1 < 0 || (d_flipBy[t1]  == index)) &&
			    (t2 < 0 || (d_flipBy[t2]  == index) ) )
			{
				isWinner = true; 
			}
		}
		else// 4->2 flip
		{
			ori -= 6;
			
			pOpp = d_neighborlist[index*3 + ori]; 
			pOppTri = decode_tri(pOpp);
			pOppOri = decode_ori(pOpp);

			pOrg	 = d_trianglelist[index*3 + (ori + 1) % 3]; 
			pDest	 = d_trianglelist[index*3 + (ori + 2) % 3]; 
			pApex	 = d_trianglelist[index*3 + ori];
			pOpp	 = d_trianglelist[pOppTri*3 + pOppOri]; 

			int t1,t2,t1_ori,t2_ori;

			if( d_PStatus[pOrg].isReduntant() && // when pOrg is the smallest redundant point
				( !d_PStatus[pDest].isReduntant() || ( d_PStatus[pDest].isReduntant() && pOrg < pDest ) ) )
			{
				t1 = d_neighborlist[index*3 + (ori + 2) % 3]; 
				t2 = d_neighborlist[pOppTri*3 + (pOppOri + 1) % 3];
			}
			else
			{
				t1 = d_neighborlist[index*3 + (ori + 1) % 3];
				t2 = d_neighborlist[pOppTri*3 + (pOppOri + 2) % 3];
			}
			t1_ori = decode_ori(t1);
			t2_ori = decode_ori(t2);
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			if ( d_flipBy[pOppTri] == index &&
				 d_flipBy[t1] == index &&
			     d_flipBy[t2] == index )
			{
				isWinner = true; 
			}
		}
	}

	if (!isWinner)
		d_TStatus[index].setFlipOri(15); // mark as losers 
}

__global__ void kernelInitLinklist(
	 PStatus    *d_PStatus,
	 int		*d_trianglelist,
	 int        *d_neighborlist,
	 TStatus	*d_TStatus,
	 int		*d_active,
	 int		*d_linklist,
	 int		numberofactive,
	 int * d_tri2subseg,
	 int iteration0,
	 int iteration1,
	 int step
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofactive ) return ;

	int index = d_active[pos];

	int ori, pOrg, pDest, pOpp, pOppTri, pOppOri;
	
	ori = d_TStatus[index].getFlipOri();

	if ( ori < 3 )//2->2 flip
	{
		pOpp	= d_neighborlist[index*3 + (ori + 1) % 3]; 
		pOppTri = decode_tri(pOpp); 
		
		d_linklist[4*pos+0] = index;
		d_linklist[4*pos+1] = pOppTri;
	}	
	else if ( ori < 6 ) // 3->1 flip
	{		
		ori -= 3;
		ori = (ori + 1) % 3;
			
		int t1	   = d_neighborlist[index*3 + ori]; //cw
		int t2	   = d_neighborlist[index*3 + (ori + 2) % 3];//ccw
		t1		   = decode_tri(t1);
		t2		   = decode_tri(t2);

		d_linklist[4*pos+0] = index;
		d_linklist[4*pos+1] = t1;
		d_linklist[4*pos+2] = t2;
	}
	else // 4-2 flip
	{
		ori -= 6;
			
		pOpp = d_neighborlist[index*3 + ori]; 
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp);

		pOrg	 = d_trianglelist[index*3 + (ori + 1) % 3];
		pDest    = d_trianglelist[index*3 + (ori + 2) % 3];
		pOpp	 = d_trianglelist[pOppTri*3 + pOppOri];

		int t1,t2,t1_ori,t2_ori;

		// triangle already changed
		// in order to test direction, need to get current neighbor's direction
		bool outward = false;
		int min = MAXINT;
		
		int p[] = {
			d_trianglelist[3*pOppTri],
			d_trianglelist[3*pOppTri+1],
			d_trianglelist[3*pOppTri+2]
		};

		for(int i=0; i<3; i++)
		{
			if( d_PStatus[p[i]].isReduntant() && p[i] < min )
				min = p[i];
		}

		if( min == p[ (pOppOri+2)%3] ) // neighbor is inward, so I am outward
			outward = true;

		// outward direction
		if( outward )
		{
			t1 = d_neighborlist[index*3 + (ori + 2) % 3]; 
			t2 = d_neighborlist[pOppTri*3 + (pOppOri + 1) % 3];
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			d_linklist[4*pos+0] = index;
			d_linklist[4*pos+1] = pOppTri;
			d_linklist[4*pos+2] = t1;
			d_linklist[4*pos+3] = t2; 
		}
		else // inward direction
		{
			t1 = d_neighborlist[index*3 + (ori + 1) % 3];
			t2 = d_neighborlist[pOppTri*3 + (pOppOri + 2) % 3];
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			d_linklist[4*pos+0] = index;
			d_linklist[4*pos+1] = pOppTri;
			d_linklist[4*pos+2] = t1;
			d_linklist[4*pos+3] = t2; 
		}
	}
}

__device__ void setTriangle(int *d_trianglelist, int pOrg, int pDest, int pApex, int tri, int ori)
{
    d_trianglelist[tri * 3+ (ori + 1) % 3] = pOrg;
    d_trianglelist[tri * 3 + (ori + 2) % 3] = pDest;
    d_trianglelist[tri * 3 + ori] = pApex;
}

__device__ void setNeighborlink(int *d_linkslot, int nOrg, int nDest, int nApex, int listIndex, int ori)
{
	int slotIndex = listIndex*3;
    d_linkslot[ slotIndex + ori] = nOrg;
    d_linkslot[ slotIndex + (ori + 1) % 3] = nDest;
    d_linkslot[ slotIndex + (ori + 2) % 3] = nApex;
}

__global__ void kernelUpdatePhase1
(
	 PStatus 	*d_PStatus,
	 int		*d_trianglelist,
	 int        *d_neighborlist,
	 int		*d_tri2subseg,
	 TStatus	*d_TStatus, 
	 int		*d_subseg2tri,
	 int		*d_flipBy, 
	 int		*d_active,
	 int		*d_linkslot,
	 int		numberofactive,
	 int iteration0,
	 int iteration1,
	 int step
 ) 
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofactive ) return ;

	int index = d_active[pos];
	
	int ori = d_TStatus[index].getFlipOri(); 

    int pOpp, pOppTri, pOppOri, nOpp, pOrg, pDest, pApex, nOrg, nDest, nApex, npDest, npApex;
	int nOrgOri, nDestOri, nApexOri, npDestOri, npApexOri;
	REAL2 vOrg,vDest,vApex,vOpp;

	if ( ori < 3 ) //2->2 flip
	{		
		pOpp = d_neighborlist[index*3 + (ori + 1) % 3]; 
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp);		

		pOrg = d_trianglelist[index*3 + (ori + 1) % 3]; 
		pDest = d_trianglelist[index*3 + (ori + 2) % 3]; 
		pApex = d_trianglelist[index*3 + ori];
		pOpp = d_trianglelist[pOppTri*3 + pOppOri]; 
            
		nOrg = d_neighborlist[index*3 + ori]; 
		nApex = d_neighborlist[index*3 + (ori + 2) % 3];
		npDest = d_neighborlist[pOppTri * 3 + (pOppOri + 1) % 3]; 
		npApex = d_neighborlist[pOppTri * 3 + (pOppOri + 2) % 3]; 

		// Update vertices + nexttri links
		setTriangle(d_trianglelist,pOrg,pDest,pOpp,index,ori);
		setTriangle(d_trianglelist,pApex,pOrg,pOpp,pOppTri,pOppOri);
		setNeighborlink(d_linkslot,nOrg+3,MAXINT,nApex+3,4*pos,ori);
		setNeighborlink(d_linkslot,MAXINT,npDest+3,npApex+3,4*pos+1,pOppOri);

		// Update tri2subseg list
		int sOrg,sApex,spDest,spApex;
		sOrg = d_tri2subseg[3*index + ori];
		sApex = d_tri2subseg[3*index + (ori+2)%3];
		spDest = d_tri2subseg[3*pOppTri + (pOppOri+1)%3];
		spApex = d_tri2subseg[3*pOppTri + (pOppOri+2)%3];

		d_tri2subseg[3*index + ori] = sOrg;
		d_tri2subseg[3*index + (ori+1)%3] = spDest;
		d_tri2subseg[3*index + (ori+2)%3] = -1;

		d_tri2subseg[3*pOppTri + pOppOri] = sApex;
		d_tri2subseg[3*pOppTri + (pOppOri+2)%3] = spApex;
		d_tri2subseg[3*pOppTri + (pOppOri+1)%3] = -1;

		// Update subseg2tri list
		if( sOrg != -1 )
			d_subseg2tri[sOrg] = (index << 2) | ori;

		if( spDest != -1 )
			d_subseg2tri[spDest] = (index << 2) | ((ori+1)%3);

		if( sApex != -1 )
			d_subseg2tri[sApex] = (pOppTri << 2) | pOppOri;

		if( spApex != -1 )
			d_subseg2tri[spApex] = (pOppTri << 2) | ((pOppOri+2)%3);

		d_TStatus[index].setCheck(true);
		d_TStatus[pOppTri].setCheck(true);  
	}
	else if(ori < 6)// 3->1 flip
	{	       
		ori -= 3; // x -> t2 -> t1
		ori = (ori + 1) % 3;
			
		int t1     = d_neighborlist[index*3 + ori]; //cw
		int t2     = d_neighborlist[index*3 + (ori + 2) % 3];//ccw
		int t1_ori = decode_ori(t1);
		int t2_ori = decode_ori(t2);
		t1		   = decode_tri(t1);
		t2		   = decode_tri(t2);

		pOrg	 = d_trianglelist[index*3 + (ori + 1) % 3]; 
		pDest	 = d_trianglelist[index*3 + (ori + 2) % 3]; 
		pApex	 = d_trianglelist[index*3 + ori];

		d_PStatus[pOrg].setDeleted();// mark point as null

		pOrg = d_trianglelist[t1*3 + t1_ori];		

		// neighbors may already changed
		nOrg  = d_neighborlist[t1*3 + (t1_ori + 2) % 3]; 
		nDest = d_neighborlist[index*3 + (ori + 1) % 3]; 
		nApex = d_neighborlist[t2*3 + (t2_ori + 1) % 3];
				
		setTriangle(d_trianglelist,pOrg,pDest,pApex,index,ori);
		setNeighborlink(d_linkslot,nOrg+3,nDest+3,nApex+3,4*pos,ori);
				
		d_linkslot[ (4*pos+1) * 3 + (t1_ori + 2) % 3] = nOrg + 3; 
		d_linkslot[ (4*pos+2) * 3 + (t2_ori + 1) % 3] = nApex + 3; 

		// Update tri2subseg list
		int sOrg,sApex,sDest;
		sOrg = d_tri2subseg[3*index + (ori+1)%3];
		sApex = d_tri2subseg[3*t1 + (t1_ori+2)%3];
		sDest = d_tri2subseg[3*t2 + (t2_ori+1)%3];

		d_tri2subseg[3*index + (ori+1)%3] = sOrg;
		d_tri2subseg[3*index + ori] = sApex;
		d_tri2subseg[3*index + (ori+2)%3] = sDest;

		// Update subseg2tri list
		if(sOrg != -1)
			d_subseg2tri[sOrg] = (index << 2) | ((ori+1)%3);
		if(sApex != -1)
			d_subseg2tri[sApex] = (index << 2) | ori;
		if(sDest != -1)
			d_subseg2tri[sDest] = (index << 2) | ((ori+2)%3);

		d_TStatus[index].setCheck(true);

		d_TStatus[t1].setNull(true); //mark triangle as null
		d_TStatus[t2].setNull(true); //mark triangle as null
	}
	else// 4->2 flip
	{
		ori -= 6;
			
		pOpp = d_neighborlist[index*3 + ori]; 
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp);

		pOrg	 = d_trianglelist[index*3 + (ori + 1) % 3]; 
		pDest	 = d_trianglelist[index*3 + (ori + 2) % 3]; 
		pApex	 = d_trianglelist[index*3 + ori];
		pOpp	 = d_trianglelist[pOppTri*3 + pOppOri]; 

		int t1,t2,t1_ori,t2_ori;

		if( d_PStatus[pOrg].isReduntant() && // when pOrg is the smallest redundant point
			( !d_PStatus[pDest].isReduntant() || ( d_PStatus[pDest].isReduntant() && pOrg < pDest ) ) )
		{
			t1 = d_neighborlist[index*3 + (ori + 2) % 3]; 
			t2 = d_neighborlist[pOppTri*3 + (pOppOri + 1) % 3];
		}
		else
		{
			t1 = d_neighborlist[index*3 + (ori + 1) % 3];
			t2 = d_neighborlist[pOppTri*3 + (pOppOri + 2) % 3];
		}
		t1_ori = decode_ori(t1);
		t2_ori = decode_ori(t2);
		t1 = decode_tri(t1);
		t2 = decode_tri(t2);

		nOpp = d_trianglelist[3*t1 + t1_ori];

		int sOrg,sDest,sApex,snOrg,snDest,snApex;

		// outward direction
		if( d_PStatus[pOrg].isReduntant() && // when pOrg is the smallest redundant point
			( !d_PStatus[pDest].isReduntant() || ( d_PStatus[pDest].isReduntant() && pOrg < pDest ) ) )
		{
			d_PStatus[pOrg].setDeleted();// mark point as null				

			// neighbors may already changed
			nOrg = d_neighborlist[pOppTri*3 + (pOppOri + 2) % 3];
			nDest = d_neighborlist[index*3 + (ori + 1) % 3]; 
			npDest = d_neighborlist[t1*3 + (t1_ori + 1) % 3];
			npApex = d_neighborlist[t2*3 + (t2_ori + 2) % 3];
			
			setTriangle(d_trianglelist,pOpp,pDest,pApex,index,ori);
			setTriangle(d_trianglelist,pOpp,pApex,nOpp,t1,t1_ori);
			setNeighborlink(d_linkslot,nOrg+3,nDest+3,MAXINT,4*pos,ori);
			setNeighborlink(d_linkslot,MAXINT,npDest+3,npApex+3,4*pos+2,t1_ori);
				
			d_linkslot[(4*pos+1)*3 + (pOppOri + 2) % 3] = nOrg + 3; 
			d_linkslot[(4*pos+3)*3 + (t2_ori + 2) % 3] = npApex + 3; 

			// Record tri2subseg list
			sOrg = d_tri2subseg[3*pOppTri + (pOppOri+2)%3];
			sDest = d_tri2subseg[3*index + (ori+1)%3];
			sApex = d_tri2subseg[3*index + (ori+2)%3];

			snOrg = d_tri2subseg[3*t1 + t1_ori];
			snDest = d_tri2subseg[3*t1 + (t1_ori+1)%3];
			snApex = d_tri2subseg[3*t2 + (t2_ori+2)%3];
		}
		else // inward direction
		{
			d_PStatus[pDest].setDeleted();// mark point as null				

			// neighbors may already changed
			nOrg = d_neighborlist[pOppTri*3 + (pOppOri + 1) % 3];
			nApex = d_neighborlist[index*3 + (ori + 2) % 3]; 
			npDest = d_neighborlist[t2*3 + (t2_ori + 1) % 3];
			npApex = d_neighborlist[t1*3 + (t1_ori + 2) % 3];

			setTriangle(d_trianglelist,pOrg,pOpp,pApex,index,ori);
			setTriangle(d_trianglelist,pApex,pOpp,nOpp,t1,t1_ori);
			setNeighborlink(d_linkslot,nOrg+3,MAXINT,nApex+3,4*pos,ori);
			setNeighborlink(d_linkslot,MAXINT,npDest+3,npApex+3,4*pos+2,t1_ori);
			
			d_linkslot[(4*pos+1)*3 + (pOppOri + 1)%3] = nOrg + 3;
			d_linkslot[(4*pos+3)*3 + (t2_ori + 1)%3] = npDest + 3;

			// Record tri2subseg list
			sOrg = d_tri2subseg[3*pOppTri + (pOppOri+1)%3];
			sDest = d_tri2subseg[3*index + (ori+1)%3];
			sApex = d_tri2subseg[3*index + (ori+2)%3];

			snOrg = d_tri2subseg[3*t1 + t1_ori];
			snDest = d_tri2subseg[3*t2 + (t2_ori+1)%3];
			snApex = d_tri2subseg[3*t1 + (t1_ori+2)%3];
		}

		// Update tri2subseg list
		d_tri2subseg[3*index + ori] = sOrg;
		d_tri2subseg[3*index + (ori+1)%3] = sDest;
		d_tri2subseg[3*index + (ori+2)%3] = sApex;

		d_tri2subseg[3*t1 + t1_ori] = snOrg;
		d_tri2subseg[3*t1 + (t1_ori+1)%3] = snDest;
		d_tri2subseg[3*t1 + (t1_ori+2)%3] = snApex;

		// Update subseg2tri list
		if(sOrg != -1)
			d_subseg2tri[sOrg] = (index << 2) | ori;
		if(sDest != -1)
			d_subseg2tri[sDest] = (index << 2) | ((ori+1)%3);
		if(sApex != -1)
			d_subseg2tri[sApex] = (index << 2) | ((ori+2)%3);
		if(snOrg != -1)
			d_subseg2tri[snOrg] = (t1 << 2) | t1_ori;
		if(snDest != -1)
			d_subseg2tri[snDest] = (t1 << 2) | ((t1_ori+1)%3);
		if(snApex != -1)
			d_subseg2tri[snApex] = (t1 << 2) | ((t1_ori+2)%3);

		d_TStatus[index].setCheck(true);
		d_TStatus[t1].setCheck(true);

		d_TStatus[pOppTri].setNull(true); //mark triangle as null
		d_TStatus[t2].setNull(true); //mark triangle as null
	}
}

__device__ void updateNeighborlink(int *d_linklist,int *d_linkslot,int wIndex,int otri,int neighbor,int numberofwinner)
{

	// in d_linklist, the winner indices (divisable by 4) are sorted
	// use binary search to get the indices for winners
	int low = 0;
	int high = numberofwinner-1;

	int listIndex = -1; // thread index for link list
	int slotIndex = -1; // thread index for link slot

	int triIndex = decode_tri(otri);
	int triOri = decode_ori(otri);

	bool found = false;
	while(high>=low)
	{
		int middle = (low + high) / 2;
		int windex = d_linklist[4*middle];

		if(windex == wIndex)
		{
			for(int i=0; i<4; i++)
			{
				int tindex = d_linklist[4*middle+i];
				if(tindex == triIndex)
				{
					listIndex = 4*middle+i;
					found = true;
					break;
				}
			}
			if(found)
				break;
		}
		else if (windex < wIndex)
			low = middle + 1;
		else
			high = middle - 1;
	}
	slotIndex = listIndex*3;
	d_linkslot[slotIndex + triOri] = -neighbor;
}

__global__ void kernelUpdatePhase2
(
	 PStatus 	*d_PStatus,
	 int		*d_trianglelist,
	 int        *d_neighborlist,
	 int		*d_tri2subseg,
	 TStatus	*d_TStatus, 
	 int		*d_subseg2tri,
	 int		*d_flipBy, 
	 int		*d_active,
	 int		*d_linklist,
	 int		*d_linkslot,
	 int		numberofactive,
	 int iteration
 ) 
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofactive ) return ;

	int index = d_active[pos];

	int ori, pOrg, pDest, pOpp, pOppTri, pOppOri, nOrg, nDest, nApex, npDest, npApex;
	int nOrgOri, nDestOri, nApexOri, npDestOri, npApexOri;
	
	ori = d_TStatus[index].getFlipOri(); 

	if ( ori < 3 )//2->2 flip
	{
		// trianglelist changed already
		// neighborlist doesn't change
		pOpp	= d_neighborlist[index*3 + (ori + 1) % 3]; 
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp); 

		nOrg = d_neighborlist[index*3 + ori];
		nApex = d_neighborlist[index*3 + (ori + 2) % 3]; 
		npDest = d_neighborlist[pOppTri*3 + (pOppOri + 1)% 3]; 
		npApex = d_neighborlist[pOppTri*3 + (pOppOri + 2) % 3];

		// Update my neighbors of my existence
		// only update winners and theirs belongings
		int tempIndex,winnerIndex;

		tempIndex = decode_tri(nOrg);
		if(tempIndex != -1)
		{
			winnerIndex = d_flipBy[tempIndex];
			if(winnerIndex != MAXINT && winnerIndex >= 0 &&
				d_TStatus[winnerIndex].getFlipOri() != 15)
				updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nOrg,encode_tri(index, ori),numberofactive); 
		}

		tempIndex = decode_tri(nApex);
		if(tempIndex != -1)
		{
			winnerIndex = d_flipBy[tempIndex];
			if(winnerIndex != MAXINT && winnerIndex >= 0 &&
				d_TStatus[winnerIndex].getFlipOri() != 15)
				updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nApex,encode_tri(pOppTri, pOppOri),numberofactive);
		}

		tempIndex = decode_tri(npDest);
		if(tempIndex != -1)
		{
			winnerIndex = d_flipBy[tempIndex];
			if(winnerIndex != MAXINT && winnerIndex >= 0 &&
				d_TStatus[winnerIndex].getFlipOri() != 15)
				updateNeighborlink(d_linklist,d_linkslot,winnerIndex,npDest,encode_tri(index, (ori + 1) % 3),numberofactive);
		}
		
		tempIndex = decode_tri(npApex);
		if(tempIndex != -1)
		{
			winnerIndex = d_flipBy[tempIndex];
			if(winnerIndex != MAXINT && winnerIndex >=0 &&
				d_TStatus[winnerIndex].getFlipOri() != 15)
				updateNeighborlink(d_linklist,d_linkslot,winnerIndex,npApex,encode_tri(pOppTri, (pOppOri + 2) % 3),numberofactive);
		}

	}	
	else if ( ori < 6 ) // 3->1 flip
	{	
		ori -= 3;
		ori = (ori + 1) % 3;
			
		int t1	   = d_neighborlist[index*3 + ori]; //cw
		int t2	   = d_neighborlist[index*3 + (ori + 2) % 3];//ccw
		int t1_ori = decode_ori(t1);
		int t2_ori = decode_ori(t2);
		t1		   = decode_tri(t1);
		t2		   = decode_tri(t2);

		nOrg  = d_neighborlist[t1 * 3 + (t1_ori + 2) % 3];
		nDest = d_neighborlist[index*3 + (ori + 1) % 3];
		nApex = d_neighborlist[t2*3 + (t2_ori + 1) % 3];

		// Update my neighbors of my existence
		// only update winners and theirs belongings
		int tempIndex,winnerIndex;

		tempIndex = decode_tri(nOrg);
		if(tempIndex != -1)
		{
			winnerIndex = d_flipBy[tempIndex];
			if(winnerIndex != MAXINT && winnerIndex >= 0 &&
				d_TStatus[winnerIndex].getFlipOri() != 15)
				updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nOrg,encode_tri(index, ori),numberofactive); 
		}
		
		tempIndex = decode_tri(nDest);
		if(tempIndex != -1)
		{
			winnerIndex = d_flipBy[tempIndex];
			if(winnerIndex != MAXINT && winnerIndex >= 0 &&
				d_TStatus[winnerIndex].getFlipOri() != 15)
				updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nDest,encode_tri(index, (ori + 1) % 3),numberofactive); 
		}

		tempIndex = decode_tri(nApex);
		if(tempIndex != -1)
		{
			winnerIndex = d_flipBy[tempIndex];
			if(winnerIndex != MAXINT && winnerIndex >= 0 &&
				d_TStatus[winnerIndex].getFlipOri() != 15)
				updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nApex,encode_tri(index, (ori + 2) % 3),numberofactive);
		}
	}
	else // 4-2 flip
	{
		ori -= 6;
			
		pOpp = d_neighborlist[index*3 + ori]; 
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp);

		pOrg	 = d_trianglelist[index*3 + (ori + 1) % 3];
		pDest    = d_trianglelist[index*3 + (ori + 2) % 3];
		pOpp	 = d_trianglelist[pOppTri*3 + pOppOri];

		int t1,t2,t1_ori,t2_ori;

		// triangle already changed
		// in order to test direction, need to get current neighbor's direction
		bool outward = false;
		int min = MAXINT;
		
		int p[] = {
			d_trianglelist[3*pOppTri],
			d_trianglelist[3*pOppTri+1],
			d_trianglelist[3*pOppTri+2]
		};

		for(int i=0; i<3; i++)
		{
			if( d_PStatus[p[i]].isReduntant() && p[i] < min )
				min = p[i];
		}

		if( min == p[ (pOppOri+2)%3] ) // neighbor is inward, so I am outward
			outward = true;

		// outward direction
		if( outward )
		{
			t1 = d_neighborlist[index*3 + (ori + 2) % 3]; 
			t2 = d_neighborlist[pOppTri*3 + (pOppOri + 1) % 3];
			t1_ori = decode_ori(t1);
			t2_ori = decode_ori(t2);
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			nOrg = d_neighborlist[3*pOppTri + (pOppOri+2)%3];
			nDest = d_neighborlist[3*index + (ori+1)%3];
			npDest = d_neighborlist[3*t1 + (t1_ori+1)%3];
			npApex = d_neighborlist[3*t2 + (t2_ori+2)%3];

			// Update my neighbors of my existence
			// only update winners and theirs belongings
			int tempIndex,winnerIndex;

			tempIndex = decode_tri(nOrg);
			if(tempIndex != -1)
			{
				winnerIndex = d_flipBy[tempIndex];
				if(winnerIndex != MAXINT && winnerIndex >= 0 &&
					d_TStatus[winnerIndex].getFlipOri() != 15)
					updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nOrg,encode_tri(index, ori),numberofactive); 
			}

			tempIndex = decode_tri(nDest);
			if(tempIndex != -1)
			{
				winnerIndex = d_flipBy[tempIndex];
				if(winnerIndex != MAXINT && winnerIndex >= 0 &&
					d_TStatus[winnerIndex].getFlipOri() != 15)
					updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nDest,encode_tri(index, (ori+1)%3),numberofactive);
			}

			tempIndex = decode_tri(npDest);
			if(tempIndex != -1)
			{
				winnerIndex = d_flipBy[tempIndex];
				if(winnerIndex != MAXINT && winnerIndex >= 0 &&
					d_TStatus[winnerIndex].getFlipOri() != 15)
					updateNeighborlink(d_linklist,d_linkslot,winnerIndex,npDest,encode_tri(t1, (t1_ori + 1) % 3),numberofactive);
			}
		
			tempIndex = decode_tri(npApex);
			if(tempIndex != -1)
			{
				winnerIndex = d_flipBy[tempIndex];
				if(winnerIndex != MAXINT && winnerIndex >= 0 &&
					d_TStatus[winnerIndex].getFlipOri() != 15)
					updateNeighborlink(d_linklist,d_linkslot,winnerIndex,npApex,encode_tri(t1, (t1_ori + 2) % 3),numberofactive);
			}

		}
		else // inward direction
		{
			t1 = d_neighborlist[index*3 + (ori + 1) % 3];
			t2 = d_neighborlist[pOppTri*3 + (pOppOri + 2) % 3];
			t1_ori = decode_ori(t1);
			t2_ori = decode_ori(t2);
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			nOrg = d_neighborlist[3*pOppTri + (pOppOri+1)%3];
			nApex = d_neighborlist[3*index + (ori+2)%3];
			npDest = d_neighborlist[3*t2 + (t2_ori+1)%3];
			npApex = d_neighborlist[3*t1 + (t1_ori+2)%3];

			// Update my neighbors of my existence
			// only update winners and theirs belongings
			int tempIndex,winnerIndex;

			tempIndex = decode_tri(nOrg);
			if(tempIndex != -1)
			{
				winnerIndex = d_flipBy[tempIndex];
				if(winnerIndex != MAXINT && winnerIndex >= 0 &&
					d_TStatus[winnerIndex].getFlipOri() != 15)
					updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nOrg,encode_tri(index, ori),numberofactive); 
			}

			tempIndex = decode_tri(nApex);
			if(tempIndex != -1)
			{
				winnerIndex = d_flipBy[tempIndex];
				if(winnerIndex != MAXINT && winnerIndex >= 0 &&
					d_TStatus[winnerIndex].getFlipOri() != 15)
					updateNeighborlink(d_linklist,d_linkslot,winnerIndex,nApex,encode_tri(index, (ori+2)%3),numberofactive);
			}

			tempIndex = decode_tri(npDest);
			if(tempIndex != -1)
			{
				winnerIndex = d_flipBy[tempIndex];
				if(winnerIndex != MAXINT && winnerIndex >= 0 &&
					d_TStatus[winnerIndex].getFlipOri() != 15)
					updateNeighborlink(d_linklist,d_linkslot,winnerIndex,npDest,encode_tri(t1, (t1_ori + 1) % 3),numberofactive);
			}
		
			tempIndex = decode_tri(npApex);
			if(tempIndex != -1)
			{
				winnerIndex = d_flipBy[tempIndex];
				if(winnerIndex != MAXINT && winnerIndex >= 0 &&
					d_TStatus[winnerIndex].getFlipOri() != 15)
					updateNeighborlink(d_linklist,d_linkslot,winnerIndex,npApex,encode_tri(t1, (t1_ori + 2) % 3),numberofactive);
			}
		}
	}
}

__device__ void updateNeighborlist(int *d_neighborlist, int tri, int pTri)
{
    if (tri >= 0)
		d_neighborlist[decode_tri(tri) * 3 + decode_ori(tri)] = pTri;
}

__device__ void resetEncseg(int * d_trianglelist,REAL2 * d_pointlist,int * d_encmarker,int segIndex,int otri0,int otri1,REAL theta, int run_mode)
{
	int pOri,pDest,pApex,tri_index,tri_ori;
	REAL2 vOrg,vDest,vApex;

	// reset first
	d_encmarker[segIndex] = -1;

	// check first side
	tri_index = decode_tri(otri0);
	tri_ori = decode_ori(otri0);

	if(otri0 != -1)
	{
		pOri = d_trianglelist[3*tri_index + (tri_ori+1)%3];
		pDest = d_trianglelist[3*tri_index + (tri_ori+2)%3];
		pApex = d_trianglelist[3*tri_index + tri_ori];

		vOrg = d_pointlist[pOri];
		vDest = d_pointlist[pDest];
		vApex = d_pointlist[pApex];

		if(checkseg4encroach(vOrg,vDest,vApex,theta, run_mode))
			d_encmarker[segIndex] = 0;
	}

	// check the other side
	tri_index = decode_tri(otri1);
	tri_ori = decode_ori(otri1);

	if(otri1 != -1)
	{
		pOri = d_trianglelist[3*tri_index + (tri_ori+1)%3];
		pDest = d_trianglelist[3*tri_index + (tri_ori+2)%3];
		pApex = d_trianglelist[3*tri_index + tri_ori];

		vOrg = d_pointlist[pOri];
		vDest = d_pointlist[pDest];
		vApex = d_pointlist[pApex];

		if(checkseg4encroach(vOrg,vDest,vApex,theta,run_mode))
			d_encmarker[segIndex] = 0;
	}
}

__global__ void kernelUpdatePhase3
(
	 REAL2		*d_pointlist,
	 PStatus 	*d_PStatus,
	 int		*d_trianglelist,
	 int        *d_neighborlist,
	 int		*d_tri2subseg,
	 TStatus	*d_TStatus, 
	 int		*d_subseg2tri,
	 int		*d_flipBy, 
	 int		*d_active,
	 int		*d_encmarker,
	 int		*d_linklist,
	 int		*d_linkslot,
	 int		numberofactive,
	 int run_mode,
	 REAL		theta,
	 int iteration,
	 int step,
	 int last_triangle
 ) 
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofactive ) return ;

	int index = d_active[pos];

    int ori, pOrg, pDest, pOpp, pOppTri, pOppOri, nOrg, nDest, nApex, npDest, npApex;
	int nOrgOri, nDestOri, nApexOri, npDestOri, npApexOri;

	int otri0,otri1;
	REAL2 vOrg,vDest,vApex;

	ori	= d_TStatus[index].getFlipOri();

	d_TStatus[index].setFlipOri(15);
	
	if(ori < 3)// 2->2 flip
	{	
		pOpp	= d_neighborlist[index*3 + (ori + 1) % 3]; 
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp); 

		// Update other links
		nOrg = d_linkslot[4*pos*3 + ori];
		nApex = d_linkslot[4*pos*3 + (ori+2)%3];
		npDest = d_linkslot[(4*pos+1)*3 + (pOppOri+1)%3];
		npApex = d_linkslot[(4*pos+1)*3 + (pOppOri+2)%3];

		if ( nOrg > 0 ) {        // My neighbor do not update me, update him
			nOrg = -(nOrg - 3); 
			updateNeighborlist(d_neighborlist,-nOrg,encode_tri(index,ori));	
		}

		if ( nApex > 0 ) {
			nApex = -(nApex - 3); 
			updateNeighborlist(d_neighborlist,-nApex,encode_tri(pOppTri, pOppOri)); 		
		}

		if ( npDest > 0 ) {
			npDest = -(npDest - 3); 
			updateNeighborlist(d_neighborlist,-npDest,encode_tri(index, (ori + 1) % 3)); 		
		}

		if ( npApex > 0 ) {
			npApex = -(npApex - 3); 
			updateNeighborlist(d_neighborlist,-npApex,encode_tri(pOppTri, (pOppOri + 2) % 3)); 		
		}

		// Update my own links
		d_neighborlist[index*3 + ori] = -nOrg;
		d_neighborlist[index*3 + (ori + 1) % 3] = -npDest; 
		d_neighborlist[index*3 + (ori + 2) % 3] = encode_tri(pOppTri, (pOppOri + 1) % 3); 
		d_neighborlist[pOppTri*3 + pOppOri] = -nApex;
		d_neighborlist[pOppTri*3 + (pOppOri + 1) % 3] = encode_tri(index, (ori + 2) % 3); 
		d_neighborlist[pOppTri*3 + (pOppOri + 2) % 3] = -npApex;

		// Update encroached marker
		int sOrg,sDest,spOrg,spApex;
		sOrg = d_tri2subseg[3*index + ori];
		sDest = d_tri2subseg[3*index + (ori+1)%3];
		spOrg = d_tri2subseg[3*pOppTri + pOppOri];
		spApex = d_tri2subseg[3*pOppTri + (pOppOri+2)%3];

		if(sOrg != -1)
		{
			otri0 = (index << 2) | ori;
			otri1 = -nOrg;
			resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sOrg,otri0,otri1,theta,run_mode);
		}

		if(sDest != -1)
		{
			otri0 = (index << 2) | ((ori+1)%3);
			otri1 = -npDest;
			resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sDest,otri0,otri1,theta,run_mode);
		}

		if(spOrg != -1)
		{
			otri0 = (pOppTri << 2) | pOppOri;
			otri1 = -nApex;
			resetEncseg(d_trianglelist,d_pointlist,d_encmarker,spOrg,otri0,otri1,theta,run_mode);
		}

		if(spApex != -1)
		{
			otri0 = (pOppTri << 2) | ((pOppOri+2)%3);
			otri1 = -npApex;
			resetEncseg(d_trianglelist,d_pointlist,d_encmarker,spApex,otri0,otri1,theta,run_mode);
		}

		// mark neighbor
		d_TStatus[pOppTri].setFlip(true);
	}
	else if( ori < 6 ) // 3-1 flip
	{
		ori -= 3;
		ori = (ori + 1) % 3;
			
		int t1	   = d_neighborlist[index*3 + ori]; //cw
		int t2	   = d_neighborlist[index*3 + (ori + 2) % 3];//ccw
		int t1_ori = decode_ori(t1);
		int t2_ori = decode_ori(t2);
		t1         = decode_tri(t1);
		t2         = decode_tri(t2);

		nOrg = d_linkslot[(4*pos+1)*3 + (t1_ori+2)%3];
		nDest = d_linkslot[4*pos*3 + (ori+1)%3];
		nApex = d_linkslot[(4*pos+2)*3 + (t2_ori+1)%3];

		if ( nOrg > 0 ) {        // My neighbor do not update me, update him
			nOrg = -(nOrg - 3); 
			updateNeighborlist(d_neighborlist,-nOrg,encode_tri(index,ori)); 		
		}

		if ( nDest > 0 ) {    
			nDest = -(nDest - 3); 
			updateNeighborlist(d_neighborlist,-nDest,encode_tri(index,(ori+1)%3));	
		}

		if ( nApex > 0 ) {
			nApex = -(nApex - 3); 
			updateNeighborlist(d_neighborlist,-nApex,encode_tri(index,(ori+2)%3)); 		
		}

		// Update my own neighbors
		d_neighborlist[index*3 + ori] = -nOrg; 
		d_neighborlist[index*3 + (ori + 1) % 3] = -nDest; 
		d_neighborlist[index*3 + (ori + 2) % 3] = -nApex; 

		// Update encroached marker
		int sOrg,sDest,sApex;
		sOrg = d_tri2subseg[3*index + ori];
		sDest = d_tri2subseg[3*index + (ori+1)%3];
		sApex = d_tri2subseg[3*index + (ori+2)%3];

		if(sOrg != -1)
		{
			otri0 = (index << 2) | ori;
			otri1 = -nOrg;
			resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sOrg,otri0,otri1,theta,run_mode);
		}

		if(sDest != -1)
		{
			otri0 = (index << 2) | ((ori+1)%3);;
			otri1 = -nDest;
			resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sDest,otri0,otri1,theta,run_mode);
		}

		if(sApex != -1)
		{
			otri0 = (index << 2) | ((ori+2)%3);;
			otri1 = -nApex;
			resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sApex,otri0,otri1,theta,run_mode);
		}

		d_TStatus[t1].setFlip(false);
		d_TStatus[t2].setFlip(false);
	}
	else // 4-2 flip
	{
		ori -= 6;
			
		pOpp = d_neighborlist[index*3 + ori]; 
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp);

		pOrg	 = d_trianglelist[index*3 + (ori + 1) % 3];
		pDest    = d_trianglelist[index*3 + (ori + 2) % 3];
		pOpp	 = d_trianglelist[pOppTri*3 + pOppOri];

		int t1,t2,t1_ori,t2_ori;

		// triangle already changed
		// in order to test direction, need to get current neighbor's direction
		bool outward = false;
		int min = MAXINT;
		
		int p[] = {
			d_trianglelist[3*pOppTri],
			d_trianglelist[3*pOppTri+1],
			d_trianglelist[3*pOppTri+2]
		};

		for(int i=0; i<3; i++)
		{
			if( d_PStatus[p[i]].isReduntant() && p[i] < min )
				min = p[i];
		}

		if( min == p[ (pOppOri+2)%3] ) // neighbor is inward, so I am outward
			outward = true;

		if( outward ) //outward direction
		{
			t1 = d_neighborlist[index*3 + (ori + 2) % 3]; 
			t2 = d_neighborlist[pOppTri*3 + (pOppOri + 1) % 3];
			t1_ori = decode_ori(t1);
			t2_ori = decode_ori(t2);
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			nOrg = d_linkslot[(4*pos+1)*3 + (pOppOri+2)%3];
			nDest = d_linkslot[4*pos*3 + (ori+1)%3];
			npDest = d_linkslot[(4*pos+2)*3 + (t1_ori+1)%3];
			npApex = d_linkslot[(4*pos+3)*3 + (t2_ori+2)%3];

			if ( nOrg > 0 ) {        // My neighbor do not update me, update him
				nOrg = -(nOrg - 3);
				updateNeighborlist(d_neighborlist,-nOrg,encode_tri(index,ori));	
			}

			if ( nDest > 0 ) {       
				nDest = -(nDest - 3);
				updateNeighborlist(d_neighborlist,-nDest,encode_tri(index,(ori+1)%3));	
			}
			
			if ( npDest > 0 ) { 
				npDest = -(npDest - 3);
				updateNeighborlist(d_neighborlist,-npDest,encode_tri(t1,(t1_ori+1)%3));	
			}

			if ( npApex > 0 ) {       
				npApex = -(npApex - 3);
				updateNeighborlist(d_neighborlist,-npApex,encode_tri(t1,(t1_ori+2)%3));	
			}

			// Update my own links
			d_neighborlist[index*3 + ori]    = -nOrg; 
			d_neighborlist[index*3 + (ori + 1) % 3] = -nDest; 
			d_neighborlist[t1*3 + (t1_ori + 1) % 3] = -npDest;
			d_neighborlist[t1*3 + (t1_ori + 2) % 3] = -npApex;

			// Update encroached marker
			int sOrg,sDest,spDest,spApex;
			sOrg = d_tri2subseg[3*index + ori];
			sDest = d_tri2subseg[3*index + (ori+1)%3];
			spDest = d_tri2subseg[3*t1 + (t1_ori+1)%3];
			spApex = d_tri2subseg[3*t1 + (t1_ori+2)%3];

			if(sOrg != -1)
			{
				otri0 = (index << 2) | ori;
				otri1 = -nOrg;
				resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sOrg,otri0,otri1,theta,run_mode);
			}

			if(sDest != -1)
			{
				otri0 = (index << 2) | ((ori+1)%3);
				otri1 = -nDest;
				resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sDest,otri0,otri1,theta,run_mode);
			}

			if(spDest != -1)
			{
				otri0 = (t1 << 2) | ((t1_ori+1)%3);
				otri1 = -npDest;
				resetEncseg(d_trianglelist,d_pointlist,d_encmarker,spDest,otri0,otri1,theta,run_mode);
			}

			if(spApex != -1)
			{
				otri0 = (t1 << 2) | ((t1_ori+2)%3);
				otri1 = -npApex;
				resetEncseg(d_trianglelist,d_pointlist,d_encmarker,spApex,otri0,otri1,theta,run_mode);
			}

		}
		else
		{
			t1 = d_neighborlist[index*3 + (ori + 1) % 3];
			t2 = d_neighborlist[pOppTri*3 + (pOppOri + 2) % 3];
			t1_ori = decode_ori(t1);
			t2_ori = decode_ori(t2);
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			nOrg = d_linkslot[(4*pos+1)*3 + (pOppOri+1)%3];
			nApex = d_linkslot[4*pos*3 + (ori+2)%3];
			npDest = d_linkslot[(4*pos+3)*3 + (t2_ori+1)%3];
			npApex = d_linkslot[(4*pos+2)*3 + (t1_ori+2)%3];

			if ( nOrg > 0 ) {        // My neighbor do not update me, update him
				nOrg = -(nOrg - 3);
				updateNeighborlist(d_neighborlist,-nOrg,encode_tri(index,ori));	
			}

			if ( nApex > 0 ) {       
				nApex = -(nApex - 3);
				updateNeighborlist(d_neighborlist,-nApex,encode_tri(index,(ori+2)%3));	
			}
			
			if ( npDest > 0 ) { 
				npDest = -(npDest - 3);
				updateNeighborlist(d_neighborlist,-npDest,encode_tri(t1,(t1_ori+1)%3));	
			}

			if ( npApex > 0 ) {       
				npApex = -(npApex - 3);
				updateNeighborlist(d_neighborlist,-npApex,encode_tri(t1,(t1_ori+2)%3));	
			}
		    // Update my own links
			d_neighborlist[index*3 + ori]    = -nOrg; 
			d_neighborlist[index*3 + (ori + 2) % 3] = -nApex; 
			d_neighborlist[t1*3 + (t1_ori + 1) % 3] = -npDest;
			d_neighborlist[t1*3 + (t1_ori + 2) % 3] = -npApex;

			// Update encroached marker
			int sOrg,sApex,spDest,spApex;
			sOrg = d_tri2subseg[3*index + ori];
			sApex = d_tri2subseg[3*index + (ori+2)%3];
			spDest = d_tri2subseg[3*t1 + (t1_ori+1)%3];
			spApex = d_tri2subseg[3*t1 + (t1_ori+2)%3];

			if(sOrg != -1)
			{
				otri0 = (index << 2) | ori;
				otri1 = -nOrg;
				resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sOrg,otri0,otri1,theta,run_mode);
			}

			if(sApex != -1)
			{
				otri0 = (index << 2) | ((ori+2)%3);
				otri1 = -nApex;
				resetEncseg(d_trianglelist,d_pointlist,d_encmarker,sApex,otri0,otri1,theta,run_mode);
			}

			if(spDest != -1)
			{
				otri0 = (t1 << 2) | ((t1_ori+1)%3);
				otri1 = -npDest;
				resetEncseg(d_trianglelist,d_pointlist,d_encmarker,spDest,otri0,otri1,theta,run_mode);
			}

			if(spApex != -1)
			{
				otri0 = (t1 << 2) | ((t1_ori+2)%3);
				otri1 = -npApex;
				resetEncseg(d_trianglelist,d_pointlist,d_encmarker,spApex,otri0,otri1,theta,run_mode);
			}
		}

		d_TStatus[t1].setFlip(true);

		d_TStatus[pOppTri].setFlip(false);
		d_TStatus[t2].setFlip(false);
	}
}

__device__ void resetEncseg(int * d_trianglelist, REAL2 * d_pointlist, int * d_encmarker, int * d_segmarker, int segIndex, int otri0, int otri1, REAL theta, int encmode)
{
	int pOri, pDest, pApex, tri_index, tri_ori;
	REAL2 vOrg, vDest, vApex;

	// reset first
	d_encmarker[segIndex] = -1;

	// check first side
	tri_index = decode_tri(otri0);
	tri_ori = decode_ori(otri0);

	if (otri0 != -1)
	{
		pOri = d_trianglelist[3 * tri_index + (tri_ori + 1) % 3];
		pDest = d_trianglelist[3 * tri_index + (tri_ori + 2) % 3];
		pApex = d_trianglelist[3 * tri_index + tri_ori];

		vOrg = d_pointlist[pOri];
		vDest = d_pointlist[pDest];
		vApex = d_pointlist[pApex];

		if (checkseg4encroach(vOrg, vDest, vApex, theta, encmode))
		{
			d_encmarker[segIndex] = 0;
			d_segmarker[segIndex] = 1;
		}
	}

	// check the other side
	tri_index = decode_tri(otri1);
	tri_ori = decode_ori(otri1);

	if (otri1 != -1)
	{
		pOri = d_trianglelist[3 * tri_index + (tri_ori + 1) % 3];
		pDest = d_trianglelist[3 * tri_index + (tri_ori + 2) % 3];
		pApex = d_trianglelist[3 * tri_index + tri_ori];

		vOrg = d_pointlist[pOri];
		vDest = d_pointlist[pDest];
		vApex = d_pointlist[pApex];

		if (checkseg4encroach(vOrg, vDest, vApex, theta, encmode))
		{
			d_encmarker[segIndex] = 0;
			d_segmarker[segIndex] = 1;
		}
	}
}

__global__ void kernelUpdatePhase3_New
(
	REAL2   *d_pointlist,
	PStatus *d_PStatus,
	int		*d_trianglelist,
	int     *d_neighborlist,
	int		*d_tri2subseg,
	TStatus	*d_TStatus,
	int		*d_subseg2tri,
	int		*d_flipBy,
	int		*d_active,
	int		*d_encmarker,
	int     *d_segmarker,
	int		*d_linklist,
	int		*d_linkslot,
	int		numberofactive,
	int encmode,
	REAL theta
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= numberofactive) return;

	int index = d_active[pos];

	int ori, pOrg, pDest, pOpp, pOppTri, pOppOri, nOrg, nDest, nApex, npDest, npApex;
	int nOrgOri, nDestOri, nApexOri, npDestOri, npApexOri;

	int otri0, otri1;
	REAL2 vOrg, vDest, vApex;

	ori = d_TStatus[index].getFlipOri();

	d_TStatus[index].setFlipOri(15);

	if (ori < 3)// 2->2 flip
	{
		pOpp = d_neighborlist[index * 3 + (ori + 1) % 3];
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp);

		// Update other links
		nOrg = d_linkslot[4 * pos * 3 + ori];
		nApex = d_linkslot[4 * pos * 3 + (ori + 2) % 3];
		npDest = d_linkslot[(4 * pos + 1) * 3 + (pOppOri + 1) % 3];
		npApex = d_linkslot[(4 * pos + 1) * 3 + (pOppOri + 2) % 3];

		if (nOrg > 0) {        // My neighbor do not update me, update him
			nOrg = -(nOrg - 3);
			updateNeighborlist(d_neighborlist, -nOrg, encode_tri(index, ori));
		}

		if (nApex > 0) {
			nApex = -(nApex - 3);
			updateNeighborlist(d_neighborlist, -nApex, encode_tri(pOppTri, pOppOri));
		}

		if (npDest > 0) {
			npDest = -(npDest - 3);
			updateNeighborlist(d_neighborlist, -npDest, encode_tri(index, (ori + 1) % 3));
		}

		if (npApex > 0) {
			npApex = -(npApex - 3);
			updateNeighborlist(d_neighborlist, -npApex, encode_tri(pOppTri, (pOppOri + 2) % 3));
		}

		// Update my own links
		d_neighborlist[index * 3 + ori] = -nOrg;
		d_neighborlist[index * 3 + (ori + 1) % 3] = -npDest;
		d_neighborlist[index * 3 + (ori + 2) % 3] = encode_tri(pOppTri, (pOppOri + 1) % 3);
		d_neighborlist[pOppTri * 3 + pOppOri] = -nApex;
		d_neighborlist[pOppTri * 3 + (pOppOri + 1) % 3] = encode_tri(index, (ori + 2) % 3);
		d_neighborlist[pOppTri * 3 + (pOppOri + 2) % 3] = -npApex;

		// Update encroached marker
		int sOrg, sDest, spOrg, spApex;
		sOrg = d_tri2subseg[3 * index + ori];
		sDest = d_tri2subseg[3 * index + (ori + 1) % 3];
		spOrg = d_tri2subseg[3 * pOppTri + pOppOri];
		spApex = d_tri2subseg[3 * pOppTri + (pOppOri + 2) % 3];

		if (sOrg != -1)
		{
			otri0 = (index << 2) | ori;
			otri1 = -nOrg;
			resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sOrg, otri0, otri1, theta, encmode);
		}

		if (sDest != -1)
		{
			otri0 = (index << 2) | ((ori + 1) % 3);
			otri1 = -npDest;
			resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sDest, otri0, otri1, theta, encmode);
		}

		if (spOrg != -1)
		{
			otri0 = (pOppTri << 2) | pOppOri;
			otri1 = -nApex;
			resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, spOrg, otri0, otri1, theta, encmode);
		}

		if (spApex != -1)
		{
			otri0 = (pOppTri << 2) | ((pOppOri + 2) % 3);
			otri1 = -npApex;
			resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, spApex, otri0, otri1, theta, encmode);
		}

		// mark neighbor
		d_TStatus[pOppTri].setFlip(true);
	}
	else if (ori < 6) // 3-1 flip
	{
		ori -= 3;
		ori = (ori + 1) % 3;

		int t1 = d_neighborlist[index * 3 + ori]; //cw
		int t2 = d_neighborlist[index * 3 + (ori + 2) % 3];//ccw
		int t1_ori = decode_ori(t1);
		int t2_ori = decode_ori(t2);
		t1 = decode_tri(t1);
		t2 = decode_tri(t2);

		nOrg = d_linkslot[(4 * pos + 1) * 3 + (t1_ori + 2) % 3];
		nDest = d_linkslot[4 * pos * 3 + (ori + 1) % 3];
		nApex = d_linkslot[(4 * pos + 2) * 3 + (t2_ori + 1) % 3];

		if (nOrg > 0) {        // My neighbor do not update me, update him
			nOrg = -(nOrg - 3);
			updateNeighborlist(d_neighborlist, -nOrg, encode_tri(index, ori));
		}

		if (nDest > 0) {
			nDest = -(nDest - 3);
			updateNeighborlist(d_neighborlist, -nDest, encode_tri(index, (ori + 1) % 3));
		}

		if (nApex > 0) {
			nApex = -(nApex - 3);
			updateNeighborlist(d_neighborlist, -nApex, encode_tri(index, (ori + 2) % 3));
		}

		// Update my own neighbors
		d_neighborlist[index * 3 + ori] = -nOrg;
		d_neighborlist[index * 3 + (ori + 1) % 3] = -nDest;
		d_neighborlist[index * 3 + (ori + 2) % 3] = -nApex;

		// Update encroached marker
		int sOrg, sDest, sApex;
		sOrg = d_tri2subseg[3 * index + ori];
		sDest = d_tri2subseg[3 * index + (ori + 1) % 3];
		sApex = d_tri2subseg[3 * index + (ori + 2) % 3];

		if (sOrg != -1)
		{
			otri0 = (index << 2) | ori;
			otri1 = -nOrg;
			resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sOrg, otri0, otri1, theta, encmode);
		}

		if (sDest != -1)
		{
			otri0 = (index << 2) | ((ori + 1) % 3);;
			otri1 = -nDest;
			resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sDest, otri0, otri1, theta, encmode);
		}

		if (sApex != -1)
		{
			otri0 = (index << 2) | ((ori + 2) % 3);;
			otri1 = -nApex;
			resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sApex, otri0, otri1, theta, encmode);
		}

		d_TStatus[t1].setFlip(false);
		d_TStatus[t2].setFlip(false);
	}
	else // 4-2 flip
	{
		ori -= 6;

		pOpp = d_neighborlist[index * 3 + ori];
		pOppTri = decode_tri(pOpp);
		pOppOri = decode_ori(pOpp);

		pOrg = d_trianglelist[index * 3 + (ori + 1) % 3];
		pDest = d_trianglelist[index * 3 + (ori + 2) % 3];
		pOpp = d_trianglelist[pOppTri * 3 + pOppOri];

		int t1, t2, t1_ori, t2_ori;

		// triangle already changed
		// in order to test direction, need to get current neighbor's direction
		bool outward = false;
		int min = MAXINT;

		int p[] = {
			d_trianglelist[3 * pOppTri],
			d_trianglelist[3 * pOppTri + 1],
			d_trianglelist[3 * pOppTri + 2]
		};

		for (int i = 0; i<3; i++)
		{
			if (d_PStatus[p[i]].isReduntant() && p[i] < min)
				min = p[i];
		}

		if (min == p[(pOppOri + 2) % 3]) // neighbor is inward, so I am outward
			outward = true;

		if (outward) //outward direction
		{
			t1 = d_neighborlist[index * 3 + (ori + 2) % 3];
			t2 = d_neighborlist[pOppTri * 3 + (pOppOri + 1) % 3];
			t1_ori = decode_ori(t1);
			t2_ori = decode_ori(t2);
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			nOrg = d_linkslot[(4 * pos + 1) * 3 + (pOppOri + 2) % 3];
			nDest = d_linkslot[4 * pos * 3 + (ori + 1) % 3];
			npDest = d_linkslot[(4 * pos + 2) * 3 + (t1_ori + 1) % 3];
			npApex = d_linkslot[(4 * pos + 3) * 3 + (t2_ori + 2) % 3];

			if (nOrg > 0) {        // My neighbor do not update me, update him
				nOrg = -(nOrg - 3);
				updateNeighborlist(d_neighborlist, -nOrg, encode_tri(index, ori));
			}

			if (nDest > 0) {
				nDest = -(nDest - 3);
				updateNeighborlist(d_neighborlist, -nDest, encode_tri(index, (ori + 1) % 3));
			}

			if (npDest > 0) {
				npDest = -(npDest - 3);
				updateNeighborlist(d_neighborlist, -npDest, encode_tri(t1, (t1_ori + 1) % 3));
			}

			if (npApex > 0) {
				npApex = -(npApex - 3);
				updateNeighborlist(d_neighborlist, -npApex, encode_tri(t1, (t1_ori + 2) % 3));
			}

			// Update my own links
			d_neighborlist[index * 3 + ori] = -nOrg;
			d_neighborlist[index * 3 + (ori + 1) % 3] = -nDest;
			d_neighborlist[t1 * 3 + (t1_ori + 1) % 3] = -npDest;
			d_neighborlist[t1 * 3 + (t1_ori + 2) % 3] = -npApex;

			// Update encroached marker
			int sOrg, sDest, spDest, spApex;
			sOrg = d_tri2subseg[3 * index + ori];
			sDest = d_tri2subseg[3 * index + (ori + 1) % 3];
			spDest = d_tri2subseg[3 * t1 + (t1_ori + 1) % 3];
			spApex = d_tri2subseg[3 * t1 + (t1_ori + 2) % 3];

			if (sOrg != -1)
			{
				otri0 = (index << 2) | ori;
				otri1 = -nOrg;
				resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sOrg, otri0, otri1, theta, encmode);
			}

			if (sDest != -1)
			{
				otri0 = (index << 2) | ((ori + 1) % 3);
				otri1 = -nDest;
				resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sDest, otri0, otri1, theta, encmode);
			}

			if (spDest != -1)
			{
				otri0 = (t1 << 2) | ((t1_ori + 1) % 3);
				otri1 = -npDest;
				resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, spDest, otri0, otri1, theta, encmode);
			}

			if (spApex != -1)
			{
				otri0 = (t1 << 2) | ((t1_ori + 2) % 3);
				otri1 = -npApex;
				resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, spApex, otri0, otri1, theta, encmode);
			}

		}
		else
		{
			t1 = d_neighborlist[index * 3 + (ori + 1) % 3];
			t2 = d_neighborlist[pOppTri * 3 + (pOppOri + 2) % 3];
			t1_ori = decode_ori(t1);
			t2_ori = decode_ori(t2);
			t1 = decode_tri(t1);
			t2 = decode_tri(t2);

			nOrg = d_linkslot[(4 * pos + 1) * 3 + (pOppOri + 1) % 3];
			nApex = d_linkslot[4 * pos * 3 + (ori + 2) % 3];
			npDest = d_linkslot[(4 * pos + 3) * 3 + (t2_ori + 1) % 3];
			npApex = d_linkslot[(4 * pos + 2) * 3 + (t1_ori + 2) % 3];

			if (nOrg > 0) {        // My neighbor do not update me, update him
				nOrg = -(nOrg - 3);
				updateNeighborlist(d_neighborlist, -nOrg, encode_tri(index, ori));
			}

			if (nApex > 0) {
				nApex = -(nApex - 3);
				updateNeighborlist(d_neighborlist, -nApex, encode_tri(index, (ori + 2) % 3));
			}

			if (npDest > 0) {
				npDest = -(npDest - 3);
				updateNeighborlist(d_neighborlist, -npDest, encode_tri(t1, (t1_ori + 1) % 3));
			}

			if (npApex > 0) {
				npApex = -(npApex - 3);
				updateNeighborlist(d_neighborlist, -npApex, encode_tri(t1, (t1_ori + 2) % 3));
			}
			// Update my own links
			d_neighborlist[index * 3 + ori] = -nOrg;
			d_neighborlist[index * 3 + (ori + 2) % 3] = -nApex;
			d_neighborlist[t1 * 3 + (t1_ori + 1) % 3] = -npDest;
			d_neighborlist[t1 * 3 + (t1_ori + 2) % 3] = -npApex;

			// Update encroached marker
			int sOrg, sApex, spDest, spApex;
			sOrg = d_tri2subseg[3 * index + ori];
			sApex = d_tri2subseg[3 * index + (ori + 2) % 3];
			spDest = d_tri2subseg[3 * t1 + (t1_ori + 1) % 3];
			spApex = d_tri2subseg[3 * t1 + (t1_ori + 2) % 3];

			if (sOrg != -1)
			{
				otri0 = (index << 2) | ori;
				otri1 = -nOrg;
				resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sOrg, otri0, otri1, theta, encmode);
			}

			if (sApex != -1)
			{
				otri0 = (index << 2) | ((ori + 2) % 3);
				otri1 = -nApex;
				resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, sApex, otri0, otri1, theta, encmode);
			}

			if (spDest != -1)
			{
				otri0 = (t1 << 2) | ((t1_ori + 1) % 3);
				otri1 = -npDest;
				resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, spDest, otri0, otri1, theta, encmode);
			}

			if (spApex != -1)
			{
				otri0 = (t1 << 2) | ((t1_ori + 2) % 3);
				otri1 = -npApex;
				resetEncseg(d_trianglelist, d_pointlist, d_encmarker, d_segmarker, spApex, otri0, otri1, theta, encmode);
			}
		}

		d_TStatus[t1].setFlip(true);

		d_TStatus[pOppTri].setFlip(false);
		d_TStatus[t2].setFlip(false);
	}
}

__global__ void kernelMarkDiametralCircle
(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int * d_trianglelist,
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_active,
	int numberofactive
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numberofactive) return;

	int sindex = d_active[pos]; // bad subseg index

	int pOrg, pDest, pApex;
	REAL2 vOrg, vDest, vApex;
	int otri, triIndex, triOri;

	// check the first triangle
	otri = d_subseg2tri[sindex];
	triIndex = decode_tri(otri);
	triOri = decode_ori(otri);

	pOrg = d_trianglelist[3 * triIndex + (triOri + 1) % 3];
	pDest = d_trianglelist[3 * triIndex + (triOri + 2) % 3];
	pApex = d_trianglelist[3 * triIndex + triOri];
	vOrg = d_pointlist[pOrg];
	vDest = d_pointlist[pDest];
	vApex = d_pointlist[pApex];

	// free steiner points and inside the diametral circle
	if (d_PStatus[pApex].isSteiner() && !d_PStatus[pApex].isSegmentSplit() &&
		((vOrg.x - vApex.x)*(vDest.x - vApex.x) + (vOrg.y - vApex.y)*(vDest.y - vApex.y) < 0.0))
	{
		// mark this point as reduntant
		d_PStatus[pApex].setRedundant();

		// mark incident triangles for flippable
		markFan(d_trianglelist, d_neighborlist, d_TStatus, triIndex, (triOri + 2) % 3);
	}

	// check the triangle on the other side
	otri = d_neighborlist[3 * triIndex + triOri];
	if (otri != -1)
	{
		triIndex = decode_tri(otri);
		triOri = decode_ori(otri);
		pApex = d_trianglelist[3 * triIndex + triOri];
		vApex = d_pointlist[pApex];

		// free steiner points and inside the diametral circle
		if (d_PStatus[pApex].isSteiner() && !d_PStatus[pApex].isSegmentSplit() &&
			((vOrg.x - vApex.x)*(vDest.x - vApex.x) + (vOrg.y - vApex.y)*(vDest.y - vApex.y) < 0.0))
		{
			// mark this point as reduntant
			d_PStatus[pApex].setRedundant();

			// mark incident triangles for flippable
			markFan(d_trianglelist, d_neighborlist, d_TStatus, triIndex, (triOri + 2) % 3);
		}
	}
}

__global__ void kernelMarkReduntantSteiner
(
	 PStatus 	*d_PStatus,
	 int		*d_trianglelist,
	 int        *d_neighborlist,
	 TStatus	*d_TStatus, 
	 int		*d_flipBy, 
	 int		*d_active,
	 int		numberofactive
 ) 
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if ( pos >= numberofactive ) return ;

	int index = d_active[pos];
	int otri	= d_flipBy[index];
	otri = negate(otri);

	const int triIndex = decode_tri(otri); 
	const int triOri = decode_ori(otri);

	if(triIndex != index) // flip by others
		d_TStatus[index].setFlip(true);

	int pindex = d_trianglelist[3*triIndex + (triOri+1)%3];
	d_PStatus[pindex].setRedundant();

	markFan(d_trianglelist,d_neighborlist,d_TStatus,triIndex,triOri);
}

__global__ void kernelResetMidInsertionMarker(
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_trimarker,
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int tindex = d_internallist[pos];
	int sindex = d_internalmarker[tindex];

	int otri,triIndex,triOri,n[3];

	// reset the first triangle and its neighbors
	otri = d_subseg2tri[sindex];
	triIndex = decode_tri(otri);
	triOri = decode_ori(otri);
	d_trimarker[triIndex] = MAXINT;
	d_TStatus[triIndex].setFlipOri(15); // indicate no change
	n[0] = decode_tri(d_neighborlist[3*triIndex]);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
	n[1] = decode_tri(d_neighborlist[3*triIndex+1]);
	n[2] = decode_tri(d_neighborlist[3*triIndex+2]);
	for(int i=0; i<3; i++)
	{
		if(n[i] != -1)
			d_TStatus[n[i]].setFlipOri(15);
	}

	// reset the triangle on the other side
	otri = d_neighborlist[3*triIndex + triOri];
	if(otri != -1)
	{
		triIndex = decode_tri(otri);
		d_trimarker[triIndex] = MAXINT;
		d_TStatus[triIndex].setFlipOri(15); // indicate no change
		n[0] = decode_tri(d_neighborlist[3*triIndex]);
		n[1] = decode_tri(d_neighborlist[3*triIndex+1]);
		n[2] = decode_tri(d_neighborlist[3*triIndex+2]);
		for(int i=0; i<3; i++)
		{
			if(n[i] != -1)
				d_TStatus[n[i]].setFlipOri(15);
		}
	}
}

__global__ void kernelMarkMidInsertion(
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_trimarker,
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive,
	int * d_trianglelist,
	int * d_tri2subseg,
	int iteration
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int tindex = d_internallist[pos];
	int sindex = d_internalmarker[tindex];

	int otri,triIndex,triOri;

	// reset the first triangle and its neighbors
	otri = d_subseg2tri[sindex];
	triIndex = decode_tri(otri);
	triOri = decode_ori(otri);
	atomicMin(d_trimarker + triIndex,sindex);

	// reset the triangle on the other side
	otri = d_neighborlist[3*triIndex + triOri];
	if(otri != -1)
	{
		triIndex = decode_tri(otri);
		atomicMin(d_trimarker + triIndex,sindex);
	}
}

__device__ void insertTriangle(int *d_trianglelist, int *d_neighborlist,
	int v0, int v1, int v2, int tri, int nApex, int nOrg, int nDest)
{
	if ( tri >= 0 ){
  		d_trianglelist[tri * 3] = v0;
		d_trianglelist[tri * 3 + 1] = v1;
		d_trianglelist[tri * 3 + 2] = v2;
		d_neighborlist[tri * 3 + 0] = nApex;
		d_neighborlist[tri * 3 + 1] = nOrg;
		d_neighborlist[tri * 3 + 2] = nDest;
	}
}

__global__ void kernelInsertMidPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int	* d_trianglelist, 
	int * d_neighborlist,
    int	* d_tri2subseg,
	TStatus * d_TStatus,
	int * d_segmentlist,
	int * d_subseg2tri,
	int * d_subseg2seg,
	int * d_encmarker,
	int * d_internalmarker,
	int * d_internallist,
	int * d_trimarker,
	int * d_emptypoints,
	int * d_emptytriangles,
	int emptypointsLength,
	int emptytrianglesLength,
	int numberofemptypoints,
	int numberofemptytriangles,
	int numberofsubseg,
	int numberofactive,
	int run_mode,
	REAL theta,
	int iteration
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int tindex = d_internallist[pos];
	int sindex = d_internalmarker[tindex];

	int tri[4],p[3],neighbor[3],seg[3];
	REAL2 v[4];

	int otri,triOri[2];
	bool acuteorg = false; // indicate if incident to segments
	bool acutedest = false; // indicate if incident to segments
	// check if I win
	// first triangle
	otri = d_subseg2tri[sindex];
	tri[0] = decode_tri(otri);
	triOri[0] = decode_ori(otri); // first triangle orientation
	if(d_trimarker[tri[0]] != sindex) // lose
		return;
	if(d_tri2subseg[3*tri[0]+ (triOri[0]+1)%3] != -1)
		acutedest = true;
	if(d_tri2subseg[3*tri[0]+ (triOri[0]+2)%3] != -1)
		acuteorg = true;

	// second triangle
	otri = d_neighborlist[3*tri[0] + triOri[0]];
	tri[2] = decode_tri(otri);
	triOri[1] = decode_ori(otri);
	if(tri[2] != -1 && d_trimarker[tri[2]] != sindex)
		return;
	if(tri[2] != -1)
	{
		if(d_tri2subseg[3*tri[2]+ (triOri[1]+1)%3] != -1)
			acuteorg = true;
		if(d_tri2subseg[3*tri[2]+ (triOri[1]+2)%3] != -1)
			acutedest = true;
	}

	// configure new point and triangles indices, status and markers
	int newID_point, newID_tri1, newID_tri2;
	newID_point = d_emptypoints[emptypointsLength - numberofemptypoints + tindex];
	newID_tri1  = d_emptytriangles[emptytrianglesLength - numberofemptytriangles + 2 * tindex];
	newID_tri2  = d_emptytriangles[emptytrianglesLength - numberofemptytriangles + 2 * tindex+1];
	d_PStatus[newID_point].createNewSegmentSplit();
	tri[1] = newID_tri1;
	tri[3] = newID_tri2;
	d_TStatus[tri[1]].clear(); // reset information
	d_TStatus[tri[3]].clear();
	d_encmarker[sindex] = -1;
	d_encmarker[numberofsubseg + tindex] = -1;

	// first old and new
	d_TStatus[tri[0]].setCheck(true); // first triangle
	d_TStatus[tri[0]].setFlip(true);  // mark for flip-flop
	d_TStatus[tri[0]].setFlipOri(0);  // mark for change
	d_TStatus[tri[1]].setNull(false); // first new triangle
	d_TStatus[tri[1]].setCheck(true);
	d_TStatus[tri[1]].setFlip(true);
	d_TStatus[tri[1]].setFlipOri(0);

	// insert mid point
	for ( int i = 0; i < 3; i++ )
			p[i] = d_trianglelist[tri[0] * 3 + (triOri[0] + i + 1)%3];

	REAL segmentlength = 
		sqrt((d_pointlist[p[0]].x - d_pointlist[p[1]].x) * (d_pointlist[p[0]].x - d_pointlist[p[1]].x) +
					(d_pointlist[p[0]].y - d_pointlist[p[1]].y) * (d_pointlist[p[0]].y - d_pointlist[p[1]].y));

	REAL split = 0.5; // split ratio
	if(acuteorg || acutedest)
	{
		REAL nearestpoweroftwo = 1.0;
		while(segmentlength > 3.0 * nearestpoweroftwo)
		{
			nearestpoweroftwo *= 2.0;
		}
		while(segmentlength < 1.5 * nearestpoweroftwo)
		{
			nearestpoweroftwo *= 0.5;
		}
		split = nearestpoweroftwo / segmentlength;
		if(acutedest)
			split = 1.0 - split;
	}

	d_pointlist[newID_point] = 
		MAKE_REAL2(
		d_pointlist[p[0]].x + (d_pointlist[p[1]].x - d_pointlist[p[0]].x) * split, 
		d_pointlist[p[0]].y + (d_pointlist[p[1]].y - d_pointlist[p[0]].y) * split);

	REAL multiplier = cuda_fast(d_pointlist[p[0]], d_pointlist[p[1]],d_pointlist[newID_point]);
#ifndef GQM2D_INEXACT
	if(multiplier == 0.0)
		multiplier = cuda_ccw(d_pointlist[p[0]], d_pointlist[p[1]], d_pointlist[newID_point]);
#endif
	REAL   divisor = ((d_pointlist[p[0]].x - d_pointlist[p[1]].x) * (d_pointlist[p[0]].x - d_pointlist[p[1]].x) +
					(d_pointlist[p[0]].y - d_pointlist[p[1]].y) * (d_pointlist[p[0]].y - d_pointlist[p[1]].y));
          
	if ( (multiplier != 0.0) && (divisor != 0.0) ) 
	{         
		multiplier = multiplier / divisor;
		/* Watch out for NANs. */
		if ( multiplier == multiplier ) 
		{
			d_pointlist[newID_point].x += multiplier * (d_pointlist[p[1]].y - d_pointlist[p[0]].y);
			d_pointlist[newID_point].y += multiplier * (d_pointlist[p[0]].x - d_pointlist[p[1]].x);			 
		}
	}

	// update triangles and subsegs relation
	for(int i=0; i<3; i++)
		seg[i] = d_tri2subseg[3*tri[0]+ (triOri[0]+i)%3];

	d_tri2subseg[3*tri[0]] = seg[2];   // old segment
	d_tri2subseg[3*tri[0]+1] = seg[0]; // old segment
	d_tri2subseg[3*tri[0]+2] = -1;     // new edge
	d_tri2subseg[3*tri[1]] = seg[1];   // old segment
	d_tri2subseg[3*tri[1]+1] = -1;	   // new edge
	d_tri2subseg[3*tri[1]+2] = numberofsubseg + tindex; // new segment
	
	d_subseg2tri[seg[0]] = (tri[0] << 2) | 1;
	d_subseg2tri[numberofsubseg + tindex] = (tri[1] << 2) | 2;
	d_subseg2seg[numberofsubseg + tindex] = d_subseg2seg[seg[0]]; // copy my parent to you
	if(seg[1] != -1)
		d_subseg2tri[seg[1]] = tri[1] << 2 | 0;
	if(seg[2] != -1)
		d_subseg2tri[seg[2]] = tri[0] << 2 | 0;
	// update encroched marker
	v[0] = d_pointlist[p[0]];
	v[1] = d_pointlist[p[1]];
	v[2] = d_pointlist[p[2]];
	v[3] = d_pointlist[newID_point];
	if(checkseg4encroach(v[0],v[3],v[2],theta,run_mode))
		d_encmarker[sindex] = 0;
	if(checkseg4encroach(v[3],v[1],v[2],theta,run_mode))
		d_encmarker[numberofsubseg + tindex] = 0;
	// update and insert triangles
	neighbor[0] = d_neighborlist[tri[0] * 3 + triOri[0]];		
	neighbor[1] = d_neighborlist[tri[0] * 3 + (triOri[0] + 1)%3];
	neighbor[2] = d_neighborlist[tri[0] * 3 + (triOri[0] + 2)%3];
	if(tri[2] == -1)
	{
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[2],p[0],
			tri[0],neighbor[2],-1,tri[1] << 2 | 1);
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[1],p[2],
			tri[1],neighbor[1],tri[0] << 2 | 2, -1);
	}
	else
	{
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[2],p[0],
			tri[0],neighbor[2],tri[2] << 2 | 2,tri[1] << 2 | 1);
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[1],p[2],
			tri[1],neighbor[1],tri[0] << 2 | 2,tri[3] << 2 | 1);
	}

	// second old and new
	if(tri[2] != -1)
	{
		d_TStatus[tri[2]].setCheck(true); // second triangle
		d_TStatus[tri[2]].setFlip(true);  // mark for flip-flop
		d_TStatus[tri[2]].setFlipOri(0);  // mark for change
		d_TStatus[tri[3]].setNull(false); // second new triangle
		d_TStatus[tri[3]].setCheck(true);
		d_TStatus[tri[3]].setFlip(true);
		d_TStatus[tri[3]].setFlipOri(0);

		// update triangles and subsegs relation
		for(int i=0; i<3; i++)
			seg[i] = d_tri2subseg[3*tri[2]+ (triOri[1]+i)%3];
		d_tri2subseg[3*tri[2]] = seg[1];   // old segment
		d_tri2subseg[3*tri[2]+1] = -1;     // new edge
		d_tri2subseg[3*tri[2]+2] = seg[0]; // old segment
		d_tri2subseg[3*tri[3]] = seg[2];   // old segment
		d_tri2subseg[3*tri[3]+1] = numberofsubseg + tindex; // new segment
		d_tri2subseg[3*tri[3]+2] = -1;

		if(seg[1] != -1)
			d_subseg2tri[seg[1]] = tri[2] << 2 | 0;
		if(seg[2] != -1)
			d_subseg2tri[seg[2]] = tri[3] << 2 | 0;
		// update encroched marker
		for ( int i = 0; i < 3; i++ )
			p[i] = d_trianglelist[tri[2] * 3 + (triOri[1] + i + 1)%3];
		v[0] = d_pointlist[p[0]];
		v[1] = d_pointlist[p[1]];
		v[2] = d_pointlist[p[2]];
		v[3] = d_pointlist[newID_point];
		if(checkseg4encroach(v[0],v[3],v[2],theta,run_mode))
			d_encmarker[numberofsubseg + tindex] = 0;
		if(checkseg4encroach(v[3],v[1],v[2],theta,run_mode))
			d_encmarker[sindex] = 0;
		// update and insert triangles
		neighbor[0] = d_neighborlist[tri[2] * 3 + triOri[1]];		
		neighbor[1] = d_neighborlist[tri[2] * 3 + (triOri[1] + 1)%3];
		neighbor[2] = d_neighborlist[tri[2] * 3 + (triOri[1] + 2)%3];
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[1],p[2],
			tri[2],neighbor[1],tri[3] << 2 | 2,tri[0] << 2 | 1);
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[2],p[0],
			tri[3],neighbor[2],tri[1] << 2 | 2,tri[2] << 2 | 1);
	}

	// reset internalmarker to -1
	d_internalmarker[tindex] = -1;
}

__global__ void kernelUpdateMidNeighbors
(
	REAL2 * d_pointlist,
	int	* d_trianglelist, 
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_encmarker,
	int * d_enclist,
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive,
	int run_mode,
	REAL theta
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int tindex = d_internallist[pos];
	int sindex = d_internalmarker[tindex];
	if(sindex != -1) // loser
		return;

	sindex = d_enclist[tindex];

	// find 4 triangles
	int otri,tri[4],p[4],iteration, pApex;
	REAL2 v[4];

	otri = d_subseg2tri[sindex];
	tri[0] = decode_tri(otri);
	otri = d_neighborlist[3*tri[0]+2];
	tri[1] = decode_tri(otri);

	otri = d_neighborlist[3*tri[0]+1];
	if(otri == -1)
		iteration = 2;
	else
		iteration = 4;
	tri[2] = decode_tri(otri);
	otri = d_neighborlist[3*tri[1]+2];
	tri[3] = decode_tri(otri);

	for ( int i = 0; i < iteration; i++ )//for each pTri[i]
	{
		int curTri	= tri[i]; 
		int seg = d_tri2subseg[3*curTri + 0];
		int pOppTri	= d_neighborlist[curTri * 3 + 0];
		int pOpp	= decode_tri(pOppTri); 
		int pOri    = decode_ori(pOppTri);

		if ( pOppTri < 0 ) // dont need to update neighbor, but need to update encroach marker
		{
			if(seg != -1)
			{
				d_encmarker[seg] = -1;
				p[0] = d_trianglelist[3*curTri + 1];
				v[0] = d_pointlist[p[0]];
				p[1] = d_trianglelist[3*curTri + 2];
				v[1] = d_pointlist[p[1]];
				pApex = d_trianglelist[3*curTri];
				v[2] = d_pointlist[pApex];

				if(checkseg4encroach(v[0],v[1],v[2],theta,run_mode))
					d_encmarker[seg] = 0;
			}
			continue;
		}

		if ( d_TStatus[pOpp].getFlipOri() == 15 )// neighbor[i] doesn't change
		{
			d_neighborlist[pOpp * 3 + pOri] = encode_tri(curTri,0);
			if(seg != -1)
			{
				d_encmarker[seg] = -1;
				p[0] = d_trianglelist[3*curTri + 1];
				v[0] = d_pointlist[p[0]];
				p[1] = d_trianglelist[3*curTri + 2];
				v[1] = d_pointlist[p[1]];
				pApex = d_trianglelist[3*curTri];
				v[2] = d_pointlist[pApex];
				pApex = d_trianglelist[3*pOpp + pOri];
				v[3] = d_pointlist[pApex];

				if(checkseg4encroach(v[0],v[1],v[2],theta,run_mode))
					d_encmarker[seg] = 0;
				else if(checkseg4encroach(v[1],v[0],v[3],theta,run_mode))
					d_encmarker[seg] = 0;
			}
		}
		else
		{	
			int neighbor;

			p[0] = d_trianglelist[curTri * 3 + 1];
			p[1] = d_trianglelist[curTri * 3 + 2];				

			for ( int j = 0; j < 3; j++ )
			{		
				if ( j==0 )					
					neighbor = pOpp; 
				else					
					neighbor = decode_tri(d_neighborlist[pOpp * 3 + j]);

				if ( neighbor > -1 )
				{						
					p[2] = d_trianglelist[neighbor * 3 + 1];
					p[3] = d_trianglelist[neighbor * 3 + 2];
										
					if ( p[0] == p[3] && p[1] == p[2] )
					{
						d_neighborlist[neighbor * 3 + 0] = encode_tri(curTri,0);

						if(seg != -1)
						{
							d_encmarker[seg] = -1;
							p[0] = d_trianglelist[3*curTri + 1];
							v[0] = d_pointlist[p[0]];
							p[1] = d_trianglelist[3*curTri + 2];
							v[1] = d_pointlist[p[1]];
							pApex = d_trianglelist[3*curTri];
							v[2] = d_pointlist[pApex];
							pApex = d_trianglelist[3*neighbor + 0];
							v[3] = d_pointlist[pApex];

							if(checkseg4encroach(v[0],v[1],v[2],theta,run_mode))
								d_encmarker[seg] = 0;
							else if(checkseg4encroach(v[1],v[0],v[3],theta,run_mode))
								d_encmarker[seg] = 0;
						}

						break;							
					}	
				}
				
			}			
		}
	}
}

__global__ void kernelUpdatePStatus2Old(
 PStatus * d_PStatus,
 int last_point
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= last_point ) return;

	if(!d_PStatus[pos].isDeleted())
		d_PStatus[pos].setOld();
}

void updatePStatus2Old
(
 PStatusD &t_PStatus,
 int last_point
)
{
	int numberofblocks = (ceil)((float)last_point / BLOCK_SIZE);
	kernelUpdatePStatus2Old<<<numberofblocks, BLOCK_SIZE>>>(
		thrust::raw_pointer_cast(&t_PStatus[0]),
		last_point);
}

/**																					**/
/**																					**/
/********* Mesh transformation routines end here							 *********/

/********* Mesh quality maintenance begins here								 *********/
/**																					**/
/**																					**/

__global__ void kernelMarkAllEncsegs(
	REAL2 * d_pointlist,
	int	* d_trianglelist,
	int * d_neighborlist,
	int * d_subseg2tri,
	int * d_encmarker,
	int numberofsubseg,
	int run_mode,
	REAL theta
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofsubseg)
		return;

	d_encmarker[pos] = -1;

	int pOrg,pDest,pApex;
	REAL2 vOrg, vDest, vApex;
	int otri,triIndex,triOri;

	// check the first triangle
	otri = d_subseg2tri[pos];
	triIndex = decode_tri(otri);
	triOri = decode_ori(otri);
	pOrg = d_trianglelist[3*triIndex + (triOri+1)%3];
	pDest = d_trianglelist[3*triIndex + (triOri+2)%3];
	pApex = d_trianglelist[3*triIndex + triOri];
	vOrg = d_pointlist[pOrg];
	vDest = d_pointlist[pDest];
	vApex = d_pointlist[pApex];

	if(checkseg4encroach(vOrg,vDest,vApex,theta,run_mode))
		d_encmarker[pos] = 0; // mark this subseg

	// check the triangle on the other side
	otri = d_neighborlist[3*triIndex + triOri];
	if(otri != -1)
	{
		triIndex = decode_tri(otri);
		triOri = decode_ori(otri);
		pApex = d_trianglelist[3*triIndex + triOri];
		vApex = d_pointlist[pApex];
		if(checkseg4encroach(vDest,vOrg,vApex,theta,run_mode))
			d_encmarker[pos] = 0;
	}
}

void markAllEncsegs(
	Real2D	&t_pointlist,
	IntD	&t_trianglelist,
	IntD	&t_neighborlist,
	IntD	&t_subseg2tri,
	IntD	&t_encmarker,
	int	numberofsubseg,
	int run_mode,
	REAL theta
)
{
	t_encmarker.resize(numberofsubseg);

	int numberofblocks = (ceil)((float)numberofsubseg / BLOCK_SIZE);
	kernelMarkAllEncsegs<<<numberofblocks, BLOCK_SIZE>>>(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_subseg2tri[0]),
		thrust::raw_pointer_cast(&t_encmarker[0]),
		numberofsubseg,
		run_mode,
		theta);
}

__device__ REAL triangle_area(REAL2 va, REAL2 vb, REAL2 vc)
{
	REAL Ax, Ay, Bx, By, Cx, Cy;
	Ax = va.x; Ay = va.y;
	Bx = vb.x; By = vb.y;
	Cx = vc.x; Cy = vc.y;
	REAL area;
	area = (Ax*(By - Cy) + Bx*(Cy - Ay) + Cx*(Ay - By)) / 2;
	if (area < 0)
		area = -area;
	return area;
}

__device__ REAL2 computeCircumcenter(REAL2 vApex, REAL2 vOrg, REAL2 vDest, REAL offConstant, bool offcenter,  REAL * priority)
{
	REAL2 circumcenter;	
	REAL xdo, ydo, xao, yao;
	REAL dodist, aodist, dadist;
	REAL denominator;
	REAL dx, dy, dxoff, dyoff;

	xdo = vDest.x - vOrg.x;
	ydo = vDest.y - vOrg.y;
	xao = vApex.x - vOrg.x;
	yao = vApex.y - vOrg.y;	  

	dodist = xdo * xdo + ydo * ydo;
	aodist = xao * xao + yao *yao;
	dadist = (vDest.x - vApex.x) * (vDest.x - vApex.x) + (vDest.y - vApex.y) * (vDest.y - vApex.y);

	REAL shortest, radius;

	REAL tmp = xdo * yao - xao * ydo;
	REAL deno = cuda_fast(vDest, vApex, vOrg);
#ifndef GQM2D_INEXACT
	if(deno == 0.0)
		deno = cuda_ccw(vDest, vApex, vOrg);
#endif

	if(tmp != 0)
		denominator = 0.5/tmp;	
	else		
		denominator = 0.5/deno;//use ccw () to ensure a positive (and reasonably accurate) result, avoiding any possibility of division by zero

	dx = (yao * dodist - ydo * aodist) * denominator;
	dy = (xdo * aodist - xao * dodist) * denominator;

	radius = dx*dx + dy*dy;

	if (offcenter && (offConstant > 0.0))/* Find the position of the off-center, as described by Alper Ungor. */
	{
		if ((dodist < aodist) && (dodist < dadist)) 
		{
			dxoff = 0.5 * xdo - offConstant * ydo;
			dyoff = 0.5 * ydo + offConstant * xdo;
			/* If the off-center is closer to the origin than the */
			/*   circumcenter, use the off-center instead.        */
			if (dxoff * dxoff + dyoff * dyoff < dx * dx + dy * dy) 
			{
				dx = dxoff;
				dy = dyoff;
			}
			shortest = dodist;
		}
		else if (aodist < dadist) 
		{
			dxoff = 0.5 * xao + offConstant * yao;
			dyoff = 0.5 * yao - offConstant * xao;
			if (dxoff * dxoff + dyoff * dyoff < dx * dx + dy * dy)
			{
				dx = dxoff;
				dy = dyoff;
			}
			shortest = aodist;
		} 
		else 
		{
			dxoff = 0.5 * (vApex.x - vDest.x) - offConstant * (vApex.y - vDest.y);
			dyoff = 0.5 * (vApex.y - vDest.y) + offConstant * (vApex.x - vDest.x);		

			if (dxoff * dxoff + dyoff * dyoff < (dx - xdo) * (dx - xdo) + (dy - ydo) * (dy - ydo))
			{
				dx = xdo + dxoff;
				dy = ydo + dyoff;
			}
			shortest = dadist;
		}
	}
	else // circumcenter
	{
		if ((dodist < aodist) && (dodist < dadist)) 
			shortest = dodist;
		else if (aodist < dadist)
			shortest = aodist;
		else
			shortest = dadist;

	}

	// Set priority to the triangle area
	*priority = triangle_area(vApex, vOrg, vDest);

	circumcenter.x = vOrg.x + dx;
	circumcenter.y = vOrg.y + dy;
	return circumcenter;
}

__global__ void kernelComputeCircumcenter(
	REAL2 * d_pointlist,
	int * d_trianglelist,
	int * d_active,
	int numberofactive,
	REAL2 * d_TCenter,
	int * d_Priority,
	REAL offConstant,
	bool offCenter
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int index = d_active[pos];

	int p[3] = {
		d_trianglelist[3*index],
		d_trianglelist[3*index+1],
		d_trianglelist[3*index+2]
	};

	REAL2 v[3] = {
		d_pointlist[p[0]],
		d_pointlist[p[1]],
		d_pointlist[p[2]]
	};

	// center coordinate
	REAL priority;
	REAL2 cc = computeCircumcenter(v[0], v[1], v[2], offConstant, offCenter, &priority); 
	d_TCenter[pos].x = cc.x;
	d_TCenter[pos].y = cc.y;

	d_Priority[pos] = __float_as_int(priority);
}

__global__ void kernelLocateSinkPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int * d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_TCenter,
#ifdef GQM2D_PRIORITY_SIZE
	uint64* d_trimarker,
#else
	int * d_trimarker,
#endif
	int * d_sinks,
	int * d_Priority,
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int index = d_internallist[pos];

	d_sinks[pos] = -1; // Initialized

	if(!d_TStatus[index].isBad())
		return;

#ifdef GQM2D_PRIORITY_SIZE
	uint64 marker = encodeUInt64Priority(d_Priority[pos], index + 1);
#endif

	REAL2 cc = d_TCenter[pos]; // circumcenter
	int priority = d_Priority[pos];

	int curTri = index; // current triangle index
	int curOri; // current triangle orientation

	int p[3];
	REAL2 v[3];
	REAL2 org;
	REAL det0,det1,det2;

	bool checkNeighbor = false;
	bool onvertex = false;
	bool split = false;
	int direction_dart = 3;

	// Consider following cases:
	// case 0: split current edge
	// case 1: move to neighbor
	// case 2: split current triangle

	// Check the bad triangle itself first
	p[0] = d_trianglelist[3*curTri];
	p[1] = d_trianglelist[3*curTri+1];
	p[2] = d_trianglelist[3*curTri+2];

	v[0] = d_pointlist[p[0]];
	v[1] = d_pointlist[p[1]];
	v[2] = d_pointlist[p[2]];

	// check if existing points
	//for(int i=0; i<3; i++)
	//{
	//	if(v[i].x == cc.x && v[i].y == cc.y)
	//		onvertex = true;
	//}

	//if(onvertex)
	//{
	//	printf("Existing vertex: my circumcenter equals to my vertex\n");
	//	d_internalmarker[pos] = -1;
	//	return;
	//}

	// three edges
	det0 = cuda_fast(v[0],v[1],cc);
#ifndef GQM2D_INEXACT
	if (det0 == 0) // not sure
		det0 = cuda_ccw(v[0],v[1],cc);
#endif

	det1 = cuda_fast(v[1],v[2],cc);
#ifndef GQM2D_INEXACT
	if (det1 == 0) // not sure
		det1 = cuda_ccw(v[1],v[2],cc);
#endif

	det2 = cuda_fast(v[2],v[0],cc);
#ifndef GQM2D_INEXACT
	if (det2 == 0) // not sure
		det2 = cuda_ccw(v[2],v[0],cc);
#endif

	// case 0: split current edge
	// Points are on the edges (segments) or
	// on the opposite side of segments
	// case 1: move to the neighbor
	if ( det0 <= 0 )
	{
		if(det0 == 0 || d_tri2subseg[3*index+2] != -1)
			split = true;
		direction_dart = 2;
	}
	else if ( det1 <= 0)
	{
		if(det1 == 0 || d_tri2subseg[3*index+0] != -1)
			split = true;
		direction_dart = 0;
	}
	else if ( det2 <= 0)
	{
		if(det2 == 0 || d_tri2subseg[3*index+1] != -1)
			split = true;
		direction_dart = 1;
	}

	if (!split && direction_dart!=3) // need to check neighbors of bad triangles
	{
		int neighbor = d_neighborlist[3*curTri+direction_dart];
		if(neighbor < 0) // circumcenter is out of boundary
		{
			checkNeighbor = false; // mark current triangle directly
			split = true;
		}
		else
		{
			org.x = v[direction_dart].x; // set up starting point
			org.y = v[direction_dart].y;
			curTri = decode_tri(neighbor); // move to the neighbor
			curOri = decode_ori(neighbor);
			checkNeighbor = true;
		}
	}

	REAL detLeft,detRight;
	bool moveleft;

	while(checkNeighbor) // trave from bad triangle to circumcenter
	{
		p[0] = d_trianglelist[3*curTri];
		p[1] = d_trianglelist[3*curTri+1];
		p[2] = d_trianglelist[3*curTri+2];

		v[0] = d_pointlist[p[0]];
		v[1] = d_pointlist[p[1]];
		v[2] = d_pointlist[p[2]];

		// onvertex check
		//if(v[curOri].x == cc.x && v[curOri].y == cc.y)
		//{
		//	printf("Tri %d: My circumcenter encountered duplicate vertex: %d(%lf,%lf), Steiner: %d, Midpoint: %d\n",
		//		index,
		//		p[curOri],v[curOri].x,v[curOri].y,
		//		d_PStatus[p[curOri]].isSteiner()? 1:0, 
		//		d_PStatus[p[curOri]].isSegmentSplit()? 1:0);
		//	onvertex = true;
		//	//d_PStatus[p[curOri]].setRedundant(); // mark for debugging
		//	//d_TStatus[index].setFlip(true);
		//	REAL2 vd[3] =
		//	{
		//		d_pointlist[d_trianglelist[3*index]],
		//		d_pointlist[d_trianglelist[3*index+1]],
		//		d_pointlist[d_trianglelist[3*index+2]]
		//	};
		//	printf("Tri %d: %d(%lf,%lf), %d(%lf,%lf), %d(%lf,%lf)\n",
		//		index, d_pointlist[d_trianglelist[3*index]],v[0].x,v[0].y,
		//		d_pointlist[d_trianglelist[3*index+1]],v[1].x,v[1].y,
		//		d_pointlist[d_trianglelist[3*index+2]],v[2].x,v[2].y);

		//	if( checkseg4encroach(v[0],v[1],v[2],15.1,0) ||
		//		checkseg4encroach(v[0],v[1],v[2],15.1,0) ||
		//		checkseg4encroach(v[0],v[1],v[2],15.1,0))
		//		printf("Tri %d: I contain an encroached segment\n",index);
		//	break;
		//}

		direction_dart = 3; // reset to default value

		int pOrg,pDest,pApex;
		pOrg = (curOri+1)%3;
		pDest = (curOri+2)%3;
		pApex = curOri;

		// left side
		detLeft = cuda_fast(v[pDest],v[pApex],cc);
#ifndef GQM2D_INEXACT
		if(detLeft == 0)
			detLeft = cuda_ccw(v[pDest],v[pApex],cc);
#endif

		// right side
		detRight = cuda_fast(v[pApex],v[pOrg],cc);
#ifndef GQM2D_INEXACT
		if(detRight == 0)
			detRight = cuda_ccw(v[pApex],v[pOrg],cc);
#endif

		if(detLeft < 0.0)
		{
			if(detRight < 0.0)
			{
				// at corner region, check inner product
				moveleft = (v[(curOri+2)%3].x - v[(curOri+1)%3].x)*(cc.x - v[curOri].x) +
					(v[(curOri+2)%3].y - v[(curOri+1)%3].y)*(cc.y - v[curOri].y) > 0.0;
			}
			else
			{
				moveleft = true;
			}
		}
		else
		{
			if(detRight < 0.0)
			{
				moveleft = false;
			}
			else
			{
				if(detLeft == 0.0)
				{
					split = true;
					direction_dart = (curOri+1)%3;
					break;
				}
				if(detRight == 0.0)
				{
					split = true;
					direction_dart = (curOri+2)%3;
					break;
				}
				break; // in triangle
			}
		}

		if(moveleft)
		{
			direction_dart = (curOri+1)%3;
		}
		else
		{
			direction_dart = (curOri+2)%3;
		}

		int neighbor = d_neighborlist[3*curTri+direction_dart];

		// circumcenter is out of boundary, mark current trianlge as segement split
		// or circumcenter is on the opposite side of segment
		if (neighbor < 0 || d_tri2subseg[3*curTri+direction_dart] != -1) 
		{
			split = true;
			break;
		}

		curTri = decode_tri(neighbor); // move to neighbor
		curOri = decode_ori(neighbor); // this edge must intersect segment 
	}

	//if(onvertex)
	//{
	//	printf("Existing vertex: Encounter same vertex when locating\n");
	//	d_internalmarker[pos] = -1;
	//	return;
	//}

#ifdef GQM2D_PRIORITY_SIZE
	atomicMax(&d_trimarker[curTri], marker);
#else
	atomicMin(&d_trimarker[curTri], index);
#endif

	//int old = atomicMin(&d_trimarker[curTri], priority);
	//if( old <= priority ) // lost already
	//	d_internalmarker[pos] = -1;

	if (split) // also mark the incident triangle
	{
		int neighbor = d_neighborlist[3*curTri+direction_dart];
		if(neighbor != -1) // prevent that steiner point happen to be on or out of the boundary
		{
			int neighborIndex = decode_tri(neighbor);
			int neighborOri = decode_ori(neighbor);
#ifdef GQM2D_PRIORITY_SIZE
			atomicMax(&d_trimarker[neighborIndex], marker);
#else
			atomicMin(&d_trimarker[neighborIndex], index);
#endif

			//old = atomicMin(&d_trimarker[neighborIndex], priority);
			//if( old <= priority ) // lost already
			//	d_internalmarker[pos] = -1;
		}
	}

	d_sinks[pos] = (curTri << 2) | direction_dart ;

	//if(direction_dart !=3)
	//	printf("%d ",d_tri2subseg[3*curTri+direction_dart]);
}

__global__ void kernelRemoveLosers
(
	 int		  *d_trianglelist, 
	 int          *d_neighborlist,
	 TStatus      *d_TStatus,
#ifdef GQM2D_PRIORITY_SIZE
	 uint64* d_trimarker,
#else
	 int * d_trimarker,
#endif
	 int		  *d_sinks,
	 int		  *d_Priority,
	 int          *d_internalmarker,
	 int		  *d_internallist,
	 int		  numberofactive 
 )
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if( pos >= numberofactive )
		return;

	if ( d_internalmarker[pos] < 0 ) // lost because of exsiting vertices
		return;

	int sink = d_sinks[pos] >> 2; 
	int index = d_internallist[pos];
	int priority = d_Priority[pos];

	int direction		= d_sinks[pos] & 3;

#ifdef GQM2D_PRIORITY_SIZE
	uint64 oldmarker;
	int old;
#else
	int winnerMarker = d_trimarker[sink];
#endif

#ifdef GQM2D_PRIORITY_SIZE
	oldmarker = d_trimarker[sink];
	old = getUInt64PriorityIndex(oldmarker) - 1;
	if(old != index)
#else
	if (winnerMarker != index)
#endif
	{
		d_internalmarker[pos] = -1; 
		return;
	}
	else if (direction != 3) // also need to check if neighbor won
	{
		int neighbor = d_neighborlist[3*sink+direction];
		if(neighbor != -1) // not on the boundary, need to check
		{
			int neighborIndex = decode_tri(neighbor);

#ifdef GQM2D_PRIORITY_SIZE
			oldmarker = d_trimarker[neighborIndex];
			old = getUInt64PriorityIndex(oldmarker) - 1;
			if (old != index)
#else
			int winnerMarker  = d_trimarker[neighborIndex];
			if (winnerMarker != index)
#endif
			//if(winnerMarker != priority) // remove
			{
				d_internalmarker[pos] = -1;
				return;
			}
		}
	}
}

__global__ void kernelMarkCavities(
	REAL2 * d_pointlist,
	int	* d_trianglelist, 
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_TCenter,
	int * d_sinks,
	int * d_Priority,
#ifdef GQM2D_PRIORITY_SIZE
	uint64* d_trimarker,
#else
	int * d_trimarker,
#endif
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive,
	int offset,
	int numofprocess
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if( pos >= numofprocess )
		return;

	if ( d_internalmarker[pos+offset] < 0 ) // lost because of locating competition
		return;

	int badIndex = d_internallist[pos+offset];
	int index = d_sinks[pos+offset] >> 2;
	int direction = d_sinks[pos+offset] & 3;
	int priority = d_Priority[pos+offset];

#ifdef GQM2D_PRIORITY_SIZE
	uint64 marker = encodeUInt64Priority(priority, badIndex + 1);
#endif

	REAL2 cc = d_TCenter[pos+offset]; // circumcenter

	// almong all these winner, mark the cavities for them

	// initialize record lists
	MyList region;
	MyList loop;

	// for current triangle
	int curIndex;
	curIndex = index;

	bool fail = false; // indicate if space is not enough

	while(true)
	{
		if(!region.find(curIndex))// record now triangle
		{
			if(!region.push(curIndex))
			{
				fail = true;
				break;
			}
			if(!loop.push(curIndex))
			{
				fail = true;
				break;
			}

#ifdef GQM2D_PRIORITY_SIZE
			atomicMax(&d_trimarker[curIndex], marker);
#else
			atomicMin(&d_trimarker[curIndex], badIndex);
#endif

			//int old = atomicMin(&d_trimarker[curIndex], priority);
			//if( old < priority ) // lost already
			//{
			//	//printf("old = %d, my = %d\n",old, bratio);
			//	d_internalmarker[pos+offset] = -1;
			//	break;
			//}
		}

		int neighbor, neighborIndex;
		int p[3];
		REAL2 v[3];

		bool recursive = false;

		for( int i=0; i<3; i++)
		{
			if(curIndex == index) // sink triangle
			{
				if(direction == 3 && d_tri2subseg[3*curIndex+i] != -1 ) // no segment split, don't need to check
					continue;

				if(direction < 3 && d_tri2subseg[3*curIndex+i] != -1 && i != direction) // if segment split, check that segment
					continue;
			}
			else if(d_tri2subseg[3*curIndex+i] != -1) // has segment edge: no need to check
				continue;

			neighbor = d_neighborlist[3*curIndex+i];

			if(neighbor == -1) // boundary: no need to check
				continue;

			neighborIndex = decode_tri(neighbor);

			if(region.find(neighborIndex)) // searched already: no need to check
				continue;

			p[0] = d_trianglelist[3*neighborIndex];
			p[1] = d_trianglelist[3*neighborIndex+1];
			p[2] = d_trianglelist[3*neighborIndex+2];
			v[0] = d_pointlist[p[0]];
			v[1] = d_pointlist[p[1]];
			v[2] = d_pointlist[p[2]];
		
			// in circle test
			REAL r = cuda_inCircle(v[0],v[1],v[2],cc);
#ifndef GQM2D_INEXACT
			if(r==0) // not sure
				r = cuda_inCircle_exact(v[0],v[1],v[2],cc);
#endif
		
			if(r>0) // inside circle: check recursively
			{
				recursive = true;
				break;
			}
		}
		if(recursive)
			curIndex = neighborIndex;
		else
		{
			loop.pop();
			if(loop.getSize() == 0) // empty, break the while loop
				break;
			else
				curIndex = loop.getLast(); // resume to previous triangle
		}
	}

	if(fail) // cavity is too large, win directly
		d_internallist[pos+offset] = negate(badIndex);
}

void markCavities(
	Real2D &t_pointlist,
	IntD &t_trianglelist, 
	IntD &t_neighborlist,
	IntD &t_tri2subseg,
	TStatusD &t_TStatus,
	Real2D &t_TCenter,
	IntD &t_sinks,
	IntD &t_Priority,
#ifdef GQM2D_PRIORITY_SIZE
	UInt64D &t_trimarker,
#else
	IntD &t_trimarker,
#endif
	IntD &t_internalmarker,
	IntD &t_internallist,
	int numberofbad,
	int maxnumofblock // for number of triangles
)
{
	int numberofprocess; // number of triangles processed per iteration
	int finishnum = 0; // numberof triangles finish already

	int numberofblocks;

	do
	{
		if(numberofbad - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberofbad - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelMarkCavities<<<numberofblocks, BLOCK_SIZE>>>(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_TCenter[0]),
			thrust::raw_pointer_cast(&t_sinks[0]),
			thrust::raw_pointer_cast(&t_Priority[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			thrust::raw_pointer_cast(&t_internalmarker[0]),
			thrust::raw_pointer_cast(&t_internallist[0]),
			numberofbad,
			finishnum,
			numberofprocess);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberofbad);
}

__global__ void kernelCheckCavities(
	REAL2 * d_pointlist,
	int	* d_trianglelist, 
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_TCenter,
	int * d_sinks,
	int * d_Priority,
#ifdef GQM2D_PRIORITY_SIZE
	uint64* d_trimarker,
#else
	int * d_trimarker,
#endif
	int * d_internalmarker,
	int * d_internallist,
	int numberofactive,
	int offset,
	int numofprocess
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if( pos >= numofprocess )
		return;

	if( d_internalmarker[pos+offset] < 0 ) // lost because of locating competition
		return;

	int badIndex = d_internallist[pos+offset];
	if(badIndex < 0) // cavity is too large, always insert this point
		return;

	int index = d_sinks[pos+offset] >> 2; // use the index for bad triangle to mark cavity
	int direction = d_sinks[pos+offset] & 3;
	int priority = d_Priority[pos+offset];

	REAL2 cc = d_TCenter[pos+offset]; // circumcenter

	// almong all these winner, check the cavities for them
	bool result = true; 

	// initialize record lists
	MyList region;
	MyList loop;

	// for current triangle
	int curIndex;
	curIndex = index;

	while(true)
	{
		if(!region.find(curIndex))// record now triangle
		{
			if(!region.push(curIndex)) // too large, break
				break;
			if(!loop.push(curIndex))
				break;

#ifdef GQM2D_PRIORITY_SIZE
			uint64 marker = d_trimarker[curIndex];
			int old = getUInt64PriorityIndex(marker) - 1;
			if (old != badIndex)
#else
			int marker = d_trimarker[curIndex];
			if(marker != badIndex)
#endif
			//if (marker != priority)
			{
				result = false;
				break;
			}
		}

		int neighbor, neighborIndex;
		int p[3];
		REAL2 v[3];

		bool recursive = false;

		for( int i=0; i<3; i++)
		{
			if(curIndex == index) // sink triangle
			{
				if(direction == 3 && d_tri2subseg[3*curIndex+i] != -1) // no segment split, don't need to check
					continue;

				if(direction < 3 && d_tri2subseg[3*curIndex+i] != -1 && i != direction) // if segment split, check that segment
					continue;
			}
			else if(d_tri2subseg[3*curIndex+i] != -1) // has segment edge: no need to check
				continue;

			neighbor = d_neighborlist[3*curIndex+i];

			if(neighbor == -1) // boundary: no need to check
				continue;

			neighborIndex = decode_tri(neighbor);

			if(region.find(neighborIndex)) // searched already: no need to check
				continue;

			p[0] = d_trianglelist[3*neighborIndex];
			p[1] = d_trianglelist[3*neighborIndex+1];
			p[2] = d_trianglelist[3*neighborIndex+2];
			v[0] = d_pointlist[p[0]];
			v[1] = d_pointlist[p[1]];
			v[2] = d_pointlist[p[2]];
		
			// in circle test
			REAL r = cuda_inCircle(v[0],v[1],v[2],cc);
#ifndef GQM2D_INEXACT
			if(r==0) // not sure
				r = cuda_inCircle_exact(v[0],v[1],v[2],cc);
#endif
		
			if(r>0) // inside circle: check recursively
			{
				recursive = true;
				break;
			}
		}
		if(recursive)
			curIndex = neighborIndex;
		else
		{
			loop.pop();
			if(loop.getSize() == 0) // empty, break the while loop
				break;
			else
				curIndex = loop.getLast(); // resume to previous triangle
		}
	}

	if(!result) // violation found, lose competition
		d_internalmarker[pos+offset] = -1;

}

void checkCavities(
	Real2D &t_pointlist,
	IntD &t_trianglelist, 
	IntD &t_neighborlist,
	IntD &t_tri2subseg,
	TStatusD &t_TStatus,
	Real2D &t_TCenter,
	IntD &t_sinks,
	IntD &t_Priority,
#ifdef GQM2D_PRIORITY_SIZE
	UInt64D &t_trimarker,
#else
	IntD &t_trimarker,
#endif
	IntD &t_internalmarker,
	IntD &t_internallist,
	int numberofbad,
	int maxnumofblock // for number of triangles
)
{
	int numberofprocess; // number of triangles processed per iteration
	int finishnum = 0; // numberof triangles finish already

	int numberofblocks = (ceil)((float)numberofbad / BLOCK_SIZE);

	do
	{
		if(numberofbad - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberofbad - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelCheckCavities<<<numberofblocks, BLOCK_SIZE>>>(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_TCenter[0]),
			thrust::raw_pointer_cast(&t_sinks[0]),
			thrust::raw_pointer_cast(&t_Priority[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
			thrust::raw_pointer_cast(&t_internalmarker[0]),
			thrust::raw_pointer_cast(&t_internallist[0]),
			numberofbad,
			finishnum,
			numberofprocess);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberofbad);
}

__global__ void kernelResetSteinerInsertionMarker(
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_internallist,
	int * d_sinks,
	int numberofactive
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int tindex = d_internallist[pos];
	
	int index = d_sinks[tindex] >> 2;
	int direction = d_sinks[tindex] & 3;

	int otri,n[3];

	// reset the first triangle and its neighbors
	d_TStatus[index].setFlipOri(15); // indicate no change
	n[0] = decode_tri(d_neighborlist[3*index]);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
	n[1] = decode_tri(d_neighborlist[3*index+1]);
	n[2] = decode_tri(d_neighborlist[3*index+2]);
	for(int i=0; i<3; i++)
	{
		if(n[i] != -1)
			d_TStatus[n[i]].setFlipOri(15);
	}

	// reset the triangle on the other side
	if(direction != 3)
	{
		otri = d_neighborlist[3*index + direction];
		if(otri != -1)
		{
			index = decode_tri(otri);
			d_TStatus[index].setFlipOri(15); // indicate no change
			n[0] = decode_tri(d_neighborlist[3*index]);
			n[1] = decode_tri(d_neighborlist[3*index+1]);
			n[2] = decode_tri(d_neighborlist[3*index+2]);
			for(int i=0; i<3; i++)
			{
				if(n[i] != -1)
					d_TStatus[n[i]].setFlipOri(15);
			}
		}
	}
}

__global__ void kernelInsertSteinerPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int	* d_trianglelist, 
	int * d_neighborlist,
    int	* d_tri2subseg,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_encmarker,
	int * d_internalmarker,
	int * d_internallist,
	REAL2 * d_TCenter,
	int	* d_sinks,
	int * d_emptypoints,
	int * d_emptytriangles,
	int emptypointsLength,
	int emptytrianglesLength,
	int numberofemptypoints,
	int numberofemptytriangles,
	int numberofsubseg,
	int numberofactive,
	REAL theta,
	int iteration
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int tindex = d_internallist[pos];
	
	int index = d_sinks[tindex] >> 2;
	int direction = d_sinks[tindex] & 3;

	if(direction != 3 && d_tri2subseg[3*index + direction] != -1) // steiner is on the segment
	{
		//printf("Return ");
		return; // mark later
	}

	int tri[4],p[3],neighbor[3],seg[3];
	REAL2 v[4];

	// configure new point and triangles indices, status and markers
	int newID_point, newID_tri1, newID_tri2;
	newID_point = d_emptypoints[emptypointsLength - numberofemptypoints + pos];
	d_PStatus[newID_point].createNewTriangleSplit();

	newID_tri1  = d_emptytriangles[emptytrianglesLength - numberofemptytriangles + 2 * pos];
	newID_tri2  = d_emptytriangles[emptytrianglesLength - numberofemptytriangles + 2 * pos+1];
	tri[0] = index;

	if(direction == 3)
	{
		// two new
		tri[1] = newID_tri1;
		tri[2] = newID_tri2;

		d_TStatus[tri[1]].clear();
		d_TStatus[tri[2]].clear();

		d_TStatus[tri[0]].setCheck(true); // first triangle
		d_TStatus[tri[0]].setFlip(true);  // mark for flip-flop
		d_TStatus[tri[0]].setFlipOri(0);  // mark for change

		d_TStatus[tri[1]].setNull(false); // first new triangle
		d_TStatus[tri[1]].setCheck(true);
		d_TStatus[tri[1]].setFlip(true);
		d_TStatus[tri[1]].setFlipOri(0);

		d_TStatus[tri[2]].setNull(false); // second new triangle
		d_TStatus[tri[2]].setCheck(true);
		d_TStatus[tri[2]].setFlip(true);
		d_TStatus[tri[2]].setFlipOri(0);

		// insert steiner point
		d_pointlist[newID_point] = d_TCenter[tindex];
		for ( int i = 0; i < 3; i++ )
			p[i] = d_trianglelist[tri[0] * 3 + (i+1)%3];

		// update triangles and subsegs relation
		for(int i=0; i<3; i++)
			seg[i] = d_tri2subseg[3*tri[0]+ i];
		d_tri2subseg[3*tri[0]] = seg[0];   // old segment
		d_tri2subseg[3*tri[0]+1] = -1;	   // new edge
		d_tri2subseg[3*tri[0]+2] = -1;     // new edge
		d_tri2subseg[3*tri[1]] = seg[1];   // old segment
		d_tri2subseg[3*tri[1]+1] = -1;	   // new edge
		d_tri2subseg[3*tri[1]+2] = -1;     // new edge
		d_tri2subseg[3*tri[2]] = seg[2];   // old segment
		d_tri2subseg[3*tri[2]+1] = -1;	   // new edge
		d_tri2subseg[3*tri[2]+2] = -1;     // new edge

		if(seg[0] != -1)
			d_subseg2tri[seg[0]] = tri[0] << 2 | 0;
		if(seg[1] != -1)
			d_subseg2tri[seg[1]] = tri[1] << 2 | 0;
		if(seg[2] != -1)
			d_subseg2tri[seg[2]] = tri[2] << 2 | 0;

		//// update encroched marker
		//v[0] = d_pointlist[p[0]];
		//v[1] = d_pointlist[p[1]];
		//v[2] = d_pointlist[p[2]];
		//v[3] = d_pointlist[newID_point];
		//if(checkseg4encroach(v[1],v[2],v[3],theta))
		//	d_encmarker[seg[1]] = 0;
		//if(checkseg4encroach(v[1],v[2],v[3],theta))
		//	d_encmarker[seg[1]] = 0;	
		//if(checkseg4encroach(v[0],v[2],v[3],theta))
		//	d_encmarker[seg[2]] = 0;

		// update and insert triangles
		for(int i=0; i<3; i++)
			neighbor[i] = d_neighborlist[tri[0]*3 + i];
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[0],p[1],
			tri[0],neighbor[0],tri[1] << 2 | 2,tri[2] << 2 | 1);
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[1],p[2],
			tri[1],neighbor[1],tri[2] << 2 | 2,tri[0] << 2 | 1);
		insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[2],p[0],
			tri[2],neighbor[2],tri[0] << 2 | 2,tri[1] << 2 | 1);

	}
	else // steiner is on the edge
	{
		int triOri[2], otri;
		triOri[0] = direction;
		otri = d_neighborlist[3*tri[0] + triOri[0]];
		triOri[1] = decode_ori(otri);

		// two new and one old
		tri[1] = newID_tri1;
		tri[2] = decode_tri(otri);
		tri[3] = newID_tri2;

		d_TStatus[tri[1]].clear(); // reset information
		d_TStatus[tri[3]].clear();

		// first old and new
		d_TStatus[tri[0]].setCheck(true); // first triangle
		d_TStatus[tri[0]].setFlip(true);  // mark for flip-flop
		d_TStatus[tri[0]].setFlipOri(0);  // mark for change

		d_TStatus[tri[1]].setNull(false); // first new triangle
		d_TStatus[tri[1]].setCheck(true);
		d_TStatus[tri[1]].setFlip(true);
		d_TStatus[tri[1]].setFlipOri(0);

		// insert steiner point on edge
		d_pointlist[newID_point] = d_TCenter[tindex];
		for ( int i = 0; i < 3; i++ )
			p[i] = d_trianglelist[tri[0] * 3 + (triOri[0] + i + 1)%3];

		// update triangles and subsegs relation
		for(int i=0; i<3; i++)
			seg[i] = d_tri2subseg[3*tri[0]+ (triOri[0]+i)%3];
		d_tri2subseg[3*tri[0]] = seg[2];   // old segment
		d_tri2subseg[3*tri[0]+1] = seg[0]; // old segment | new edge (should be -1)
		d_tri2subseg[3*tri[0]+2] = -1;     // new edge
		d_tri2subseg[3*tri[1]] = seg[1];   // old segment
		d_tri2subseg[3*tri[1]+1] = -1;	   // new edge
		d_tri2subseg[3*tri[1]+2] = seg[0]; // old segment | new edge (should be -1)
	
		if(seg[1] != -1)
			d_subseg2tri[seg[1]] = tri[1] << 2 | 0;
		if(seg[2] != -1)
			d_subseg2tri[seg[2]] = tri[0] << 2 | 0;

		//// update encroched marker
		//v[0] = d_pointlist[p[0]];
		//v[1] = d_pointlist[p[1]];
		//v[2] = d_pointlist[p[2]];
		//v[3] = d_pointlist[newID_point];
		//if(seg[2] != -1 && checkseg4encroach(v[0],v[2],v[3],theta))
		//	d_encmarker[seg[2]] = 0;
		//if(seg[1] != -1 && checkseg4encroach(v[1],v[2],v[3],theta))
		//	d_encmarker[seg[1]] = 0;

		// update and insert triangles
		neighbor[0] = d_neighborlist[tri[0] * 3 + triOri[0]];		
		neighbor[1] = d_neighborlist[tri[0] * 3 + (triOri[0] + 1)%3];
		neighbor[2] = d_neighborlist[tri[0] * 3 + (triOri[0] + 2)%3];
		if(tri[2] == -1)
		{
			insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[2],p[0],
				tri[0],neighbor[2],-1,tri[1] << 2 | 1);
			insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[1],p[2],
				tri[1],neighbor[1],tri[0] << 2 | 2, -1);
		}
		else
		{
			insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[2],p[0],
				tri[0],neighbor[2],tri[2] << 2 | 2,tri[1] << 2 | 1);
			insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[1],p[2],
				tri[1],neighbor[1],tri[0] << 2 | 2,tri[3] << 2 | 1);
		}

		// second old and new
		if(tri[2] != -1)
		{
			d_TStatus[tri[2]].setCheck(true); // second triangle
			d_TStatus[tri[2]].setFlip(true);  // mark for flip-flop
			d_TStatus[tri[2]].setFlipOri(0);  // mark for change

			d_TStatus[tri[3]].setNull(false); // second new triangle
			d_TStatus[tri[3]].setCheck(true);
			d_TStatus[tri[3]].setFlip(true);
			d_TStatus[tri[3]].setFlipOri(0);

			for ( int i = 0; i < 3; i++ )
				p[i] = d_trianglelist[tri[2] * 3 + (triOri[1] + i + 1)%3];

			// update triangles and subsegs relation
			for(int i=0; i<3; i++)
				seg[i] = d_tri2subseg[3*tri[2]+ (triOri[1]+i)%3];
			d_tri2subseg[3*tri[2]] = seg[1];   // old segment
			d_tri2subseg[3*tri[2]+1] = -1;     // new edge
			d_tri2subseg[3*tri[2]+2] = seg[0]; // old segment | new edge (should be -1)
			d_tri2subseg[3*tri[3]] = seg[2];   // old segment
			d_tri2subseg[3*tri[3]+1] = seg[0]; // old segment | new edge (should be -1)
			d_tri2subseg[3*tri[3]+2] = -1;

			if(seg[1] != -1)
				d_subseg2tri[seg[1]] = tri[2] << 2 | 0;
			if(seg[2] != -1)
				d_subseg2tri[seg[2]] = tri[3] << 2 | 0;

			//// update encroched marker
			
			//v[0] = d_pointlist[p[0]];
			//v[1] = d_pointlist[p[1]];
			//v[2] = d_pointlist[p[2]];
			//v[3] = d_pointlist[newID_point];
			//if(seg[2] != -1 && checkseg4encroach(v[0],v[2],v[3],theta))
			//	d_encmarker[seg[2]] = 0;
			//if(seg[1] != -1 && checkseg4encroach(v[1],v[2],v[3],theta))
			//	d_encmarker[seg[1]] = 0;

			// update and insert triangles
			neighbor[0] = d_neighborlist[tri[2] * 3 + triOri[1]];		
			neighbor[1] = d_neighborlist[tri[2] * 3 + (triOri[1] + 1)%3];
			neighbor[2] = d_neighborlist[tri[2] * 3 + (triOri[1] + 2)%3];
			insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[1],p[2],
				tri[2],neighbor[1],tri[3] << 2 | 2,tri[0] << 2 | 1);
			insertTriangle(d_trianglelist,d_neighborlist,newID_point,p[2],p[0],
				tri[3],neighbor[2],tri[1] << 2 | 2,tri[2] << 2 | 1);
		}

	}
}

__global__ void kernelUpdateSteinerNeighbors
(
	REAL2 * d_pointlist,
	int	* d_trianglelist, 
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	int * d_subseg2tri,
	int * d_encmarker,
	int * d_internalmarker,
	int * d_internallist,
	int * d_sinks,
	int numberofactive,
	int run_mode,
	REAL theta,
	int debug_iter
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int tindex = d_internallist[pos];
	
	int index = d_sinks[tindex] >> 2;
	int direction = d_sinks[tindex] & 3;

	if(direction != 3 && d_TStatus[index].getFlipOri() == 15) // steiner is on the segment
		return; // mark later

	// find 4 triangles
	int otri,tri[4],p[4],iteration, pApex;
	REAL2 v[4];

	if(direction == 3)
	{
		iteration = 3;

		tri[0] = index;
		otri = d_neighborlist[3*tri[0]+1];
		tri[1] = decode_tri(otri);
		otri = d_neighborlist[3*tri[0]+2];
		tri[2] = decode_tri(otri);
	}
	else
	{
		tri[0] = index;
		otri = d_neighborlist[3*tri[0]+2];
		tri[1] = decode_tri(otri);

		otri = d_neighborlist[3*tri[0]+1];
		if(otri == -1)
			iteration = 2;
		else
		{
			iteration = 4;
			tri[2] = decode_tri(otri);
			otri = d_neighborlist[3*tri[1]+2];
			tri[3] = decode_tri(otri);
		}
	}

	for ( int i = 0; i < iteration; i++ )//for each pTri[i]
	{
		int curTri	= tri[i]; 
		int seg = d_tri2subseg[3*curTri + 0];
		int pOppTri	= d_neighborlist[curTri * 3 + 0];
		int pOpp	= decode_tri(pOppTri); 
		int pOri    = decode_ori(pOppTri);
		
		if ( pOppTri < 0 ) // dont need to update neighbor, but need to update encroach marker
		{
			if(seg != -1)
			{
				d_encmarker[seg] = -1;
				p[0] = d_trianglelist[3*curTri + 1];
				v[0] = d_pointlist[p[0]];
				p[1] = d_trianglelist[3*curTri + 2];
				v[1] = d_pointlist[p[1]];
				pApex = d_trianglelist[3*curTri];
				v[2] = d_pointlist[pApex];

				if(checkseg4encroach(v[0],v[1],v[2],theta,run_mode))
					d_encmarker[seg] = 0;
			}
			continue;
		}

		if ( d_TStatus[pOpp].getFlipOri() == 15 )// neighbor[i] doesn't change
		{
			d_neighborlist[pOpp * 3 + pOri] = encode_tri(curTri,0);
			if(seg != -1)
			{
				d_encmarker[seg] = -1;
				p[0] = d_trianglelist[3*curTri + 1];
				v[0] = d_pointlist[p[0]];
				p[1] = d_trianglelist[3*curTri + 2];
				v[1] = d_pointlist[p[1]];
				pApex = d_trianglelist[3*curTri];
				v[2] = d_pointlist[pApex];
				pApex = d_trianglelist[3*pOpp + pOri];
				v[3] = d_pointlist[pApex];

				if(checkseg4encroach(v[0],v[1],v[2],theta,run_mode))
					d_encmarker[seg] = 0;
				else if(checkseg4encroach(v[1],v[0],v[3],theta,run_mode))
					d_encmarker[seg] = 0;
			}
		}
		else
		{	
			int neighbor;

			p[0] = d_trianglelist[curTri * 3 + 1];
			p[1] = d_trianglelist[curTri * 3 + 2];				

			for ( int j = 0; j < 3; j++ )
			{		
				if ( j==0 )					
					neighbor = pOpp; 
				else					
					neighbor = decode_tri(d_neighborlist[pOpp * 3 + j]);

				if ( neighbor > -1 )
				{						
					p[2] = d_trianglelist[neighbor * 3 + 1];
					p[3] = d_trianglelist[neighbor * 3 + 2];
										
					if ( p[0] == p[3] && p[1] == p[2] )
					{
						d_neighborlist[neighbor * 3 + 0] = encode_tri(curTri,0);

						if(seg != -1)
						{
							d_encmarker[seg] = -1;
							p[0] = d_trianglelist[3*curTri + 1];
							v[0] = d_pointlist[p[0]];
							p[1] = d_trianglelist[3*curTri + 2];
							v[1] = d_pointlist[p[1]];
							pApex = d_trianglelist[3*curTri];
							v[2] = d_pointlist[pApex];
							pApex = d_trianglelist[3*neighbor + 0];
							v[3] = d_pointlist[pApex];

							if(checkseg4encroach(v[0],v[1],v[2],theta,run_mode))
								d_encmarker[seg] = 0;
							else if(checkseg4encroach(v[1],v[0],v[3],theta,run_mode))
								d_encmarker[seg] = 0;
						}

						break;							
					}	
				}
				
			}			
		}
	}
}

__global__ void kernelMarkSteinerOnsegs
(
	int * d_tri2subseg,
	int * d_enclist,
	int * d_internallist,
	int * d_sinks,
	int numberofactive
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int tindex = d_internallist[pos];
	
	int index = d_sinks[tindex] >> 2;
	int direction = d_sinks[tindex] & 3;

	if(direction != 3)
	{
		int seg = d_tri2subseg[3*index + direction];
		//printf("  %d",seg);

		if(seg != -1) // steiner is on the segment
		{
			d_enclist[seg] = 0; // mark this seg
		}
	}
}

__global__ void kernelUpdateSteinerOnsegs
(
	int * d_encmarker,
	int * d_internallist,
	int numberofactive
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofactive)
		return;

	int sindex = d_internallist[pos];

	d_encmarker[sindex] = 1;
}


__global__ void kernelComputeSplittingPointAndPriority(
	REAL2 * d_pointlist,
	int * d_subseg2tri,
	int * d_trianglelist,
	int * d_neighborlist,
	int	* d_tri2subseg,
	int * d_elementlist,
	int numberofelements,
	int numberofsubsegs,
	REAL2 * d_insertpt,
#ifdef GQM2D_PRIORITY_SIZE
	REAL * d_priorityreal,
#endif
	REAL offConstant,
	bool offCenter
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofelements)
		return;

	int eleidx = d_elementlist[pos];
	if(pos < numberofsubsegs) // subsegments
	{
		int otri, tri[2], triOri[2];
		bool acuteorg = false; // indicate if incident to segments
		bool acutedest = false; // indicate if incident to segments

		// first triangle
		otri = d_subseg2tri[eleidx];
		tri[0] = decode_tri(otri);
		triOri[0] = decode_ori(otri); // first triangle orientation
		if (d_tri2subseg[3 * tri[0] + (triOri[0] + 1) % 3] != -1)
			acutedest = true;
		if (d_tri2subseg[3 * tri[0] + (triOri[0] + 2) % 3] != -1)
			acuteorg = true;

		// second triangle
		otri = d_neighborlist[3 * tri[0] + triOri[0]];
		tri[1] = decode_tri(otri);
		triOri[1] = decode_ori(otri);
		if (tri[1] != -1)
		{
			if (d_tri2subseg[3 * tri[1] + (triOri[1] + 1) % 3] != -1)
				acuteorg = true;
			if (d_tri2subseg[3 * tri[1] + (triOri[1] + 2) % 3] != -1)
				acutedest = true;
		}

		int p[3];
		REAL2 cp[3];
		for (int i = 0; i < 3; i++)
		{
			p[i] = d_trianglelist[tri[0] * 3 + (triOri[0] + i + 1) % 3];
			cp[i] = d_pointlist[p[i]];
		}

		REAL squarelength = (cp[0].x - cp[1].x) * (cp[0].x - cp[1].x) + (cp[0].y - cp[1].y) * (cp[0].y - cp[1].y);
		REAL segmentlength = sqrt(squarelength);

		REAL split = 0.5; // split ratio
		if (acuteorg || acutedest)
		{
			REAL nearestpoweroftwo = 1.0;
			while (segmentlength > 3.0 * nearestpoweroftwo)
			{
				nearestpoweroftwo *= 2.0;
			}
			while (segmentlength < 1.5 * nearestpoweroftwo)
			{
				nearestpoweroftwo *= 0.5;
			}
			split = nearestpoweroftwo / segmentlength;
			if (acutedest)
				split = 1.0 - split;
		}

		REAL2 newpt = 
			MAKE_REAL2(
				cp[0].x + (cp[1].x - cp[0].x) * split,
				cp[0].y + (cp[1].y - cp[0].y) * split);

		REAL multiplier = cuda_fast(cp[0], cp[1], newpt);
#ifndef GQM2D_INEXACT
		if(multiplier == 0.0)
			multiplier = cuda_ccw(cp[0], cp[1], newpt);
#endif
		REAL divisor = squarelength;

		if ((multiplier != 0.0) && (divisor != 0.0))
		{
			multiplier = multiplier / divisor;
			/* Watch out for NANs. */
			if (multiplier == multiplier)
			{
				newpt.x += multiplier * (cp[1].y - cp[0].y);
				newpt.y += multiplier * (cp[0].x - cp[1].x);
			}
		}

		d_insertpt[pos] = newpt;

		
#ifdef GQM2D_PRIORITY_SIZE
		d_priorityreal[pos] = segmentlength;
#endif
	}
	else // triangles
	{

		int p[3] = {
			d_trianglelist[3 * eleidx],
			d_trianglelist[3 * eleidx + 1],
			d_trianglelist[3 * eleidx + 2]
		};

		REAL2 v[3] = {
			d_pointlist[p[0]],
			d_pointlist[p[1]],
			d_pointlist[p[2]]
		};

		// center coordinate
		REAL priority;
		REAL2 cc = computeCircumcenter(v[0], v[1], v[2], offConstant, offCenter, &priority);
		d_insertpt[pos].x = cc.x;
		d_insertpt[pos].y = cc.y;

#ifdef GQM2D_PRIORITY_SIZE
		d_priorityreal[pos] = priority;
#endif

	}
}

__global__ void kernelModifyPriority(
	REAL* d_priorityreal,
	int* d_priority,
	REAL offset0,
	REAL offset1,
	int * d_elementlist,
	int numberofelements,
	int numberofsubsegs
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numberofelements)
		return;

	int eleidx = d_elementlist[pos];
	REAL offset;
	if (pos < numberofsubsegs) // subsegments
	{
		offset = offset0;
	}
	else // triangles
	{
		offset = offset1;
	}

	REAL priority = d_priorityreal[pos] + offset;
	d_priorityreal[pos] = priority;
	d_priority[pos] = __float_as_int((float)priority);
}

__global__ void kernelLocateSplittingPoints(
	REAL2 * d_pointlist,
	int * d_subseg2tri,
	int * d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_insertpt,
#ifdef GQM2D_PRIORITY_SIZE
	uint64 * d_trimarker,
	int * d_priority,
#else
	int * d_trimarker,
#endif
	int * d_sinks,
	int * d_elementlist,
	int numberofelements,
	int numberofsubsegs
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numberofelements)
		return;

	int eleidx = d_elementlist[pos];
	if (eleidx < 0)
		return;

	d_sinks[pos] = -1; // Initialized
#ifdef GQM2D_PRIORITY_SIZE
	uint64 marker = encodeUInt64Priority(d_priority[pos], pos + 1);
	uint64 oldmarker;
	int old;
#else
	int marker = pos;
	int oldmarker;
#endif

	if (pos < numberofsubsegs)
	{
		int otri, tri, triOri;
		otri = d_subseg2tri[eleidx];
		d_sinks[pos] = otri;

		// mark first triangle
		tri = decode_tri(otri);
		triOri = decode_ori(otri);
#ifdef GQM2D_PRIORITY_SIZE
		oldmarker = atomicMax(&d_trimarker[tri], marker);
		if (marker > oldmarker)
		{
			old = getUInt64PriorityIndex(oldmarker) - 1;
			if (old >= 0)
			{
				d_elementlist[old] = -1;
			}
		}
		else if (marker < oldmarker)
		{
			d_elementlist[pos] = -1;
		}
#else 
		oldmarker = atomicMin(&d_trimarker[tri], marker);
		if (marker < oldmarker)
		{
			if (oldmarker != MAXINT)
			{
				d_elementlist[oldmarker] = -1;
			}
		}
		else if (marker > oldmarker)
		{
			d_elementlist[pos] = -1;
		}
#endif
		

		// mark second triangle
		otri = d_neighborlist[3 * tri + triOri];
		tri = decode_tri(otri);
		if (tri != -1)
		{
#ifdef GQM2D_PRIORITY_SIZE
			oldmarker = atomicMax(&d_trimarker[tri], marker);
			if (marker > oldmarker)
			{
				old = getUInt64PriorityIndex(oldmarker) - 1;
				if (old >= 0)
				{
					d_elementlist[old] = -1;
				}
			}
			else if (marker < oldmarker)
			{
				d_elementlist[pos] = -1;
			}
#else 
			oldmarker = atomicMin(&d_trimarker[tri], marker);
			if (marker < oldmarker)
			{
				if (oldmarker != MAXINT)
				{
					d_elementlist[oldmarker] = -1;
				}
			}
			else if (marker > oldmarker)
			{
				d_elementlist[pos] = -1;
			}
#endif
		}
	}
	else
	{
		if (!d_TStatus[eleidx].isBad())
			return;

		REAL2 cc = d_insertpt[pos]; // circumcenter

		int curTri = eleidx; // current triangle index
		int curOri; // current triangle orientation

		int p[3];
		REAL2 v[3];
		REAL2 org;
		REAL det0, det1, det2;

		bool checkNeighbor = false;
		bool onvertex = false;
		bool split = false;
		int direction_dart = 3;

		// Consider following cases:
		// case 0: split current edge
		// case 1: move to neighbor
		// case 2: split current triangle

		// Check the bad triangle itself first
		p[0] = d_trianglelist[3 * curTri];
		p[1] = d_trianglelist[3 * curTri + 1];
		p[2] = d_trianglelist[3 * curTri + 2];

		v[0] = d_pointlist[p[0]];
		v[1] = d_pointlist[p[1]];
		v[2] = d_pointlist[p[2]];

		// three edges
		det0 = cuda_fast(v[0], v[1], cc);
#ifndef GQM2D_INEXACT
		if (det0 == 0) // not sure
			det0 = cuda_ccw(v[0], v[1], cc);
#endif

		det1 = cuda_fast(v[1], v[2], cc);
#ifndef GQM2D_INEXACT
		if (det1 == 0) // not sure
			det1 = cuda_ccw(v[1], v[2], cc);
#endif

		det2 = cuda_fast(v[2], v[0], cc);
#ifndef GQM2D_INEXACT
		if (det2 == 0) // not sure
			det2 = cuda_ccw(v[2], v[0], cc);
#endif

		// case 0: split current edge
		// Points are on the edges (segments) or
		// on the opposite side of segments
		// case 1: move to the neighbor
		if (det0 <= 0)
		{
			if (det0 == 0 || d_tri2subseg[3 * eleidx + 2] != -1)
				split = true;
			direction_dart = 2;
		}
		else if (det1 <= 0)
		{
			if (det1 == 0 || d_tri2subseg[3 * eleidx + 0] != -1)
				split = true;
			direction_dart = 0;
		}
		else if (det2 <= 0)
		{
			if (det2 == 0 || d_tri2subseg[3 * eleidx + 1] != -1)
				split = true;
			direction_dart = 1;
		}

		if (!split && direction_dart != 3) // need to check neighbors of bad triangles
		{
			int neighbor = d_neighborlist[3 * curTri + direction_dart];
			if (neighbor < 0) // circumcenter is out of boundary
			{
				checkNeighbor = false; // mark current triangle directly
				split = true;
			}
			else
			{
				org.x = v[direction_dart].x; // set up starting point
				org.y = v[direction_dart].y;
				curTri = decode_tri(neighbor); // move to the neighbor
				curOri = decode_ori(neighbor);
				checkNeighbor = true;
			}
		}

		REAL detLeft, detRight;
		bool moveleft;

		while (checkNeighbor) // trave from bad triangle to circumcenter
		{
			p[0] = d_trianglelist[3 * curTri];
			p[1] = d_trianglelist[3 * curTri + 1];
			p[2] = d_trianglelist[3 * curTri + 2];

			v[0] = d_pointlist[p[0]];
			v[1] = d_pointlist[p[1]];
			v[2] = d_pointlist[p[2]];

			direction_dart = 3; // reset to default value

			int pOrg, pDest, pApex;
			pOrg = (curOri + 1) % 3;
			pDest = (curOri + 2) % 3;
			pApex = curOri;

			// left side
			detLeft = cuda_fast(v[pDest], v[pApex], cc);
#ifndef GQM2D_INEXACT
			if (detLeft == 0)
				detLeft = cuda_ccw(v[pDest], v[pApex], cc);
#endif

			// right side
			detRight = cuda_fast(v[pApex], v[pOrg], cc);
#ifndef GQM2D_INEXACT
			if (detRight == 0)
				detRight = cuda_ccw(v[pApex], v[pOrg], cc);
#endif

			if (detLeft < 0.0)
			{
				if (detRight < 0.0)
				{
					// at corner region, check inner product
					moveleft = (v[(curOri + 2) % 3].x - v[(curOri + 1) % 3].x)*(cc.x - v[curOri].x) +
						(v[(curOri + 2) % 3].y - v[(curOri + 1) % 3].y)*(cc.y - v[curOri].y) > 0.0;
				}
				else
				{
					moveleft = true;
				}
			}
			else
			{
				if (detRight < 0.0)
				{
					moveleft = false;
				}
				else
				{
					if (detLeft == 0.0)
					{
						split = true;
						direction_dart = (curOri + 1) % 3;
						break;
					}
					if (detRight == 0.0)
					{
						split = true;
						direction_dart = (curOri + 2) % 3;
						break;
					}
					break; // in triangle
				}
			}

			if (moveleft)
			{
				direction_dart = (curOri + 1) % 3;
			}
			else
			{
				direction_dart = (curOri + 2) % 3;
			}

			int neighbor = d_neighborlist[3 * curTri + direction_dart];

			// circumcenter is out of boundary, mark current trianlge as segement split
			// or circumcenter is on the opposite side of segment
			if (neighbor < 0 || d_tri2subseg[3 * curTri + direction_dart] != -1)
			{
				split = true;
				break;
			}

			curTri = decode_tri(neighbor); // move to neighbor
			curOri = decode_ori(neighbor); // this edge must intersect segment 
		}

#ifdef GQM2D_PRIORITY_SIZE
		oldmarker = atomicMax(&d_trimarker[curTri], marker);
		if (marker > oldmarker)
		{
			old = getUInt64PriorityIndex(oldmarker) - 1;
			if (old >= 0)
			{
				d_elementlist[old] = -1;
			}
		}
		else if (marker < oldmarker)
		{
			d_elementlist[pos] = -1;
		}
#else 
		oldmarker = atomicMin(&d_trimarker[curTri], marker);
		if (marker < oldmarker)
		{
			if (oldmarker != MAXINT)
			{
				d_elementlist[oldmarker] = -1;
			}
		}
		else if (marker > oldmarker)
		{
			d_elementlist[pos] = -1;
		}
#endif

		if (split) // also mark the incident triangle
		{
			int neighbor = d_neighborlist[3 * curTri + direction_dart];
			if (neighbor != -1) // prevent that steiner point happen to be on or out of the boundary
			{
				int neighborIndex = decode_tri(neighbor);
				int neighborOri = decode_ori(neighbor);

#ifdef GQM2D_PRIORITY_SIZE
				oldmarker = atomicMax(&d_trimarker[neighborIndex], marker);
				if (marker > oldmarker)
				{
					old = getUInt64PriorityIndex(oldmarker) - 1;
					if (old >= 0)
					{
						d_elementlist[old] = -1;
					}
				}
				else if (marker < oldmarker)
				{
					d_elementlist[pos] = -1;
				}
#else 
				oldmarker = atomicMin(&d_trimarker[neighborIndex], marker);
				if (marker < oldmarker)
				{
					if (oldmarker != MAXINT)
					{
						d_elementlist[oldmarker] = -1;
					}
				}
				else if (marker > oldmarker)
				{
					d_elementlist[pos] = -1;
				}
#endif
			}
		}

		d_sinks[pos] = (curTri << 2) | direction_dart;
	}
}

__global__ void kernelFastCavityCheck(
	REAL2 * d_pointlist,
	int * d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_insertpt,
#ifdef GQM2D_PRIORITY_SIZE
	uint64 * d_trimarker,
	int * d_priority,
#else
	int * d_trimarker,
#endif
	int * d_sinks,
	int * d_elementlist,
	int numberofelements
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= numberofelements)
		return;

	int eleidx = d_elementlist[pos];
	if (eleidx < 0) // lost in locating competition
		return;

	int sink = d_sinks[pos];
	int tri = sink >> 2;
	int direction = sink & 3;

	REAL2 cc = d_insertpt[pos]; // splitting points
#ifdef GQM2D_PRIORITY_SIZE
	uint64 marker = encodeUInt64Priority(d_priority[pos], pos + 1);
	uint64 oldmarker;
	int old;
#else
	int marker = pos;
	int oldmarker;
#endif

	int stack_loop[2 * CAVITY_FAST_SIZE];
	int stack_loop_checked = 0;
	int stack_loop_cur = 0, stack_loop_tail = 0, stack_loop_size = 0;

	int neightri, neightriIdx, neightriOri;
	int neighIndex;

	// initialize loop stack
	int i;
	if (direction == 3) // in triangle
	{
		for (i = 0; i < 3; i++)
		{
			neighIndex = 3 * tri + i;
			if (d_tri2subseg[neighIndex] != -1) // skip this side
				continue;
			neightri = d_neighborlist[neighIndex];
			if (neightri != -1)
			{
				stack_loop[stack_loop_tail] = neightri;
				stack_loop_tail++;
				stack_loop_size++;
			}
		}
	}
	else // on edge or subseg
	{
		// check first triangle
		for (i = 0; i < 3; i++)
		{
			neighIndex = 3 * tri + i;
			if (i == direction || d_tri2subseg[neighIndex] != -1)
				continue;
			neightri = d_neighborlist[neighIndex];
			if (neightri != -1)
			{
				stack_loop[stack_loop_tail] = neightri;
				stack_loop_tail++;
				stack_loop_size++;
			}
		}
		// check second triangle
		neightri = d_neighborlist[3 * tri + direction];
		if (neightri != -1)
		{
			neightriIdx = decode_tri(neightri);
			neightriOri = decode_ori(neightri);
			for (i = 0; i < 3; i++)
			{
				neighIndex = 3 * neightriIdx + i;
				if (i == neightriOri || d_tri2subseg[neighIndex] != -1)
					continue;
				neightri = d_neighborlist[neighIndex];
				if (neightri != -1)
				{
					stack_loop[stack_loop_tail] = neightri;
					stack_loop_tail++;
					stack_loop_size++;
				}
			}
		}
	}

	if (stack_loop_size == 0) // nothing to check
		return;

	// fast expanding check
	int curtri, curtriIdx, curtriOri;
	int p[3];
	REAL2 v[3];
	int subseg;

	while (true)
	{
		if (d_elementlist[pos] < 0) // defeated by others
			return;

		curtri = stack_loop[stack_loop_cur]; // this should not be -1
		curtriIdx = decode_tri(curtri);
		for (i = 0; i < 3; i++)
		{
			p[i] = d_trianglelist[3 * curtriIdx + i];
			v[i] = d_pointlist[p[i]];
		}

		// in circle test
		REAL r = cuda_inCircle(v[0], v[1], v[2], cc);
#ifndef GQM2D_INEXACT
		if (r == 0) // not sure
			r = cuda_inCircle_exact(v[0], v[1], v[2], cc);
#endif

		if (r > 0) // inside circle
		{
#ifdef GQM2D_PRIORITY_SIZE
			oldmarker = atomicMax(&d_trimarker[curtriIdx], marker);
			if (marker > oldmarker)
			{
				// mark the other as loser
				old = getUInt64PriorityIndex(oldmarker) - 1;
				if (old >= 0)
				{
					d_elementlist[old] = -1;
				}
#else
			oldmarker = atomicMin(&d_trimarker[curtriIdx], marker);
			if(marker < oldmarker)
			{
				if (oldmarker != MAXINT)
				{
					d_elementlist[oldmarker] = -1;
				}
#endif

				// update the stack
				stack_loop_checked++;
				if (stack_loop_checked >= CAVITY_FAST_SIZE) // enough check
					return;

				curtriOri = decode_ori(curtri);
				neighIndex = 3 * curtriIdx + (curtriOri + 1) % 3;
				subseg = d_tri2subseg[neighIndex];
				neightri = d_neighborlist[neighIndex];
				if (subseg == -1 && neightri != -1) // append
				{
					stack_loop_size++;
					if (stack_loop_size > 2 * CAVITY_FAST_SIZE)
						return;
					stack_loop[stack_loop_tail] = neightri;
					stack_loop_tail = (stack_loop_tail + 1) % (2 * CAVITY_FAST_SIZE);
				}

				neighIndex = 3 * curtriIdx + (curtriOri + 2) % 3;
				subseg = d_tri2subseg[neighIndex];
				neightri = d_neighborlist[neighIndex];
				if (subseg == -1 && neightri != -1) // append
				{
					stack_loop_size++;
					if (stack_loop_size > 2 * CAVITY_FAST_SIZE)
						return;
					stack_loop[stack_loop_tail] = neightri;
					stack_loop_tail = (stack_loop_tail + 1) % (2 * CAVITY_FAST_SIZE);
				}

				stack_loop_cur = (stack_loop_cur + 1) % (2 * CAVITY_FAST_SIZE);
				stack_loop_size--;
				if (stack_loop_size == 0)
					return;

				continue;
			}
#ifdef GQM2D_PRIORITY_SIZE
			else if (marker < oldmarker) // lost
#else
			else if (marker > oldmarker) // lost
#endif
			{
				d_elementlist[pos] = -1;
				return;
			}
			else
			{
				// marked already, do nothing
			}
		}

		// no check happened, update stack
		stack_loop_size--;
		if (stack_loop_size == 0)
			return;
		stack_loop_cur = (stack_loop_cur + 1) % (2 * CAVITY_FAST_SIZE);
	}
}

__global__ void kernelMarkOnSegs
(
	int *d_trianglelist,
	int *d_neighborlist,
	int * d_tri2subseg,
	int* d_segmarker,
	int	*d_sinks,
	int * d_elementlist,
	int numberofelements,
	int numberofsubsegs
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= numberofelements)
		return;

	int eleidx = d_elementlist[pos];
	if(eleidx < 0)
		return;

	int sink = d_sinks[pos];
	int tri = sink >> 2;

	int direction = sink & 3;
	
	if (direction != 3)
	{
		if (pos >= numberofsubsegs) // triangle splitting
		{
			int seg = d_tri2subseg[3 * tri + direction];
			if (seg != -1)
			{
				// splitting point of triangle on subseg, lost
				d_elementlist[pos] = -1;
				// recorded, will be marked as encroached later
				d_segmarker[seg] = 1;
				return;
			}
		}
	}

}

__global__ void kernelResetNeighborMarker(
	int * d_neighborlist,
	TStatus * d_TStatus,
	int * d_sinks,
	int * d_threadlist,
	int numberofsteiners
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numberofsteiners)
		return;

	int tindex = d_threadlist[pos];

	int index = d_sinks[tindex] >> 2;
	int direction = d_sinks[tindex] & 3;

	int otri, n[3];

	// reset the first triangle and its neighbors
	d_TStatus[index].setFlipOri(15); // indicate no change
	n[0] = decode_tri(d_neighborlist[3 * index]);
	n[1] = decode_tri(d_neighborlist[3 * index + 1]);
	n[2] = decode_tri(d_neighborlist[3 * index + 2]);
	for (int i = 0; i<3; i++)
	{
		if (n[i] != -1)
			d_TStatus[n[i]].setFlipOri(15);
	}

	// reset the triangle on the other side
	if (direction != 3)
	{
		otri = d_neighborlist[3 * index + direction];
		if (otri != -1)
		{
			index = decode_tri(otri);
			d_TStatus[index].setFlipOri(15); // indicate no change
			n[0] = decode_tri(d_neighborlist[3 * index]);
			n[1] = decode_tri(d_neighborlist[3 * index + 1]);
			n[2] = decode_tri(d_neighborlist[3 * index + 2]);
			for (int i = 0; i<3; i++)
			{
				if (n[i] != -1)
					d_TStatus[n[i]].setFlipOri(15);
			}
		}
	}
}

__global__ void kernelInsertSplittingPoints(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
#ifdef GQM2D_PRIORITY_SIZE
	REAL* d_pointpriority,
	REAL* d_priorityreal,
#endif
	int * d_subseg2tri,
	int * d_subseg2seg,
	int * d_encmarker,
	int * d_segmarker,
	int	* d_trianglelist,
	int * d_neighborlist,
	int	* d_tri2subseg,
	TStatus * d_TStatus,
	REAL2 * d_insertpt,
	int	* d_sinks,
	int * d_emptypoints,
	int * d_emptytriangles,
	int * d_elementlist,
	int * d_threadlist,
	int emptypointsLength,
	int emptytrianglesLength,
	int numberofemptypoints,
	int numberofemptytriangles,
	int numberofsubseg,
	int numberofwonsegs,
	int numberofsteiners,
	int encmode,
	REAL theta
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numberofsteiners)
		return;

	int tindex = d_threadlist[pos];
	int eleidx = d_elementlist[tindex];

	int sink = d_sinks[tindex];
	int index = sink >> 2;
	int direction = sink & 3;

	int tri[4], p[3], neighbor[3], seg[3];
	REAL2 v[4];

	// configure new point and triangles indices, status and markers
	int newID_point, newID_tri1, newID_tri2, newID_seg;
	newID_point = d_emptypoints[emptypointsLength - numberofemptypoints + pos];
	d_pointlist[newID_point] = d_insertpt[tindex];
#ifdef GQM2D_PRIORITY_SIZE
	d_pointpriority[newID_point] = d_priorityreal[tindex];
#endif

	newID_tri1 = d_emptytriangles[emptytrianglesLength - numberofemptytriangles + 2 * pos];
	newID_tri2 = d_emptytriangles[emptytrianglesLength - numberofemptytriangles + 2 * pos + 1];
	tri[0] = index;

	if (direction == 3)
	{
		d_PStatus[newID_point].createNewTriangleSplit();

		// two new
		tri[1] = newID_tri1;
		tri[2] = newID_tri2;

		d_TStatus[tri[1]].clear();
		d_TStatus[tri[2]].clear();

		d_TStatus[tri[0]].setCheck(true); // first triangle
		d_TStatus[tri[0]].setFlip(true);  // mark for flip-flop
		d_TStatus[tri[0]].setFlipOri(0);  // mark for change

		d_TStatus[tri[1]].setNull(false); // first new triangle
		d_TStatus[tri[1]].setCheck(true);
		d_TStatus[tri[1]].setFlip(true);
		d_TStatus[tri[1]].setFlipOri(0);

		d_TStatus[tri[2]].setNull(false); // second new triangle
		d_TStatus[tri[2]].setCheck(true);
		d_TStatus[tri[2]].setFlip(true);
		d_TStatus[tri[2]].setFlipOri(0);

		// insert steiner point
		for (int i = 0; i < 3; i++)
			p[i] = d_trianglelist[tri[0] * 3 + (i + 1) % 3];

		// update triangles and subsegs relation
		for (int i = 0; i<3; i++)
			seg[i] = d_tri2subseg[3 * tri[0] + i];
		d_tri2subseg[3 * tri[0]] = seg[0];   // old segment
		d_tri2subseg[3 * tri[0] + 1] = -1;	   // new edge
		d_tri2subseg[3 * tri[0] + 2] = -1;     // new edge
		d_tri2subseg[3 * tri[1]] = seg[1];   // old segment
		d_tri2subseg[3 * tri[1] + 1] = -1;	   // new edge
		d_tri2subseg[3 * tri[1] + 2] = -1;     // new edge
		d_tri2subseg[3 * tri[2]] = seg[2];   // old segment
		d_tri2subseg[3 * tri[2] + 1] = -1;	   // new edge
		d_tri2subseg[3 * tri[2] + 2] = -1;     // new edge

		if (seg[0] != -1)
			d_subseg2tri[seg[0]] = tri[0] << 2 | 0;
		if (seg[1] != -1)
			d_subseg2tri[seg[1]] = tri[1] << 2 | 0;
		if (seg[2] != -1)
			d_subseg2tri[seg[2]] = tri[2] << 2 | 0;

		// update and insert triangles
		for (int i = 0; i<3; i++)
			neighbor[i] = d_neighborlist[tri[0] * 3 + i];
		insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[0], p[1],
			tri[0], neighbor[0], tri[1] << 2 | 2, tri[2] << 2 | 1);
		insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[1], p[2],
			tri[1], neighbor[1], tri[2] << 2 | 2, tri[0] << 2 | 1);
		insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[2], p[0],
			tri[2], neighbor[2], tri[0] << 2 | 2, tri[1] << 2 | 1);

	}
	else // splitting is on the edge or subseg
	{
		bool segsplit = pos < numberofwonsegs;

		int triOri[2], otri;
		triOri[0] = direction;
		otri = d_neighborlist[3 * tri[0] + triOri[0]];
		triOri[1] = decode_ori(otri);

		// two new and one old
		tri[1] = newID_tri1;
		tri[2] = decode_tri(otri);
		tri[3] = newID_tri2;

		d_TStatus[tri[1]].clear(); // reset information
		d_TStatus[tri[3]].clear();

		if (segsplit)
		{
			d_PStatus[newID_point].createNewSegmentSplit();

			newID_seg = numberofsubseg + pos;
			d_encmarker[eleidx] = -1;
			d_encmarker[newID_seg] = -1;
			d_segmarker[eleidx] = -1;
			d_segmarker[newID_seg] = -1;
		}
		else
		{
			d_PStatus[newID_point].createNewTriangleSplit();
		}

		// first old and new
		d_TStatus[tri[0]].setCheck(true); // first triangle
		d_TStatus[tri[0]].setFlip(true);  // mark for flip-flop
		d_TStatus[tri[0]].setFlipOri(0);  // mark for change

		d_TStatus[tri[1]].setNull(false); // first new triangle
		d_TStatus[tri[1]].setCheck(true);
		d_TStatus[tri[1]].setFlip(true);
		d_TStatus[tri[1]].setFlipOri(0);

		// insert steiner point on edge
		for (int i = 0; i < 3; i++)
			p[i] = d_trianglelist[tri[0] * 3 + (triOri[0] + i + 1) % 3];

		// update triangles and subsegs relation
		for (int i = 0; i<3; i++)
			seg[i] = d_tri2subseg[3 * tri[0] + (triOri[0] + i) % 3];
		d_tri2subseg[3 * tri[0]] = seg[2];   // old segment
		d_tri2subseg[3 * tri[0] + 1] = seg[0]; // old segment | new edge (-1)
		d_tri2subseg[3 * tri[0] + 2] = -1;     // new edge
		d_tri2subseg[3 * tri[1]] = seg[1];   // old segment
		d_tri2subseg[3 * tri[1] + 1] = -1;	   // new edge
		d_tri2subseg[3 * tri[1] + 2] = segsplit ? newID_seg : seg[0]; // old segment | new edge (-1)

		if (segsplit)
		{
			d_subseg2tri[seg[0]] = (tri[0] << 2) | 1;
			d_subseg2tri[newID_seg] = (tri[1] << 2) | 2;
			d_subseg2seg[newID_seg] = d_subseg2seg[seg[0]]; // copy my parent to you
		}
		if (seg[1] != -1)
			d_subseg2tri[seg[1]] = tri[1] << 2 | 0;
		if (seg[2] != -1)
			d_subseg2tri[seg[2]] = tri[0] << 2 | 0;

		// update encroched marker
		if (segsplit)
		{
			v[0] = d_pointlist[p[0]];
			v[1] = d_pointlist[p[1]];
			v[2] = d_pointlist[p[2]];
			v[3] = d_pointlist[newID_point];
			if (checkseg4encroach(v[0], v[3], v[2], theta, encmode))
			{
				d_encmarker[eleidx] = 0;
				d_segmarker[eleidx] = 1;
			}

			if (checkseg4encroach(v[3], v[1], v[2], theta, encmode))
			{
				d_encmarker[newID_seg] = 0;
				d_segmarker[newID_seg] = 1;
			}
		}

		// update and insert triangles
		neighbor[0] = d_neighborlist[tri[0] * 3 + triOri[0]];
		neighbor[1] = d_neighborlist[tri[0] * 3 + (triOri[0] + 1) % 3];
		neighbor[2] = d_neighborlist[tri[0] * 3 + (triOri[0] + 2) % 3];
		if (tri[2] == -1)
		{
			insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[2], p[0],
				tri[0], neighbor[2], -1, tri[1] << 2 | 1);
			insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[1], p[2],
				tri[1], neighbor[1], tri[0] << 2 | 2, -1);
		}
		else
		{
			insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[2], p[0],
				tri[0], neighbor[2], tri[2] << 2 | 2, tri[1] << 2 | 1);
			insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[1], p[2],
				tri[1], neighbor[1], tri[0] << 2 | 2, tri[3] << 2 | 1);
		}

		// second old and new
		if (tri[2] != -1)
		{
			d_TStatus[tri[2]].setCheck(true); // second triangle
			d_TStatus[tri[2]].setFlip(true);  // mark for flip-flop
			d_TStatus[tri[2]].setFlipOri(0);  // mark for change

			d_TStatus[tri[3]].setNull(false); // second new triangle
			d_TStatus[tri[3]].setCheck(true);
			d_TStatus[tri[3]].setFlip(true);
			d_TStatus[tri[3]].setFlipOri(0);

			for (int i = 0; i < 3; i++)
				p[i] = d_trianglelist[tri[2] * 3 + (triOri[1] + i + 1) % 3];

			// update triangles and subsegs relation
			for (int i = 0; i<3; i++)
				seg[i] = d_tri2subseg[3 * tri[2] + (triOri[1] + i) % 3];
			d_tri2subseg[3 * tri[2]] = seg[1];   // old segment
			d_tri2subseg[3 * tri[2] + 1] = -1;     // new edge
			d_tri2subseg[3 * tri[2] + 2] = seg[0]; // old segment | new edge (-1)
			d_tri2subseg[3 * tri[3]] = seg[2];   // old segment
			d_tri2subseg[3 * tri[3] + 1] = segsplit? newID_seg : seg[0]; // old segment | new edge (-1)
			d_tri2subseg[3 * tri[3] + 2] = -1;

			if (seg[1] != -1)
				d_subseg2tri[seg[1]] = tri[2] << 2 | 0;
			if (seg[2] != -1)
				d_subseg2tri[seg[2]] = tri[3] << 2 | 0;

			// update encroched marker
			if (segsplit)
			{
				for (int i = 0; i < 3; i++)
					p[i] = d_trianglelist[tri[2] * 3 + (triOri[1] + i + 1) % 3];
				v[0] = d_pointlist[p[0]];
				v[1] = d_pointlist[p[1]];
				v[2] = d_pointlist[p[2]];
				v[3] = d_pointlist[newID_point];

				if (checkseg4encroach(v[0], v[3], v[2], theta, encmode))
				{
					d_encmarker[newID_seg] = 0;
					d_segmarker[newID_seg] = 1;
				}

				if (checkseg4encroach(v[3], v[1], v[2], theta, encmode))
				{
					d_encmarker[eleidx] = 0;
					d_segmarker[eleidx] = 1;
				}
			}

			// update and insert triangles
			neighbor[0] = d_neighborlist[tri[2] * 3 + triOri[1]];
			neighbor[1] = d_neighborlist[tri[2] * 3 + (triOri[1] + 1) % 3];
			neighbor[2] = d_neighborlist[tri[2] * 3 + (triOri[1] + 2) % 3];
			insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[1], p[2],
				tri[2], neighbor[1], tri[3] << 2 | 2, tri[0] << 2 | 1);
			insertTriangle(d_trianglelist, d_neighborlist, newID_point, p[2], p[0],
				tri[3], neighbor[2], tri[1] << 2 | 2, tri[2] << 2 | 1);
		}
	}
}

__global__ void kernelUpdateNeighbors
(
	REAL2 * d_pointlist,
	PStatus * d_PStatus,
	int * d_subseg2tri,
	int * d_encmarker,
	int * d_segmarker,
	int	* d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	TStatus * d_TStatus,
	int * d_sinks,
	int * d_emptypoints,
	int * d_threadlist,
	int emptypointsLength,
	int numberofemptypoints,
	int numberofwonsegs,
	int numberofsteiners,
	int encmode,
	REAL theta
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numberofsteiners)
		return;

	bool segsplit = pos < numberofwonsegs;

	int tindex = d_threadlist[pos];

	int index = d_sinks[tindex] >> 2;
	int direction = d_sinks[tindex] & 3;

	// find 4 triangles
	int otri, tri[4], p[4], iteration, pApex;
	REAL2 v[4];

	if (direction == 3)
	{
		iteration = 3;

		tri[0] = index;
		otri = d_neighborlist[3 * tri[0] + 1];
		tri[1] = decode_tri(otri);
		otri = d_neighborlist[3 * tri[0] + 2];
		tri[2] = decode_tri(otri);
	}
	else
	{
		tri[0] = index;
		otri = d_neighborlist[3 * tri[0] + 2];
		tri[1] = decode_tri(otri);

		otri = d_neighborlist[3 * tri[0] + 1];
		if (otri == -1)
			iteration = 2;
		else
		{
			iteration = 4;
			tri[2] = decode_tri(otri);
			otri = d_neighborlist[3 * tri[1] + 2];
			tri[3] = decode_tri(otri);
		}
	}

	bool encroaching = false;

	for (int i = 0; i < iteration; i++)//for each pTri[i]
	{
		int curTri = tri[i];
		int seg = d_tri2subseg[3 * curTri + 0];
		int pOppTri = d_neighborlist[curTri * 3 + 0];
		int pOpp = decode_tri(pOppTri);
		int pOri = decode_ori(pOppTri);

		if (pOppTri < 0) // dont need to update neighbor, but need to update encroach marker
		{
			if (seg != -1)
			{
				d_encmarker[seg] = -1;
				p[0] = d_trianglelist[3 * curTri + 1];
				v[0] = d_pointlist[p[0]];
				p[1] = d_trianglelist[3 * curTri + 2];
				v[1] = d_pointlist[p[1]];
				pApex = d_trianglelist[3 * curTri];
				v[2] = d_pointlist[pApex];

				if (checkseg4encroach(v[0], v[1], v[2], theta, encmode))
				{
					d_encmarker[seg] = 0;
					d_segmarker[seg] = 1;
					encroaching = true;
				}
			}
			continue;
		}

		if (d_TStatus[pOpp].getFlipOri() == 15)// neighbor[i] doesn't change
		{
			d_neighborlist[pOpp * 3 + pOri] = encode_tri(curTri, 0);
			if (seg != -1)
			{
				d_encmarker[seg] = -1;
				p[0] = d_trianglelist[3 * curTri + 1];
				v[0] = d_pointlist[p[0]];
				p[1] = d_trianglelist[3 * curTri + 2];
				v[1] = d_pointlist[p[1]];
				pApex = d_trianglelist[3 * curTri];
				v[2] = d_pointlist[pApex];
				pApex = d_trianglelist[3 * pOpp + pOri];
				v[3] = d_pointlist[pApex];

				if (checkseg4encroach(v[0], v[1], v[2], theta, encmode))
				{
					d_encmarker[seg] = 0;
					d_segmarker[seg] = 1;
					encroaching = true;
				}
				else if (checkseg4encroach(v[1], v[0], v[3], theta, encmode))
				{
					d_encmarker[seg] = 0;
					d_segmarker[seg] = 1;
					encroaching = true;
				}
			}
		}
		else
		{
			int neighbor;

			p[0] = d_trianglelist[curTri * 3 + 1];
			p[1] = d_trianglelist[curTri * 3 + 2];

			for (int j = 0; j < 3; j++)
			{
				if (j == 0)
					neighbor = pOpp;
				else
					neighbor = decode_tri(d_neighborlist[pOpp * 3 + j]);

				if (neighbor > -1)
				{
					p[2] = d_trianglelist[neighbor * 3 + 1];
					p[3] = d_trianglelist[neighbor * 3 + 2];

					if (p[0] == p[3] && p[1] == p[2])
					{
						d_neighborlist[neighbor * 3 + 0] = encode_tri(curTri, 0);

						if (seg != -1)
						{
							d_encmarker[seg] = -1;
							p[0] = d_trianglelist[3 * curTri + 1];
							v[0] = d_pointlist[p[0]];
							p[1] = d_trianglelist[3 * curTri + 2];
							v[1] = d_pointlist[p[1]];
							pApex = d_trianglelist[3 * curTri];
							v[2] = d_pointlist[pApex];
							pApex = d_trianglelist[3 * neighbor + 0];
							v[3] = d_pointlist[pApex];

							if (checkseg4encroach(v[0], v[1], v[2], theta, encmode))
							{
								d_encmarker[seg] = 0;
								d_segmarker[seg] = 1;
								encroaching = true;
							}
							else if (checkseg4encroach(v[1], v[0], v[3], theta, encmode))
							{
								d_encmarker[seg] = 0;
								d_segmarker[seg] = 1;
								encroaching = true;
							}
						}

						break;
					}
				}

			}
		}
	}

	if (encroaching && !segsplit)
	{
		// all incident triangles of the new point have already been marked
		// so no need to call markFan
		int newID_point = d_emptypoints[emptypointsLength - numberofemptypoints + pos];
		d_PStatus[newID_point].setRedundant(); // mark as reduntant to remove in flipflop
	}
}

__global__ void markBadSubsegsAsEncroached
(
	int * d_encmarker,
	int * d_segmarker,
	int numberofsubsegs
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numberofsubsegs)
		return;

	if(d_segmarker[pos] >= 0)
		d_encmarker[pos] = 1;
}

/**																					**/
/**																					**/
/********* Mesh quality maintenance end here								 *********/

/********* Initialization routine begin here								 *********/
/**																					**/
/**																					**/

__global__ void kernelUpdateNeighborsFormat2Otri(
	int * d_trianglelist,
	int * d_neighborlist,
	int numberoftriangles
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberoftriangles)
		return;

	int neighbors[3] = {
		d_neighborlist[3 * pos],
		d_neighborlist[3 * pos + 1],
		d_neighborlist[3 * pos + 2]
	};

	int neighborsOri[3] = {
		findIncidentOri(d_trianglelist,pos,neighbors[0]),
		findIncidentOri(d_trianglelist,pos,neighbors[1]),
		findIncidentOri(d_trianglelist,pos,neighbors[2])
	};

	for (int i = 0; i < 3; i++)
		d_neighborlist[3 * pos + i] = encode_tri(neighbors[i], neighborsOri[i]);
}

void updateNeighborsFormat2Otri
(
	IntD	&t_trianglelist,
	IntD	&t_neighborlist,
	int numberoftriangles
 )
{
	int numberofblocks = (ceil)((float)(numberoftriangles) / BLOCK_SIZE);

	kernelUpdateNeighborsFormat2Otri<<<numberofblocks, BLOCK_SIZE>>>(
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		numberoftriangles);
}

__global__ void kernelInitSubsegs(
	int * d_segmentlist,
	int * d_subseg2tri,
	int * d_trianglelist,
	int * d_neighborlist,
	int * d_tri2subseg,
	int numberofsegments
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= numberofsegments)
		return;

	int sOrg, sDest;
	sOrg = d_segmentlist[2 * pos + 0];
	sDest = d_segmentlist[2 * pos + 1];
	int p[3];

	int otri, triIndex, triOri;

	// First triangle
	triIndex = d_subseg2tri[pos];
	p[0] = d_trianglelist[3 * triIndex + 0];
	p[1] = d_trianglelist[3 * triIndex + 1];
	p[2] = d_trianglelist[3 * triIndex + 2];

	if ((p[0] == sOrg && p[1] == sDest) || (p[0] == sDest && p[1] == sOrg)) // ori 2
	{
		triOri = 2;
	}
	else if ((p[1] == sOrg && p[2] == sDest) || (p[1] == sDest && p[2] == sOrg)) // ori 0
	{
		triOri = 0;
	}
	else if ((p[2] == sOrg && p[0] == sDest) || (p[2] == sDest && p[0] == sOrg)) // ori 1
	{
		triOri = 1;
	}
	otri = (triIndex << 2) | triOri;
	d_subseg2tri[pos] = otri;
	d_tri2subseg[3 * triIndex + triOri] = pos;
	
	// Second triangle
	otri = d_neighborlist[3 * triIndex + triOri];
	if (otri != -1)
	{
		triIndex = decode_tri(otri);
		triOri = decode_ori(otri);
		d_tri2subseg[3 * triIndex + triOri] = pos;
	}
}

__global__ void kernelInitSubsegs(
 int * d_segmentlist,
 int * d_subseg2tri,
 int * d_trianglelist,
 int * d_tri2subseg,
 int numberoftriangles,
 int numberofsegments,
 int scanoffset,
 int numofscan,
 int trioffset,
 int numoftri
)
{
	__shared__ int s[2*BLOCK_SIZE];

	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	int threadId = threadIdx.x;

	int p[3];

	if(pos < numoftri)
	{
		p[0] = d_trianglelist[3*(pos+trioffset)];
		p[1] = d_trianglelist[3*(pos+trioffset)+1];
		p[2] = d_trianglelist[3*(pos+trioffset)+2];
	}

	for(int i=scanoffset; i<scanoffset+numofscan; i++)
	{
		if( threadId+i*BLOCK_SIZE < numberofsegments)
		{
			s[2*threadId] = d_segmentlist[2*(threadId+i*BLOCK_SIZE)];
			s[2*threadId+1] = d_segmentlist[2*(threadId+i*BLOCK_SIZE)+1];
		}
		__syncthreads(); // wait for all threads in the same block finish reading

		if(pos < numoftri)
		{
			for(int j = 0; j< BLOCK_SIZE; j++)
			{
				int sindex = j+i*BLOCK_SIZE; // subsegment index
				if( sindex < numberofsegments)
				{
					int index0 = s[2*j];
					int index1 = s[2*j+1];

					if( (p[0] == index0 && p[1] == index1) || (p[0] == index1 && p[1] == index0) ) // ori 2
					{
						int otri = (pos << 2) | 2;
						atomicMax(d_subseg2tri + sindex, otri ); // larger index
						d_tri2subseg[3*pos + 2] = sindex;
					}
					else if ((p[1] == index0 && p[2] == index1) || (p[1] == index1 && p[2] == index0)) // ori 0
					{
						int otri = (pos << 2) | 0;
						atomicMax(d_subseg2tri + sindex, otri );
						d_tri2subseg[3*pos + 0] = sindex;
					}
					else if ((p[2] == index0 && p[0] == index1) || (p[2] == index1 && p[0] == index0) ) // ori 1
					{
						int otri = (pos << 2) | 1;
						atomicMax(d_subseg2tri + sindex, otri );
						d_tri2subseg[3*pos + 1] = sindex;
					}
				}
			}
		}
		__syncthreads(); // wait for all threads in the same block finish writing
	}

}

void initSubsegs
(
	IntD	&t_segmentlist,
	IntD	&t_subseg2tri,
	IntD    &t_trianglelist,
	IntD    &t_neighborlist,
	IntD    &t_tri2subseg,
	int numberofsegments
)
{
	int numberofblocks = (ceil)((float)numberofsegments / BLOCK_SIZE);
	kernelInitSubsegs<<<numberofblocks, BLOCK_SIZE>>>(
		thrust::raw_pointer_cast(&t_segmentlist[0]),
		thrust::raw_pointer_cast(&t_subseg2tri[0]),
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tri2subseg[0]),
		numberofsegments);
}


/**																					**/
/**																					**/
/********* Initialization routine end here									 *********/

/********* Compact routine begin here										 *********/
/**																					**/
/**																					**/

__global__ void kernelMarkValidTriangles_compact
(
	TStatus *d_TStatus,
	int		*d_valid, 
	int		numberofprocess,
	int		poffset
 )
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if ( pos >= numberofprocess )
		return ; 

	int triIndex = pos + poffset;

	d_valid[triIndex] = d_TStatus[triIndex].isNull() ? 0 : 1;	

	if ( !d_TStatus[triIndex].isNull() )
	{
		//d_TStatus[pos].setCheck(false);
	}
}

__global__ void kernelCollectTrianglesEmptySlots_compact
(
	TStatus	*d_TStatus, 
	int	*d_prefix, 
	int	*d_empty, 
	int		numberofprocess,
	int		poffset
 )
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofprocess || !d_TStatus[pos+poffset].isNull() ) 
        return ; 

    int id = pos + poffset - d_prefix[pos+poffset];

    d_empty[id] = pos + poffset; 
}

__global__ void kernelFillTrianglesEmptySlots_compact
(
	TStatus *d_TStatus, 
	int *d_prefix, 
	int *d_empty, 
	int *d_trianglelist,      
	int *d_neighborlist,
	int numberofprocess,
	int poffset,
	int numberofelements_new, 
	int offset
 )
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofprocess) 
        return ;

	if (d_TStatus[pos+poffset].isNull())
	{
		d_prefix[pos+poffset] = -1; // indicates it is empty
		return;
	}

    int value;

    if ( pos+poffset < numberofelements_new ) 
        value = pos+poffset; 
    else 
	{
        value = d_empty[d_prefix[pos+poffset] - offset]; 

        for ( int i = 0; i < 3; i++ )
		{
            d_trianglelist[value * 3 + i] = d_trianglelist[(pos+poffset) * 3 + i]; 
			d_neighborlist[value * 3 + i] = d_neighborlist[(pos+poffset) * 3 + i]; 
		}
    }        

    d_prefix[pos+poffset] = value; 
}

__global__ void kernelFixNeighbors_compact
(
 int	*d_neighborlist, 
 int	*d_newIndex, 
 int	numberofprocess,
 int	poffset
 ) 
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( pos >= numberofprocess )
        return ;

	int triIndex = pos+poffset;

    if ( d_neighborlist[3*triIndex] >= 0 )
        d_neighborlist[3*triIndex] = 
			encode_tri(d_newIndex[decode_tri(d_neighborlist[3*triIndex])], decode_ori(d_neighborlist[3*triIndex])); 

    if ( d_neighborlist[3*triIndex+1] >= 0 )
        d_neighborlist[3*triIndex+1] = 
			encode_tri(d_newIndex[decode_tri(d_neighborlist[3*triIndex+1])], decode_ori(d_neighborlist[3*triIndex+1])); 

    if ( d_neighborlist[3*triIndex+2] >= 0 )
        d_neighborlist[3*triIndex+2] = 
			encode_tri(d_newIndex[decode_tri(d_neighborlist[3*triIndex+2])], decode_ori(d_neighborlist[3*triIndex+2])); 
      
}

__global__ void kernelFixTStatus_compact
(
 TStatus	*d_TStatus,
 int		*d_newIndex, 
 int		numberofprocess,
 int		poffset
 ) 
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( pos >= numberofprocess || d_newIndex[pos+poffset] == -1)
		return;

	int new_index = d_newIndex[pos+poffset];
	d_TStatus[new_index] = d_TStatus[pos+poffset];
}

__global__ void kernelFixSubseg2tri_compact
(
 int	*d_subseg2tri, 
 int	*d_newIndex, 
 int	numberofprocess,
 int	poffset
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( pos >= numberofprocess )
        return ;

	int old_triIndex = decode_tri(d_subseg2tri[pos+poffset]);
	int ori = decode_ori(d_subseg2tri[pos+poffset]);
	int new_triIndex = d_newIndex[old_triIndex];

	d_subseg2tri[pos+poffset] = encode_tri(new_triIndex,ori);
}

int compactTrianlgeList
(
 IntD		&t_trianglelist,  
 IntD       &t_neighborlist,
 TStatusD	&t_TStatus, 
 IntD		&t_subseg2tri,
 IntD       &t_valid,
 IntD       &t_prefix,
 int		numberoftriangles,
 int		numberofsubsegs,
 int		maxnumofblock
)
{
	int numberofprocess; // number of elements processed per iteration
	int finishnum; // numberof elements finish already

	t_valid.resize(numberoftriangles);
	t_prefix.resize(numberoftriangles);

	int numberofblocks;

	// Mark the valid triangles in the list
	finishnum = 0;
	do
	{
		if(numberoftriangles - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberoftriangles - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelMarkValidTriangles_compact<<< numberofblocks, BLOCK_SIZE >>>(
			thrust::raw_pointer_cast(&t_TStatus[0]), 
			thrust::raw_pointer_cast(&t_valid[0]), 
			numberofprocess,
			finishnum); 

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberoftriangles);

	// Compute the offset of them in the new list
	thrust::fill_n(t_prefix.begin(),numberoftriangles,0);
	thrust::exclusive_scan( t_valid.begin(), t_valid.begin() + numberoftriangles, t_prefix.begin() ); 

	int newnTris, lastitem, offset; 
	cudaMemcpy(&newnTris, thrust::raw_pointer_cast(&t_prefix[0]) + numberoftriangles - 1, sizeof(int), cudaMemcpyDeviceToHost); 
	cudaMemcpy(&lastitem, thrust::raw_pointer_cast(&t_valid[0]) + numberoftriangles - 1, sizeof(int), cudaMemcpyDeviceToHost); 
	newnTris += lastitem; 	

	if ( newnTris == numberoftriangles )
		return newnTris;

	cudaMemcpy(&offset, thrust::raw_pointer_cast(&t_prefix[0]) + newnTris, sizeof(int), cudaMemcpyDeviceToHost); 

	// Find all empty slots in the list
	finishnum = 0;
	do
	{
		if(numberoftriangles - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberoftriangles - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelCollectTrianglesEmptySlots_compact<<< numberofblocks, BLOCK_SIZE >>>(
			thrust::raw_pointer_cast(&t_TStatus[0]), 
			thrust::raw_pointer_cast(&t_prefix[0]), 
			thrust::raw_pointer_cast(&t_valid[0]), 
			numberofprocess,
			finishnum);

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberoftriangles);

	// Move those valid triangles at the end of the list
	// to the holes in the list. 
		// Find all empty slots in the list
	finishnum = 0;
	do
	{
		if(numberoftriangles - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberoftriangles - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);
		
		kernelFillTrianglesEmptySlots_compact<<< numberofblocks, BLOCK_SIZE  >>>(
			thrust::raw_pointer_cast(&t_TStatus[0]), 
			thrust::raw_pointer_cast(&t_prefix[0]), 
			thrust::raw_pointer_cast(&t_valid[0]), 
			thrust::raw_pointer_cast(&t_trianglelist[0]), 
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			numberofprocess,
			finishnum,
			newnTris, 
			offset); 

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberoftriangles);

	// Fix the neighbors after the index of our triangles are mixed up
	finishnum = 0;
	do
	{
		if(newnTris - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = newnTris - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelFixNeighbors_compact<<< numberofblocks, BLOCK_SIZE >>>(
			thrust::raw_pointer_cast(&t_neighborlist[0]), 
			thrust::raw_pointer_cast(&t_prefix[0]),
			numberofprocess,
			finishnum);

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < newnTris);

	// Fix TStatus
	finishnum = 0;
	do
	{
		if(numberoftriangles - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberoftriangles - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelFixTStatus_compact<<<numberofblocks, BLOCK_SIZE>>>(
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_prefix[0]),
			numberofprocess,
			finishnum);

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberoftriangles);

	// Fix subseg2tri
	finishnum = 0;
	do
	{
		if(numberofsubsegs - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberofsubsegs - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);
		
		kernelFixSubseg2tri_compact<<<numberofblocks, BLOCK_SIZE>>>(
			thrust::raw_pointer_cast(&t_subseg2tri[0]),
			thrust::raw_pointer_cast(&t_prefix[0]),
			numberofprocess,
			finishnum);

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberofsubsegs);

	return newnTris;

}


__global__ void kernelMarkValidPoints_compact
(
 PStatus	*d_PStatus, 
 int		*d_valid, 
 int		numberofprocess,
 int		poffset
 )
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( pos >= numberofprocess )
        return ;

	d_valid[pos+poffset] = (!d_PStatus[pos+poffset].isDeleted() ? 1 : 0); 

	// Also clear the 'new' status of the point
	//d_PStatus[pos].setOld(); 
}

__global__ void kernelCollectEmptyPointSlots_compact
(
 PStatus	*d_PStatus, 
 int		*d_prefix, 
 int		*d_empty, 
 int		numberofprocess,
 int		poffset
 )
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( pos >= numberofprocess )
        return ;

	if ( !d_PStatus[pos+poffset].isDeleted() ) 
	{	
		// Reset the 'new' status
        //d_PStatus[pos].setOld(); 
		return ;
	}

    int id = pos+poffset - d_prefix[pos+poffset]; 

    d_empty[id] = pos+poffset; 
}

__global__ void kernelFillEmptyPointSlots_compact
(
 PStatus	*d_PStatus, 
 int		*d_prefix, 
 int		*d_empty, 
 REAL2		*d_pointlist, 
 int		numberofprocess,
 int		poffset,
 int		newnPoints, 
 int		offset
 )
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if ( pos >= numberofprocess ) 
        return;

	if (d_PStatus[pos+poffset].isDeleted() )
	{
		d_prefix[pos+poffset] = -1;
		return; // indicate it is empty
	}

    int value;

    if (pos+poffset < newnPoints) 
        value = pos+poffset; 
    else 
	{
        value = d_empty[d_prefix[pos+poffset] - offset]; 		
		d_pointlist[value] = d_pointlist[pos+poffset]; 		
    }        

    d_prefix[pos+poffset] = value; 
}

__global__ void kernelUpdateTriangleList_compact
(
 int *d_trianglelist, 
 int *d_prefix, 
 int *d_valid, 
 int numberofprocess,
 int poffset,
 int newnPoints
 )
{
	 int pos = blockIdx.x * blockDim.x + threadIdx.x;
	 if ( pos >= numberofprocess )
		 return ; 

	 int p;

	 for ( int i = 0; i < 3; i++ )
	 {
		p = d_trianglelist[(pos+poffset)*3 + i];		
		if ( d_valid[p] == 1 || p >= newnPoints )
			d_trianglelist[(pos+poffset)*3 + i] = d_prefix[p];	
	 }
}

__global__ void kernelFixPStatus_compact
(
 PStatus	*d_PStatus,
 int		*d_newIndex, 
 int		numberofprocess,
 int		poffset
 ) 
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( pos >= numberofprocess || d_newIndex[pos+poffset] == -1)
		return;

	int new_index = d_newIndex[pos+poffset];
	d_PStatus[new_index] = d_PStatus[pos+poffset];
}

int compactPointList
(
 IntD		&t_trianglelist, 
 Real2D		&t_pointlist,  
 PStatusD	&t_PStatus,
 IntD       &t_valid,
 IntD       &t_prefix,
 int		numberoftriangles, 
 int		numberofpoints,
 int		maxnumofblock
 )
{
	int numberofprocess; // number of elements processed per iteration
	int finishnum; // numberof elements finish already

	t_valid.resize(numberofpoints);
	t_prefix.resize(numberofpoints);

	int numberofblocks;

	// Mark the valid points in the list
	finishnum = 0;
	do
	{
		if(numberofpoints - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberofpoints - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelMarkValidPoints_compact<<< numberofblocks, BLOCK_SIZE >>>(
			thrust::raw_pointer_cast(&t_PStatus[0]), 
			thrust::raw_pointer_cast(&t_valid[0]), 
			numberofprocess,
			finishnum); 

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberofpoints);

	// Compute the offset of them in the new list
	thrust::fill_n(t_prefix.begin(),numberofpoints,0);
	thrust::exclusive_scan( t_valid.begin(), t_valid.begin() + numberofpoints, t_prefix.begin() ); 

	int newnPoints, lastitem, offset; 
	cudaMemcpy(&newnPoints, thrust::raw_pointer_cast(&t_prefix[0]) + numberofpoints - 1, sizeof(int), cudaMemcpyDeviceToHost); 
	cudaMemcpy(&lastitem, thrust::raw_pointer_cast(&t_valid[0]) + numberofpoints - 1, sizeof(int), cudaMemcpyDeviceToHost); 
	newnPoints += lastitem; 

	if ( numberofpoints == newnPoints )
		return newnPoints;

	cudaMemcpy(&offset, thrust::raw_pointer_cast(&t_prefix[0]) + newnPoints, sizeof(int), cudaMemcpyDeviceToHost); 
  
	// Find all empty slots in the list
	finishnum = 0;
	do
	{
		if(numberofpoints - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberofpoints - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelCollectEmptyPointSlots_compact<<< numberofblocks, BLOCK_SIZE >>>(
			thrust::raw_pointer_cast(&t_PStatus[0]), 
			thrust::raw_pointer_cast(&t_prefix[0]), 
			thrust::raw_pointer_cast(&t_valid[0]),
			numberofprocess,
			finishnum); 

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberofpoints);

	// Move those valid points at the end of the list
	// to the holes in the list.
	finishnum = 0;
	do
	{
		if(numberofpoints - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberofpoints - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);
	
		kernelFillEmptyPointSlots_compact<<< numberofblocks, BLOCK_SIZE >>>(
			thrust::raw_pointer_cast(&t_PStatus[0]), 
			thrust::raw_pointer_cast(&t_prefix[0]), 
			thrust::raw_pointer_cast(&t_valid[0]),
			thrust::raw_pointer_cast(&t_pointlist[0]), 
			numberofprocess,
			finishnum,
			newnPoints,
			offset); 

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberofpoints);
	
	// Fix triangle list
	finishnum = 0;
	do
	{
		if(numberoftriangles - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberoftriangles - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);
	
		kernelUpdateTriangleList_compact<<<numberofblocks, BLOCK_SIZE>>>(
			thrust::raw_pointer_cast(&t_trianglelist[0]), 
			thrust::raw_pointer_cast(&t_prefix[0]), 
			thrust::raw_pointer_cast(&t_valid[0]),
			numberofprocess,
			finishnum,
			newnPoints);

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberoftriangles);

	// Fix PStatus
	finishnum = 0;
	do
	{
		if(numberofpoints - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberofpoints - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelFixPStatus_compact<<<numberofblocks, BLOCK_SIZE>>>(
			thrust::raw_pointer_cast(&t_PStatus[0]),
			thrust::raw_pointer_cast(&t_prefix[0]),
			numberofprocess,
			finishnum);

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberofpoints);

	return newnPoints; 
}

__global__ void kernelCompactSegment
(
 int *d_trianglelist,
 int *d_subseg2tri,
 int *d_segmentlist,
 int numberofprocess,
 int poffset
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if ( pos >= numberofprocess )
		return ; 

	int otri = d_subseg2tri[pos+poffset];
	int triIndex = decode_tri(otri);
	int triOri = decode_ori(otri);
	int p[] = {
		d_trianglelist[3*triIndex + (triOri+1)%3],
		d_trianglelist[3*triIndex + (triOri+2)%3]
	};

	d_segmentlist[2*(pos+poffset)] = p[0];
	d_segmentlist[2*(pos+poffset)+1] = p[1];
}

void compactSegmentList
(
 IntD &t_trianglelist,
 IntD &t_subseg2tri,
 IntD &t_segmentlist,
 int numberofsubsegs,
 int maxnumofblock
)
{
	t_segmentlist.resize(2*numberofsubsegs);

	int numberofprocess; // number of elements processed per iteration
	int finishnum; // numberof elements finish already

	int numberofblocks;

	// Convert subseg2tri to segment list
	finishnum = 0;
	do
	{
		if(numberofsubsegs - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberofsubsegs - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelCompactSegment<<<numberofblocks,BLOCK_SIZE>>>(
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_subseg2tri[0]),
			thrust::raw_pointer_cast(&t_segmentlist[0]),
			numberofprocess,
			finishnum);

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberofsubsegs);
}

__global__ void kernelUpdateNeighborsFormat2Int(
	int * d_neighborlist,
	int numberofprocess,
	int offset
)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if(pos >= numberofprocess)
		return;

	int triIndex = pos + offset;

	int neighbors[3] = {
		d_neighborlist[3*triIndex],
		d_neighborlist[3*triIndex+1],
		d_neighborlist[3*triIndex+2]
	};

	for(int i=0; i<3; i++)
		d_neighborlist[3*triIndex + i] = decode_tri(neighbors[i]);
}

void updateNeighborsFormat2Int
(
 int *d_neighborlist,
 int numberoftriangles,
 int maxnumofblock
 )
{
	int numberofprocess; // number of triangles processed per iteration
	int finishnum = 0; // numberof triangles finish already

	int numberofblocks;

	do
	{
		if(numberoftriangles - finishnum < maxnumofblock*BLOCK_SIZE)
			numberofprocess = numberoftriangles - finishnum;
		else
			numberofprocess = maxnumofblock*BLOCK_SIZE;

		numberofblocks = (ceil)((float)(numberofprocess) / BLOCK_SIZE);

		kernelUpdateNeighborsFormat2Int<<<numberofblocks, BLOCK_SIZE>>>(
			d_neighborlist,
			numberofprocess,
			finishnum);

		cudaDeviceSynchronize();
		finishnum += numberofprocess;

	}while(finishnum < numberoftriangles);


}

/**																					**/
/**																					**/
/********* Compact routine end here											 *********/

/********* GPU main routine begin here										 *********/
/**																					**/
/**																					**/

/*************************************************************************************/
/*																					 */
/*  GPU_Refine_Quality()   Compute quality mesh on GPU.								 */
/*                                                                                   */
/*************************************************************************************/


/**																					**/
/**																					**/
/********* GPU main routine end here										 *********/