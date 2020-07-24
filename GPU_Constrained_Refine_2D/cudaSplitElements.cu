#include "cudaSplitElements.h"
#include "cudaSplitEncsegs.h"
#include "cudaFlipFlop.h"
#include "cudaMesh.h"

void splitElements(
	Real2D	 &t_pointlist,
	PStatusD &t_PStatus,
	IntD	 &t_segmentlist,
	IntD	 &t_subseg2tri,
	IntD	 &t_subseg2seg,
	IntD	 &t_encmarker,
	IntD	 &t_trianglelist,
	IntD	 &t_neighborlist,
	IntD	 &t_tri2subseg,
	TStatusD &t_TStatus,
	IntD	 &t_emptypoints,
	IntD	 &t_emptytriangles,
	int pointblock,
	int triblock,
	int * numberofemptypoints,
	int * numberofemptytriangles,
	int * numberofpoints,
	int	* numberoftriangles,
	int	* numberofsubseg,
	int offconstant,
	int offcenter,
	int encmode,
	int filtermode,
	int unifymode,
	REAL theta,
	REAL size
)
{
	printf("Splitting bad elements....\n");

	IntD t_badelementlist;
	IntD t_badsubseglist, t_badtrianglelist;
	IntD t_threadlist;

#ifdef GQM2D_PRIORITY_SIZE
	RealD t_pointpriority(t_PStatus.size(), 0.0);
	UInt64D t_trimarker(t_TStatus.size());
#else
	IntD t_trimarker(t_TStatus.size());
#endif
	IntD t_flipBy(t_TStatus.size());

	// Record subsegments that were encroached
	// and are not encroached because their
	// diametral circles have been cleared.
	// Such subsegments need to be split
	IntD t_segmarker(*numberofsubseg, -1);

	Real2D t_insertpt;
	IntD t_sinks;
#ifdef GQM2D_PRIORITY_SIZE
	RealD t_priorityreal;
	IntD t_priority;
#endif

	int numberofbadelements;
	int numberofbadsubsegs, numberofbadtriangles;

	int numberofwonsegs;
	int numberofsteiners;
	int numberofonsegs;

	int numberofblocks;

	int iteration = 0;
	int iter_numofpt = thrust::count_if(t_PStatus.begin(), t_PStatus.end(), isNotDeleted());
#ifndef GQM2D_QUIET
	printf("  iteration -1: number of points = %d, 0\n", iter_numofpt);
#endif

#ifdef GQM2D_CHECKMEMORY
	cudaDeviceSynchronize();
	gpuMemoryCheck();
#endif

	StopWatchInterface *iter_timer = 0;
	sdkCreateTimer(&iter_timer);

	// Reset timer
	cudaDeviceSynchronize();
	sdkResetTimer(&iter_timer);
	sdkStartTimer(&iter_timer);

#ifdef GQM2D_ITER_PROFILING
	clock_t tv[2];
	int npt[2];
	tv[0] = clock();
#endif

	while (true)
	{
#ifdef GQM2D_ITER_PROFILING
		npt[0] = thrust::count_if(t_PStatus.begin(), t_PStatus.end(), isNotDeleted());
#endif
		// Compute bad element list
		numberofbadsubsegs = updateActiveListByMarker_Slot(t_encmarker, t_badsubseglist, *numberofsubseg);
		numberofbadtriangles = 
			updateActiveListToBadTriangles(
				t_pointlist,
				t_PStatus,
				t_trianglelist,
				t_neighborlist,
				t_segmentlist,
				t_subseg2seg,
				t_tri2subseg,
				t_TStatus,
				t_threadlist, // temporarily used
				t_badtrianglelist,
				*numberoftriangles,
				theta,
				size);

		if (unifymode == 0) // do not split subsegments and triangles together
		{
			if (numberofbadsubsegs > 0)
				numberofbadtriangles = 0;
		}

		numberofbadelements = numberofbadsubsegs + numberofbadtriangles;
		if (numberofbadelements == 0)
			break;

#ifndef GQM2D_QUIET
		printf("  \niteration %d: #%d bad elements (#%d subsegs, #%d triangles)\n",
			iteration, numberofbadelements, numberofbadsubsegs, numberofbadtriangles);
#endif

#ifdef GQM2D_CHECKMEMORY
		cudaDeviceSynchronize();
		gpuMemoryCheck();
#endif

		t_badelementlist.resize(numberofbadelements);
		thrust::copy_n(t_badsubseglist.begin(), numberofbadsubsegs, t_badelementlist.begin());
		thrust::copy_n(t_badtrianglelist.begin(), numberofbadtriangles, t_badelementlist.begin() + numberofbadsubsegs);

		t_insertpt.resize(numberofbadelements);
		t_sinks.resize(numberofbadelements);
#ifdef GQM2D_PRIORITY_SIZE
		t_priorityreal.resize(numberofbadelements);
		t_priority.resize(numberofbadelements);
#endif

		// Compute splitting points and priorites
		numberofblocks = (ceil)((float)numberofbadelements / BLOCK_SIZE);
		kernelComputeSplittingPointAndPriority << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_subseg2tri[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_badelementlist[0]),
			numberofbadelements,
			numberofbadsubsegs,
			thrust::raw_pointer_cast(&t_insertpt[0]),
#ifdef GQM2D_PRIORITY_SIZE
			thrust::raw_pointer_cast(&t_priorityreal[0]),
#endif
			offconstant,
			offcenter);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

#ifdef GQM2D_PRIORITY_SIZE

		// Modify priorities and convert them into integers
		// Make sure subseg > triangle
		double priority_min[2], priority_max[2], priority_offset[2] = { 0, 0 };
		thrust::pair<RealD::iterator, RealD::iterator> priority_pair;
		if (numberofbadtriangles > 0)
		{
			priority_pair =
				thrust::minmax_element(
					t_priorityreal.begin() + numberofbadsubsegs,
					t_priorityreal.end());
			priority_min[1] = *priority_pair.first;
			priority_max[1] = *priority_pair.second;
			priority_offset[1] = 0;
#ifdef GQM2D_DEBUG
			printf("MinMax Real priorities for triangles: %lf, %lf\n", priority_min[1], priority_max[1]);
			printf("Offset: %lf\n", priority_offset[1]);
#endif
		}

		if (numberofbadsubsegs > 0)
		{
			priority_pair =
				thrust::minmax_element(
					t_priorityreal.begin(),
					t_priorityreal.begin() + numberofbadsubsegs);
			priority_min[0] = *priority_pair.first;
			priority_max[0] = *priority_pair.second;
			if (numberofbadtriangles > 0)
				priority_offset[0] = priority_max[1] + priority_offset[1] + 10 - priority_min[0];
			else
				priority_offset[0] = 0;
#ifdef GQM2D_DEBUG
			printf("MinMax Real priorities for subsegs: %lf, %lf\n", priority_min[0], priority_max[0]);
			printf("Offset: %lf\n", priority_offset[0]);
#endif
		}

		kernelModifyPriority << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_priorityreal[0]),
			thrust::raw_pointer_cast(&t_priority[0]),
			priority_offset[0],
			priority_offset[1],
			thrust::raw_pointer_cast(&t_badelementlist[0]),
			numberofbadelements,
			numberofbadsubsegs);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
		if (numberofbadsubsegs > 0)
		{
			priority_pair =
				thrust::minmax_element(
					t_priorityreal.begin(),
					t_priorityreal.begin() + numberofbadsubsegs);
			priority_min[0] = *priority_pair.first;
			priority_max[0] = *priority_pair.second;
			printf("MinMax Real priorities for subsegs after modification: %lf, %lf\n", priority_min[0], priority_max[0]);
		}

		if (numberofbadtriangles > 0)
		{
			priority_pair =
				thrust::minmax_element(
					t_priorityreal.begin() + numberofbadsubsegs,
					t_priorityreal.end());
			priority_min[1] = *priority_pair.first;
			priority_max[1] = *priority_pair.second;
			printf("MinMax Real priorities for triangles after modification: %lf, %lf\n", priority_min[1], priority_max[1]);
		}
#endif

#endif
		// Locate splitting and do marking competition
#ifdef GQM2D_PRIORITY_SIZE
		thrust::fill(t_trimarker.begin(), t_trimarker.begin() + *numberoftriangles, 0);
#else
		thrust::fill(t_trimarker.begin(), t_trimarker.begin() + *numberoftriangles, MAXINT);
#endif

		kernelLocateSplittingPoints << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_subseg2tri[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_insertpt[0]),
			thrust::raw_pointer_cast(&t_trimarker[0]),
#ifdef GQM2D_PRIORITY_SIZE
			thrust::raw_pointer_cast(&t_priority[0]),
#endif
			thrust::raw_pointer_cast(&t_sinks[0]),
			thrust::raw_pointer_cast(&t_badelementlist[0]),
			numberofbadelements,
			numberofbadsubsegs);

#ifdef GQM2D_DEBUG
		//printf("number of winner after kernelLocateSplittingPoints = %d\n", 
		//	thrust::count_if(t_badelementlist.begin(), t_badelementlist.end(), isNotNegativeInt()));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		if (filtermode)
		{
			// Fast cavities checking to avoid unnecessary point insertions
			// and encroachment

			kernelFastCavityCheck << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_insertpt[0]),
				thrust::raw_pointer_cast(&t_trimarker[0]),
#ifdef GQM2D_PRIORITY_SIZE
				thrust::raw_pointer_cast(&t_priority[0]),
#endif
				thrust::raw_pointer_cast(&t_sinks[0]),
				thrust::raw_pointer_cast(&t_badelementlist[0]),
				numberofbadelements);

#ifdef GQM2D_DEBUG
			//printf("number of winner after kernelFastCavityCheck = %d\n",
			//	thrust::count_if(t_badelementlist.begin(), t_badelementlist.end(), isNotNegativeInt()));
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif
		}

		// Mark subsegs who contain splitting points of winners as bad subsegs in t_segmarker
		kernelMarkOnSegs << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_segmarker[0]),
			thrust::raw_pointer_cast(&t_sinks[0]),
			thrust::raw_pointer_cast(&t_badelementlist[0]),
			numberofbadelements,
			numberofbadsubsegs);

#ifdef GQM2D_DEBUG
		//printf("number of winner after kernelMarkOnSegs = %d\n",
		//	thrust::count_if(t_badelementlist.begin(), t_badelementlist.end(), isNotNegativeInt()));
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		numberofwonsegs = thrust::count_if(t_badelementlist.begin(), t_badelementlist.begin() + numberofbadsubsegs, isNotNegativeInt());
		numberofsteiners = updateActiveListByMarker_Slot(t_badelementlist, t_threadlist, numberofbadelements);
#ifdef GQM2D_DEBUG
		printf("numberofsteiners = %d, numberofwonsegs = %d\n", numberofsteiners, numberofwonsegs);
#endif

		if (numberofsteiners == 0)
			goto END;

		// Prepare memory for new elements
		if (numberofsteiners > *numberofemptypoints)
		{
			*numberofemptypoints = updateEmptyPoints(t_PStatus, t_emptypoints);
			int num = 0;
			while (numberofsteiners > *numberofemptypoints + num*pointblock)
				num++;
			if (num != 0)
			{
				int old_size = t_PStatus.size();
				PStatus emptyPoint;
				emptyPoint.setDeleted();
				t_pointlist.resize(old_size + num*pointblock);
				t_PStatus.resize(old_size + num*pointblock, emptyPoint);
#ifdef GQM2D_PRIORITY_SIZE
				t_pointpriority.resize(old_size + num*pointblock, 0.0);
#endif
				*numberofemptypoints = updateEmptyPoints(t_PStatus, t_emptypoints);
			}
		}

		if (2 * numberofsteiners > *numberofemptytriangles)
		{
			*numberofemptytriangles = updateEmptyTriangles(t_TStatus, t_emptytriangles);
			int num = 0;
			while (2 * numberofsteiners > *numberofemptytriangles + num*triblock)
				num++;
			if (num != 0)
			{
				int old_size = t_TStatus.size();
				TStatus emptyTri(true, false, false);
				t_trianglelist.resize(3 * (old_size + num*triblock));
				t_neighborlist.resize(3 * (old_size + num*triblock));
				t_tri2subseg.resize(3 * (old_size + num*triblock), -1);
				t_TStatus.resize(old_size + num*triblock, emptyTri);
				t_trimarker.resize(old_size + num*triblock);
				t_flipBy.resize(old_size + num*triblock);
				*numberofemptytriangles = updateEmptyTriangles(t_TStatus, t_emptytriangles);
			}
		}

		t_subseg2tri.resize(*numberofsubseg + numberofwonsegs);
		t_subseg2seg.resize(*numberofsubseg + numberofwonsegs);
		t_encmarker.resize(*numberofsubseg + numberofwonsegs, -1);
		t_segmarker.resize(*numberofsubseg + numberofwonsegs, -1);

		// Insert splitting point
		numberofblocks = (ceil)((float)numberofsteiners / BLOCK_SIZE);

		kernelResetNeighborMarker << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_sinks[0]),
			thrust::raw_pointer_cast(&t_threadlist[0]),
			numberofsteiners);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		kernelInsertSplittingPoints << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_PStatus[0]),
#ifdef GQM2D_PRIORITY_SIZE
			thrust::raw_pointer_cast(&t_pointpriority[0]),
			thrust::raw_pointer_cast(&t_priorityreal[0]),
#endif
			thrust::raw_pointer_cast(&t_subseg2tri[0]),
			thrust::raw_pointer_cast(&t_subseg2seg[0]),
			thrust::raw_pointer_cast(&t_encmarker[0]),
			thrust::raw_pointer_cast(&t_segmarker[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_insertpt[0]),
			thrust::raw_pointer_cast(&t_sinks[0]),
			thrust::raw_pointer_cast(&t_emptypoints[0]),
			thrust::raw_pointer_cast(&t_emptytriangles[0]),
			thrust::raw_pointer_cast(&t_badelementlist[0]),
			thrust::raw_pointer_cast(&t_threadlist[0]),
			t_emptypoints.size(),
			t_emptytriangles.size(),
			*numberofemptypoints,
			*numberofemptytriangles,
			*numberofsubseg,
			numberofwonsegs,
			numberofsteiners,
			encmode,
			theta);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		kernelUpdateNeighbors << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_PStatus[0]),
			thrust::raw_pointer_cast(&t_subseg2tri[0]),
			thrust::raw_pointer_cast(&t_encmarker[0]),
			thrust::raw_pointer_cast(&t_segmarker[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_sinks[0]),
			thrust::raw_pointer_cast(&t_emptypoints[0]),
			thrust::raw_pointer_cast(&t_threadlist[0]),
			t_emptypoints.size(),
			*numberofemptypoints,
			numberofwonsegs,
			numberofsteiners,
			encmode,
			theta);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// Update iteration variables
		// (1) check if there are any slots before last points/triangles
		// (2) update last points/triangles/subsegs
		// (3) update number of empty points/triangles

		int slot_before, slot_after;

		// point slots
		slot_after = t_PStatus.size() - *numberofpoints;
		slot_before = *numberofemptypoints - slot_after;
		if (slot_before < numberofsteiners)
			*numberofpoints += numberofsteiners - slot_before;
		*numberofemptypoints -= numberofsteiners;

		// triangle slots
		slot_after = t_TStatus.size() - *numberoftriangles;
		slot_before = *numberofemptytriangles - slot_after;
		if (slot_before < 2 * numberofsteiners)
			*numberoftriangles += 2 * numberofsteiners - slot_before;
		*numberofemptytriangles -= 2 * numberofsteiners;

		// subseg
		*numberofsubseg += numberofwonsegs;

#ifdef GQM2D_DEBUG
		iter_numofpt = thrust::count_if(t_PStatus.begin(), t_PStatus.end(), isNotDeleted());
		printf(" Number of points before flipFlop = %d\n", iter_numofpt);
#endif

		// Maintain denauly property, do flip-flop
		flipFlop(
			t_pointlist,
			t_PStatus,
#ifdef GQM2D_PRIORITY_SIZE
			t_pointpriority,
#endif
			t_trianglelist,
			t_neighborlist,
			t_tri2subseg,
			t_TStatus,
			t_subseg2tri,
			t_flipBy, // flipBy
			t_sinks,    // flipActive
			t_encmarker,
			t_segmarker,
			t_threadlist, // linklist 
			t_badelementlist,   // linkslot
			*numberoftriangles,
			encmode,
			theta,
			-1,
			-1);

#ifdef GQM2D_DEBUG_2
		printf("non-negative in segmarker = %d\n", thrust::count_if(t_segmarker.begin(), t_segmarker.end(), isNotNegativeInt()));
#endif



		// mark bad subsegs as encroached subsegs using t_segmarker
END:	if (*numberofsubseg > 0)
		{
			numberofblocks = (ceil)((float)(*numberofsubseg) / BLOCK_SIZE);
			markBadSubsegsAsEncroached << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_encmarker[0]),
				thrust::raw_pointer_cast(&t_segmarker[0]),
				*numberofsubseg);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif
		}

		numberofblocks = (ceil)((float)(*numberofpoints) / BLOCK_SIZE);
		kernelUpdatePStatus2Old << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_PStatus[0]),
			*numberofpoints);

#ifdef GQM2D_DEBUG
		iter_numofpt = thrust::count_if(t_PStatus.begin(), t_PStatus.end(), isNotDeleted());
		printf(" Number of points after flipFlop = %d\n", iter_numofpt);
#endif

#ifdef GQM2D_ITER_PROFILING
		cudaDeviceSynchronize();
		tv[1] = clock();
		npt[1] = thrust::count_if(t_PStatus.begin(), t_PStatus.end(), isNotDeleted());
		printf("%d, %lf, %d, %d\n", iteration, (REAL)(tv[1] - tv[0]), numberofbadelements, npt[1] - npt[0]);
#endif

		iteration++;
	}

	// Get timer.
	cudaDeviceSynchronize();
	sdkStopTimer(&iter_timer);
	printf("1. Split bad elements time = %.3f ms\n", sdkGetTimerValue(&iter_timer));
}