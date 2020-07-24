#include "cudaFlipFlop.h"

void flipFlop
(
	Real2D			&t_pointlist,
	PStatusD		&t_PStatus,
	IntD			&t_trianglelist,
	IntD           &t_neighborlist,
	IntD			&t_tri2subseg,
	TStatusD		&t_TStatus,
	IntD			&t_subseg2tri,
	IntD           &t_flipBy,
	IntD           &t_active,
	IntD			&t_encmarker,
	IntD			&t_linklist,
	IntD			&t_linkslot,
	int			numberoftriangles,
	int encmode,
	REAL theta,
	int iteration0,
	int iteration1
)
{
	int numberofactive = 0;
	int numberofwinner = 0;
	int numberofnegate = 0;
	int step = 0;

	int numberofblocks;

	do
	{
		// update active list to triangles that need to check flipflop
		numberofactive = updateActiveListByFlipMarker(t_TStatus, t_active, numberoftriangles);

#ifdef GQM2D_DEBUG_2
		printf(" numberofactive = %d\n", numberofactive);
#endif

		if (numberofactive == 0)
			break;

		// init t_flipBy and flipOri
		thrust::fill_n(t_flipBy.begin(), numberoftriangles, MAXINT);
		numberofblocks = (ceil)((float)numberoftriangles / BLOCK_SIZE);
		kernelInitFlipOri << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_TStatus[0]),
			numberoftriangles);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// check flipping
		numberofblocks = (ceil)((float)numberofactive / BLOCK_SIZE);
		kernelCheckFlipping << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_PStatus[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_flipBy[0]),
			thrust::raw_pointer_cast(&t_active[0]),
			numberofactive);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// mark winner and update active list to winner
		numberofblocks = (ceil)((float)numberofactive / BLOCK_SIZE);
		kernelMarkFlipWinner << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_PStatus[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_flipBy[0]),
			thrust::raw_pointer_cast(&t_active[0]),
			numberofactive);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		numberofwinner = updateActiveListByFlipWinner(t_TStatus, t_active, numberoftriangles, numberofactive);

#ifdef GQM2D_DEBUG_2
		printf(" numberofwinner = %d\n", numberofwinner);
#endif
		if (numberofwinner > 0)
		{
			// set up neighbor link (store neighbor information temporarily)
			t_linklist.resize(4 * numberofwinner);
			t_linkslot.resize(4 * numberofwinner * 3);
			thrust::fill(t_linklist.begin(), t_linklist.end(), -1);

			numberofblocks = (ceil)((float)numberofwinner / BLOCK_SIZE);
			kernelInitLinklist << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				thrust::raw_pointer_cast(&t_linklist[0]),
				numberofwinner,
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				iteration0,
				iteration1,
				step);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// update neighbor information: 
			// phase 1
			numberofblocks = (ceil)((float)numberofwinner / BLOCK_SIZE);
			kernelUpdatePhase1 << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				thrust::raw_pointer_cast(&t_linkslot[0]),
				numberofwinner,
				iteration0,
				iteration1,
				step);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// phase 2
			numberofblocks = (ceil)((float)numberofwinner / BLOCK_SIZE);
			kernelUpdatePhase2 << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				thrust::raw_pointer_cast(&t_linklist[0]),
				thrust::raw_pointer_cast(&t_linkslot[0]),
				numberofwinner,
				iteration0);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// phase 3
			numberofblocks = (ceil)((float)numberofwinner / BLOCK_SIZE);
			kernelUpdatePhase3 << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				thrust::raw_pointer_cast(&t_encmarker[0]),
				thrust::raw_pointer_cast(&t_linklist[0]),
				thrust::raw_pointer_cast(&t_linkslot[0]),
				numberofwinner,
				encmode,
				theta,
				iteration0,
				step,
				numberoftriangles);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif
		}

		// mark reduntant steiners and incident triangles
		numberofnegate = updateActiveListByFlipNegate(t_flipBy, t_active, numberoftriangles, numberofactive);
#ifdef GQM2D_DEBUG_2
		printf(" numberofnegate = %d\n", numberofnegate);
#endif
		if (numberofnegate > 0)
		{
			numberofblocks = (ceil)((float)numberofnegate / BLOCK_SIZE);
			kernelMarkReduntantSteiner << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				numberofnegate);
		}

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		step++;

	} while (numberofactive > 0);
}

// this flipFlop clears redudant points inside diametral circles
void flipFlop
(
	Real2D			&t_pointlist,
	PStatusD		&t_PStatus,
#ifdef GQM2D_PRIORITY_SIZE
	RealD			&t_pointpriority,
#endif
	IntD			&t_trianglelist,
	IntD           &t_neighborlist,
	IntD			&t_tri2subseg,
	TStatusD		&t_TStatus,
	IntD			&t_subseg2tri,
	IntD           &t_flipBy,
	IntD           &t_active,
	IntD			&t_encmarker,
	IntD            &t_segmarker,
	IntD			&t_linklist,
	IntD			&t_linkslot,
	int			numberoftriangles,
	int encmode,
	REAL theta,
	int iteration0,
	int iteration1
)
{
	int numberofactive = 0;
	int numberofwinner = 0;
	int numberofbadseg = 0;
	int numberofnegate = 0;
	int step = 0;

	int numberofblocks;

	do
	{
		// update active list to triangles that need to check flipflop
		numberofactive = updateActiveListByFlipMarker(t_TStatus, t_active, numberoftriangles);

#ifdef GQM2D_DEBUG_2
		printf(" numberofactive = %d\n", numberofactive);
#endif

		if (numberofactive == 0)
			break;

		// init t_flipBy and flipOri
		thrust::fill_n(t_flipBy.begin(), numberoftriangles, MAXINT);
		numberofblocks = (ceil)((float)numberoftriangles / BLOCK_SIZE);
		kernelInitFlipOri << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_TStatus[0]),
			numberoftriangles);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// check flipping
		numberofblocks = (ceil)((float)numberofactive / BLOCK_SIZE);
#ifdef GQM2D_PRIORITY_SIZE
		kernelCheckFlipping_New << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_PStatus[0]),
			thrust::raw_pointer_cast(&t_pointpriority[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_flipBy[0]),
			thrust::raw_pointer_cast(&t_active[0]),
			numberofactive);
#else
		kernelCheckFlipping << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_pointlist[0]),
			thrust::raw_pointer_cast(&t_PStatus[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_tri2subseg[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_flipBy[0]),
			thrust::raw_pointer_cast(&t_active[0]),
			numberofactive);
#endif

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		// mark winner and update active list to winner
		numberofblocks = (ceil)((float)numberofactive / BLOCK_SIZE);
		kernelMarkFlipWinner << <numberofblocks, BLOCK_SIZE >> >(
			thrust::raw_pointer_cast(&t_PStatus[0]),
			thrust::raw_pointer_cast(&t_trianglelist[0]),
			thrust::raw_pointer_cast(&t_neighborlist[0]),
			thrust::raw_pointer_cast(&t_TStatus[0]),
			thrust::raw_pointer_cast(&t_flipBy[0]),
			thrust::raw_pointer_cast(&t_active[0]),
			numberofactive);

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		numberofwinner = updateActiveListByFlipWinner(t_TStatus, t_active, numberoftriangles, numberofactive);

#ifdef GQM2D_DEBUG_2
		printf(" numberofwinner = %d\n", numberofwinner);
#endif
		if (numberofwinner > 0)
		{
			// set up neighbor link (store neighbor information temporarily)
			t_linklist.resize(4 * numberofwinner);
			t_linkslot.resize(4 * numberofwinner * 3);
			thrust::fill(t_linklist.begin(), t_linklist.end(), -1);

			numberofblocks = (ceil)((float)numberofwinner / BLOCK_SIZE);
			kernelInitLinklist << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				thrust::raw_pointer_cast(&t_linklist[0]),
				numberofwinner,
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				iteration0,
				iteration1,
				step);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// update neighbor information: 
			// phase 1
			numberofblocks = (ceil)((float)numberofwinner / BLOCK_SIZE);
			kernelUpdatePhase1 << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				thrust::raw_pointer_cast(&t_linkslot[0]),
				numberofwinner,
				iteration0,
				iteration1,
				step);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// phase 2
			numberofblocks = (ceil)((float)numberofwinner / BLOCK_SIZE);
			kernelUpdatePhase2 << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				thrust::raw_pointer_cast(&t_linklist[0]),
				thrust::raw_pointer_cast(&t_linkslot[0]),
				numberofwinner,
				iteration0);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// phase 3
			numberofblocks = (ceil)((float)numberofwinner / BLOCK_SIZE);
			kernelUpdatePhase3_New << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				thrust::raw_pointer_cast(&t_encmarker[0]),
				thrust::raw_pointer_cast(&t_segmarker[0]),
				thrust::raw_pointer_cast(&t_linklist[0]),
				thrust::raw_pointer_cast(&t_linkslot[0]),
				numberofwinner,
				encmode,
				theta);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif
		}

		// mark steiners inside diametral circles as redundant and
		// mark incident triangles as active
		numberofbadseg = updateActiveListByMarker_Slot(t_segmarker, t_active, t_segmarker.size());
#ifdef GQM2D_DEBUG_2
		printf(" numberofbadseg = %d\n", numberofbadseg);
#endif
		if (numberofbadseg > 0)
		{
			numberofblocks = (ceil)((float)numberofbadseg / BLOCK_SIZE);
			kernelMarkDiametralCircle << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				numberofbadseg);
		}

		// mark dependent steiners as redundant and mark incident triangles as active
		numberofnegate = updateActiveListByFlipNegate(t_flipBy, t_active, numberoftriangles, numberofactive);
#ifdef GQM2D_DEBUG_2
		printf(" numberofnegate = %d\n", numberofnegate);
#endif
		if (numberofnegate > 0)
		{
			numberofblocks = (ceil)((float)numberofnegate / BLOCK_SIZE);
			kernelMarkReduntantSteiner << <numberofblocks, BLOCK_SIZE >> > (
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_active[0]),
				numberofnegate);
		}

#ifdef GQM2D_DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif

		step++;

	} while (numberofactive > 0);
}