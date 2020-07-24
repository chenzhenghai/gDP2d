#include "cudaSplitTriangles.h"
#include "cudaSplitEncsegs.h"
#include "cudaFlipFlop.h"
#include "cudaMesh.h"

void splitTriangles(
	Real2D	 &t_pointlist,
	PStatusD &t_PStatus,
	IntD	 &t_trianglelist,
	IntD	 &t_neighborlist,
	IntD	 &t_tri2subseg,
	TStatusD &t_TStatus,
	IntD	 &t_subseg2tri,
	IntD	 &t_encmarker,
	IntD	 &t_enclist,
	IntD	 &t_internalmarker,
	IntD	 &t_internallist,
	Real2D	 &t_TCenter,
	IntD	 &t_sinks,
	IntD	 &t_Priority,
	IntD	 &t_trimarker,
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
	REAL theta,
	int debug_iter
)
{
	int numberofbad;
	int numberofwinner;
	int numberofsteiner;
	int numberofonsegs;

	int numberofblocks;

#ifdef GQM2D_PRIORITY_SIZE
	UInt64D t_trimarker64(t_TStatus.size());
#endif

	// Compute circumcenter
	// t_internalmarker is bad triangles list initially
	numberofbad = t_internalmarker.size();
	t_internallist.resize(numberofbad);
	t_TCenter.resize(numberofbad);
	t_sinks.resize(numberofbad);
	t_Priority.resize(numberofbad);

	thrust::copy(t_internalmarker.begin(), t_internalmarker.end(), t_internallist.begin());

	// t_internallist stores the indices for bad triangles

	numberofblocks = (ceil)((float)numberofbad / BLOCK_SIZE);
	kernelComputeCircumcenter << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_internallist[0]),
		numberofbad,
		thrust::raw_pointer_cast(&t_TCenter[0]),
		thrust::raw_pointer_cast(&t_Priority[0]),
		offconstant,
		offcenter);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// Locate sinks
	// Use t_trimaker as marker for locating
#ifdef GQM2D_PRIORITY_SIZE
	thrust::fill(t_trimarker64.begin(), t_trimarker64.begin() + *numberoftriangles, 0);
#else
	thrust::fill(t_trimarker.begin(), t_trimarker.begin() + *numberoftriangles, MAXINT);
#endif

	numberofblocks = (ceil)((float)numberofbad / BLOCK_SIZE);
	kernelLocateSinkPoints << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_PStatus[0]),
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tri2subseg[0]),
		thrust::raw_pointer_cast(&t_TStatus[0]),
		thrust::raw_pointer_cast(&t_TCenter[0]),
#ifdef GQM2D_PRIORITY_SIZE
		thrust::raw_pointer_cast(&t_trimarker64[0]),
#else
		thrust::raw_pointer_cast(&t_trimarker[0]),
#endif
		thrust::raw_pointer_cast(&t_sinks[0]),
		thrust::raw_pointer_cast(&t_Priority[0]),
		thrust::raw_pointer_cast(&t_internalmarker[0]),
		thrust::raw_pointer_cast(&t_internallist[0]),
		numberofbad);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	numberofblocks = (ceil)((float)numberofbad / BLOCK_SIZE);
	kernelRemoveLosers << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_TStatus[0]),
#ifdef GQM2D_PRIORITY_SIZE
		thrust::raw_pointer_cast(&t_trimarker64[0]),
#else
		thrust::raw_pointer_cast(&t_trimarker[0]),
#endif
		thrust::raw_pointer_cast(&t_sinks[0]),
		thrust::raw_pointer_cast(&t_Priority[0]),
		thrust::raw_pointer_cast(&t_internalmarker[0]),
		thrust::raw_pointer_cast(&t_internallist[0]),
		numberofbad);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	//numberofwinner = updateActiveListByMarker_Slot(t_internalmarker,t_internallist,numberofbad);
	//printf("numberofwinner = %d\n",numberofwinner);

	// Independent Steiner points
	// (1). Mark cavities: use trimarker as marker
	// (2). Check cavities and remove dependent points (excluding incident cavities,
	// which will be removed in flip-flop stage)
#ifdef GQM2D_PRIORITY_SIZE
	thrust::fill(t_trimarker64.begin(), t_trimarker64.begin() + *numberoftriangles, 0);
#else
	thrust::fill(t_trimarker.begin(), t_trimarker.begin() + *numberoftriangles, MAXINT);
#endif

	markCavities(
		t_pointlist,
		t_trianglelist,
		t_neighborlist,
		t_tri2subseg,
		t_TStatus,
		t_TCenter,
		t_sinks,
		t_Priority,
#ifdef GQM2D_PRIORITY_SIZE
		t_trimarker64,
#else
		t_trimarker,
#endif
		t_internalmarker,
		t_internallist,
		numberofbad,
		2500);

	checkCavities(
		t_pointlist,
		t_trianglelist,
		t_neighborlist,
		t_tri2subseg,
		t_TStatus,
		t_TCenter,
		t_sinks,
		t_Priority,
#ifdef GQM2D_PRIORITY_SIZE
		t_trimarker64,
#else
		t_trimarker,
#endif
		t_internalmarker,
		t_internallist,
		numberofbad,
		2500);

	numberofsteiner = updateActiveListByMarker_Slot(t_internalmarker, t_internallist, numberofbad);
#ifdef GQM2D_DEBUG_2
	printf(" numberofsteiner = %d\n", numberofsteiner);
#endif

	// for steiners on segs, need to mark them as encroached segs
	// (1). mark these segs in enclist
	// (2). do flip-flop
	// (3). mark these segs in encmarker using enclist

	// initialized to mark the steiner points on segments
	t_enclist.resize(*numberofsubseg);
	thrust::fill(t_enclist.begin(), t_enclist.begin() + *numberofsubseg, -1);
	numberofblocks = (ceil)((float)numberofsteiner / BLOCK_SIZE);
	kernelMarkSteinerOnsegs << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_tri2subseg[0]),
		thrust::raw_pointer_cast(&t_enclist[0]),
		thrust::raw_pointer_cast(&t_internallist[0]),
		thrust::raw_pointer_cast(&t_sinks[0]),
		numberofsteiner);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// check if there is enough space
	// numberofsteiner points are going to be inserted
	// 2*numberofsteiner slots are needed
	if (numberofsteiner > *numberofemptypoints)
	{
		*numberofemptypoints = updateEmptyPoints(t_PStatus, t_emptypoints);
		int num = 0;
		while (numberofsteiner > *numberofemptypoints + num*pointblock)
			num++;
		if (num != 0)
		{
			int old_size = t_PStatus.size();
			PStatus emptyPoint;
			emptyPoint.setDeleted();
			t_pointlist.resize(old_size + num*pointblock);
			t_PStatus.resize(old_size + num*pointblock, emptyPoint);
			*numberofemptypoints = updateEmptyPoints(t_PStatus, t_emptypoints);
		}
	}

	if (2 * numberofsteiner > *numberofemptytriangles)
	{
		*numberofemptytriangles = updateEmptyTriangles(t_TStatus, t_emptytriangles);
		int num = 0;
		while (2 * numberofsteiner > *numberofemptytriangles + num*triblock)
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
			*numberofemptytriangles = updateEmptyTriangles(t_TStatus, t_emptytriangles);
		}
	}

	// insert steiner points
	numberofblocks = (ceil)((float)numberofsteiner / BLOCK_SIZE);
	kernelResetSteinerInsertionMarker << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_TStatus[0]),
		thrust::raw_pointer_cast(&t_internallist[0]),
		thrust::raw_pointer_cast(&t_sinks[0]),
		numberofsteiner);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	numberofblocks = (ceil)((float)numberofsteiner / BLOCK_SIZE);
	kernelInsertSteinerPoints << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_PStatus[0]),
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tri2subseg[0]),
		thrust::raw_pointer_cast(&t_TStatus[0]),
		thrust::raw_pointer_cast(&t_subseg2tri[0]),
		thrust::raw_pointer_cast(&t_encmarker[0]),
		thrust::raw_pointer_cast(&t_internalmarker[0]),
		thrust::raw_pointer_cast(&t_internallist[0]),
		thrust::raw_pointer_cast(&t_TCenter[0]),
		thrust::raw_pointer_cast(&t_sinks[0]),
		thrust::raw_pointer_cast(&t_emptypoints[0]),
		thrust::raw_pointer_cast(&t_emptytriangles[0]),
		t_emptypoints.size(),
		t_emptytriangles.size(),
		*numberofemptypoints,
		*numberofemptytriangles,
		*numberofsubseg,
		numberofsteiner,
		theta,
		debug_iter);

#ifdef GQM2D_DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	numberofblocks = (ceil)((float)numberofsteiner / BLOCK_SIZE);
	kernelUpdateSteinerNeighbors << <numberofblocks, BLOCK_SIZE >> >(
		thrust::raw_pointer_cast(&t_pointlist[0]),
		thrust::raw_pointer_cast(&t_trianglelist[0]),
		thrust::raw_pointer_cast(&t_neighborlist[0]),
		thrust::raw_pointer_cast(&t_tri2subseg[0]),
		thrust::raw_pointer_cast(&t_TStatus[0]),
		thrust::raw_pointer_cast(&t_subseg2tri[0]),
		thrust::raw_pointer_cast(&t_encmarker[0]),
		thrust::raw_pointer_cast(&t_internalmarker[0]),
		thrust::raw_pointer_cast(&t_internallist[0]),
		thrust::raw_pointer_cast(&t_sinks[0]),
		numberofsteiner,
		encmode,
		theta,
		debug_iter);

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
	if (slot_before < numberofsteiner)
		*numberofpoints += numberofsteiner - slot_before;
	*numberofemptypoints -= numberofsteiner;

	// triangle slots
	slot_after = t_TStatus.size() - *numberoftriangles;
	slot_before = *numberofemptytriangles - slot_after;
	if (slot_before < 2 * numberofsteiner)
		*numberoftriangles += 2 * numberofsteiner - slot_before;
	*numberofemptytriangles -= 2 * numberofsteiner;

	// check if encroachment markers are updated correctly
#ifdef GQM2D_DEBUG_3
	{
		printf("Iteration %d: After Insert steiner points\n",iteration);
		int * debug_em = new int[*numberofsubseg];
		int * debug_tl = new int[3 * (*numberoftriangles)];
		int * debug_nl = new int[3 * (*numberoftriangles)];
		REAL2 * debug_pl = new REAL2[*numberofpoints];
		int * debug_st = new int[*numberofsubseg];
		cudaMemcpy(debug_em, thrust::raw_pointer_cast(&t_encmarker[0]), sizeof(int)*(*numberofsubseg), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_nl, thrust::raw_pointer_cast(&t_neighborlist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)**numberofpoints, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_st, thrust::raw_pointer_cast(&t_subseg2tri[0]), sizeof(int)**numberofsubseg, cudaMemcpyDeviceToHost);

		for (int i = 0; i< *numberofsubseg; i++)
		{
			int otri = debug_st[i];
			int tri = otri >> 2;
			int ori = otri & 3;
			int p[3];
			REAL2 v[3];
			p[0] = debug_tl[3 * tri + (ori + 1) % 3];
			p[1] = debug_tl[3 * tri + (ori + 2) % 3];
			p[2] = debug_tl[3 * tri + ori];
			v[0] = debug_pl[p[0]];
			v[1] = debug_pl[p[1]];
			v[2] = debug_pl[p[2]];
			bool tag = false; // indicate if this segment is encroached or not

			REAL goodcoss = cos(theta * PI / 180.0);
			goodcoss *= goodcoss;
			REAL dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
				(v[0].y - v[2].y)*(v[1].y - v[2].y);

			if (dotproduct < 0.0) // angle > 90
			{
				// here, we use diametral lens to speedup the algorithm
				if (encmode || dotproduct * dotproduct >=
					(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
					((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
					((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
					tag = true;
			}

			otri = debug_nl[3 * tri + ori];
			if (otri != -1)
			{
				tri = otri >> 2;
				ori = otri & 3;
				p[2] = debug_tl[3 * tri + ori];
				v[2] = debug_pl[p[2]];
				dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
					(v[0].y - v[2].y)*(v[1].y - v[2].y);
				if (dotproduct < 0.0) // angle > 90
				{
					// here, we use diametral lens to speedup the algorithm
					if (encmode || dotproduct * dotproduct >=
						(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
						((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
						((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
						tag = true;
				}
			}

			if (debug_em[i] == -1 && tag)
				printf("Segment %d: I am encroached but marked as non-encroached\n", i);

			if (debug_em[i] == 0 && !tag)
				printf("Segment %d: I am not encroached but marked as encroached\n", i);

		}
		printf("Finish Checking\n");
		delete[] debug_em;
		delete[] debug_tl;
		delete[] debug_nl;
		delete[] debug_pl;
		delete[] debug_st;
	}
#endif

	// maintain denauly property, do flip-flop
	flipFlop(
		t_pointlist,
		t_PStatus,
		t_trianglelist,
		t_neighborlist,
		t_tri2subseg,
		t_TStatus,
		t_subseg2tri,
		t_trimarker, // flipBy
		t_sinks,    // flipActive
		t_encmarker,
		t_internalmarker, // linklist 
		t_internallist,   // linkslot
		*numberoftriangles,
		encmode,
		theta,
		-1,
		-1);

#ifdef GQM2D_DEBUG_3
	{
		//printf("Iteration %d: After Insert mid points\n",iteration);
		int * debug_em = new int[*numberofsubseg];
		int * debug_tl = new int[3 * (*numberoftriangles)];
		int * debug_nl = new int[3 * (*numberoftriangles)];
		REAL2 * debug_pl = new REAL2[*numberofpoints];
		int * debug_st = new int[*numberofsubseg];
		cudaMemcpy(debug_em, thrust::raw_pointer_cast(&t_encmarker[0]), sizeof(int)*(*numberofsubseg), cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_nl, thrust::raw_pointer_cast(&t_neighborlist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)**numberofpoints, cudaMemcpyDeviceToHost);
		cudaMemcpy(debug_st, thrust::raw_pointer_cast(&t_subseg2tri[0]), sizeof(int)**numberofsubseg, cudaMemcpyDeviceToHost);

		for (int i = 0; i< *numberofsubseg; i++)
		{
			int otri = debug_st[i];
			int tri = otri >> 2;
			int ori = otri & 3;
			int p[3];
			REAL2 v[3];
			p[0] = debug_tl[3 * tri + (ori + 1) % 3];
			p[1] = debug_tl[3 * tri + (ori + 2) % 3];
			p[2] = debug_tl[3 * tri + ori];
			v[0] = debug_pl[p[0]];
			v[1] = debug_pl[p[1]];
			v[2] = debug_pl[p[2]];
			bool tag = false; // indicate if this segment is encroached or not

			REAL goodcoss = cos(theta * PI / 180.0);
			goodcoss *= goodcoss;
			REAL dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
				(v[0].y - v[2].y)*(v[1].y - v[2].y);

			if (dotproduct < 0.0) // angle > 90
			{
				// here, we use diametral lens to speedup the algorithm
				if (encmode || dotproduct * dotproduct >=
					(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
					((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
					((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
					tag = true;
			}

			otri = debug_nl[3 * tri + ori];
			if (otri != -1)
			{
				tri = otri >> 2;
				ori = otri & 3;
				p[2] = debug_tl[3 * tri + ori];
				v[2] = debug_pl[p[2]];
				dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
					(v[0].y - v[2].y)*(v[1].y - v[2].y);
				if (dotproduct < 0.0) // angle > 90
				{
					// here, we use diametral lens to speedup the algorithm
					if (encmode || dotproduct * dotproduct >=
						(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
						((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
						((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
						tag = true;
				}
			}

			if (debug_em[i] == -1 && tag)
				printf(" Segment %d: I am encroached but marked as non-encroached\n", i);

			if (debug_em[i] == 0 && !tag)
				printf(" Segment %d: I am not encroached but marked as encroached\n", i);

		}
		//printf("Finish Checking\n");
	}
#endif

	numberofonsegs = updateActiveListByMarker_Slot(t_enclist, t_internallist, *numberofsubseg);
#ifdef GQM2D_DEBUG_2
	printf(" numberofonsegs = %d\n", numberofonsegs);
#endif
	if (numberofonsegs > 0)
	{
		numberofblocks = (ceil)((float)numberofonsegs / BLOCK_SIZE);
		kernelUpdateSteinerOnsegs << <numberofblocks, BLOCK_SIZE >> > (
			thrust::raw_pointer_cast(&t_encmarker[0]),
			thrust::raw_pointer_cast(&t_internallist[0]),
			numberofonsegs);
	}
}